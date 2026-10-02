# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The prompt-lookup kernel against a plain Python reference.

The kernel only fills the draft slots after MTP's, so a wrong slot never shows
in outputs, which verification keeps lossless: it only costs acceptance. It is
asserted here instead, on request rows held in UVA memory as the model runner
holds them.
"""

import random

import pytest
import torch

from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor
from vllm.v1.worker.gpu.spec_decode.gemma4.prompt_lookup import STATS, prompt_lookup

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA required for prompt lookup tests", allow_module_level=True)

MAX_REQS = 16
MAX_LEN = 8192
VOCAB = 40


def _reference(row, length, mtp, m, k, min_n, max_n):
    """The draft for one request, its extension flag, and whether a match
    exists."""
    seq = list(row[:length]) + list(mtp[:m])
    best = None
    for end in range(min_n, length):
        n = 0
        for i in range(max_n):
            if end - 1 - i < 0 or row[end - 1 - i] != seq[-1 - i]:
                break
            n = i + 1
        if n >= min_n and (best is None or (n, end) > best):
            best = (n, end)
    out = list(mtp[:m]) + [mtp[m - 1]] * (k - m)
    available = 0
    if best is not None:
        available = min(length - best[1], k - m)
        out[m : m + available] = row[best[1] : best[1] + available]
    return out, available > 0, best is not None


@pytest.mark.parametrize(("m", "k", "min_n", "max_n"), [(9, 15, 2, 8), (5, 11, 4, 6)])
@pytest.mark.parametrize("seed", range(20))
def test_prompt_lookup_matches_reference(seed, m, k, min_n, max_n):
    rng = random.Random(seed)
    tokens = StagedWriteTensor(
        (MAX_REQS, MAX_LEN), dtype=torch.int32, device="cuda", uva_instead_of_gpu=True
    )
    rows, lengths = [], []
    for r in range(MAX_REQS):
        length = rng.choice(
            [0, 1, 3, 9, rng.randint(10, 200), rng.randint(200, MAX_LEN)]
        )
        rows.append([rng.randrange(VOCAB) for _ in range(length)])
        lengths.append(length)
        if length:
            tokens.stage_write(r, 0, rows[-1])
    tokens.apply_write()
    total = torch.tensor(lengths, dtype=torch.int32, device="cuda")

    num_reqs = MAX_REQS - 2
    idx = list(range(MAX_REQS))
    rng.shuffle(idx)
    idx = idx[:num_reqs]
    idx[rng.randrange(num_reqs)] = -1
    drafts = []
    for r in idx:
        draft = [rng.randrange(VOCAB) for _ in range(k)]
        if r >= 0 and lengths[r] > 40 and rng.random() < 0.6:
            # MTP's draft copies an earlier span, as it does in copied text.
            start = rng.randrange(0, lengths[r] - m)
            draft[:m] = rows[r][start : start + m]
        drafts.append(draft)

    draft = torch.tensor(drafts, dtype=torch.int64, device="cuda")
    out = torch.zeros_like(draft)
    flags = [rng.randint(0, 1) for _ in range(MAX_REQS)]
    extended = torch.tensor(flags, dtype=torch.int32, device="cuda")
    sampled = [rng.choice([0, 1, rng.randint(1, k + 1)]) for _ in idx]
    rejected = [k + 1 - s if rng.random() < 0.7 else 0 for s in sampled]
    stats = torch.zeros(len(STATS), dtype=torch.int64, device="cuda")
    prompt_lookup(
        tokens.gpu,
        total,
        torch.tensor(idx, dtype=torch.int32, device="cuda"),
        draft,
        out,
        extended,
        torch.tensor(sampled, dtype=torch.int32, device="cuda"),
        torch.tensor(rejected, dtype=torch.int32, device="cuda"),
        stats,
        m,
        min_n,
        max_n,
    )

    expected_stats = [0] * len(STATS)
    for b, r in enumerate(idx):
        if r < 0:
            assert out[b].tolist() == drafts[b][:m] + [drafts[b][m - 1]] * (k - m)
            continue
        if sampled[b] + rejected[b] == k + 1:
            expected_stats[0] += 1
            expected_stats[1] += sampled[b] - 1
            if flags[r]:
                expected_stats[2] += 1
                expected_stats[3] += max(sampled[b] - 1 - m, 0)
        ref, was_extended, found = _reference(
            rows[r], lengths[r], drafts[b], m, k, min_n, max_n
        )
        assert out[b].tolist() == ref, (b, r, lengths[r])
        assert int(extended[r]) == int(was_extended)
        expected_stats[4] += 1
        expected_stats[5] += int(found)
    assert stats.tolist() == expected_stats
