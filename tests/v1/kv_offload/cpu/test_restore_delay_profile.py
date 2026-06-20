# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import random

import pytest

from vllm.v1.kv_offload.cpu.gpu_worker import RestoreDelayProfile


def test_restore_delay_profile_rejects_invalid_quantiles():
    with pytest.raises(ValueError):
        RestoreDelayProfile(
            min_ms=5.0,
            median_ms=4.0,
            p99_ms=10.0,
            max_ms=12.0,
        )


def test_restore_delay_profile_sampling_is_deterministic():
    profile = RestoreDelayProfile(
        min_ms=1.0,
        median_ms=2.0,
        p99_ms=5.0,
        max_ms=8.0,
        seed=17,
    )
    left_rng = random.Random(profile.seed)
    right_rng = random.Random(profile.seed)

    left = [profile.sample_ms(left_rng) for _ in range(16)]
    right = [profile.sample_ms(right_rng) for _ in range(16)]
    assert left == right


def test_restore_delay_profile_respects_bounds():
    profile = RestoreDelayProfile(
        min_ms=0.8,
        median_ms=1.8,
        p99_ms=5.0,
        max_ms=8.0,
        seed=11,
    )
    rng = random.Random(profile.seed)
    samples = [profile.sample_ms(rng) for _ in range(1024)]
    assert min(samples) >= 0.8 - 1e-9
    assert max(samples) <= 8.0 + 1e-9
