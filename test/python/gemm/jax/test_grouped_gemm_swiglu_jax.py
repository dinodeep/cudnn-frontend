# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Legacy JAX layouts and direct API-class samples remain unsupported; canonical MXFP8 wrapper coverage lives in test_grouped_gemm_canonical_jax.py."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
ml_dtypes = pytest.importorskip("ml_dtypes")
import jax.numpy as jnp


def make_jax_inputs(m=256, n=256, k=128, experts=2):
    """Plausible (physical-layout) inputs; rejection fires before shape validation."""
    rng = np.random.default_rng(20260809)
    a_j = jnp.asarray(rng.integers(0, 255, (m, k // 2, 1), dtype=np.uint8))  # fp4x2 container
    b_j = jnp.asarray(rng.integers(0, 255, (n, k // 2, experts), dtype=np.uint8))
    rest_k = -(-(-(-k // 16) // 4))  # ceil_div(ceil_div(k, 16), 4)
    sfa_j = jnp.asarray(rng.integers(0, 127, (32, 4, -(-m // 128), 4, rest_k, 1), dtype=np.uint8).view(ml_dtypes.float8_e8m0fnu))
    sfb_j = jnp.asarray(rng.integers(0, 127, (32, 4, -(-n // 128), 4, rest_k, experts), dtype=np.uint8).view(ml_dtypes.float8_e8m0fnu))
    group_m = m // experts
    offsets_j = jnp.asarray(np.arange(group_m, m + 1, group_m, dtype=np.int32))
    alpha_j = jnp.asarray(np.ones(experts, dtype=np.float32))
    return a_j, b_j, sfa_j, sfb_j, offsets_j, alpha_j


@pytest.mark.L0
def test_grouped_gemm_swiglu_jax_writes_active_scales_without_initialization():
    from cudnn.jax import grouped_gemm_swiglu

    rows, hidden, combined = 512, 256, 256
    inputs = dict(
        a_tensor=jnp.full((rows, hidden), 0.125, ml_dtypes.float8_e4m3fn),
        b_tensor=jnp.concatenate(
            [
                jnp.full((1, combined, hidden), 0.125, ml_dtypes.float8_e4m3fn),
                jnp.full((1, combined, hidden), 0.25, ml_dtypes.float8_e4m3fn),
            ]
        ),
        sfa_tensor=jnp.full((1, 4, 2, 32, 4, 4), 127, jnp.uint8),
        sfb_tensor=jnp.full((2, 2, 2, 32, 4, 4), 127, jnp.uint8),
        padded_offsets=jnp.array([256, 512], jnp.int32),
        alpha_tensor=jnp.ones((2,), jnp.float32),
        prob_tensor=jnp.ones((rows,), jnp.float32),
        norm_const_tensor=jnp.ones((1,), jnp.float32),
    )
    result = jax.jit(lambda **kwargs: grouped_gemm_swiglu(**kwargs, discrete_col_sfd=True))(**inputs)
    np.testing.assert_array_equal(np.asarray(result["c_tensor"])[:256], 4)
    np.testing.assert_array_equal(np.asarray(result["c_tensor"])[256:], 8)
    col_scales = np.asarray(result["sfd_col_tensor"]).view(np.uint8).reshape(2, -1)
    assert np.all((col_scales > 0) & (col_scales < 255))
    assert np.unique(col_scales[0]).size == np.unique(col_scales[1]).size == 1
    assert col_scales[0, 0] != col_scales[1, 0]


@pytest.mark.L0
def test_grouped_gemm_swiglu_jax_rejected_with_clear_error():
    from cudnn.frost.buffers import cutedsl_requirement_error

    requirement = cutedsl_requirement_error("JAX grouped GEMM tests")
    if requirement:
        pytest.skip(requirement)
    from cudnn import grouped_gemm_swiglu_wrapper_sm100

    a_j, b_j, sfa_j, sfb_j, offsets_j, alpha_j = make_jax_inputs()
    with pytest.raises(ValueError, match="JAX requires canonical A"):
        grouped_gemm_swiglu_wrapper_sm100(
            a_tensor=a_j,
            b_tensor=b_j,
            sfa_tensor=sfa_j,
            sfb_tensor=sfb_j,
            padded_offsets=offsets_j,
            alpha_tensor=alpha_j,
            sf_vec_size=32,
            prob_tensor=jnp.ones((a_j.shape[0],), jnp.float32),
            norm_const_tensor=jnp.ones((1,), jnp.float32),
            d_dtype=ml_dtypes.float8_e4m3fn,
        )


@pytest.mark.L0
def test_grouped_gemm_swiglu_jax_api_class_rejected():
    from cudnn.gemm.cutedsl.grouped.swiglu.api import GroupedGemmSwigluSm100

    a_j, b_j, sfa_j, sfb_j, offsets_j, alpha_j = make_jax_inputs()
    with pytest.raises(ValueError, match="not expressible as JAX arrays"):
        GroupedGemmSwigluSm100(
            sample_a=a_j,
            sample_b=b_j,
            sample_c=None,
            sample_d=None,
            sample_sfa=sfa_j,
            sample_sfb=sfb_j,
            sample_padded_offsets=offsets_j,
            sample_alpha=alpha_j,
            sample_d_col=None,
        )


@pytest.mark.L0
def test_grouped_gemm_swiglu_unknown_framework_rejected():
    from cudnn import grouped_gemm_swiglu_wrapper_sm100

    a_np = np.zeros((256, 64, 1), dtype=np.uint8)
    with pytest.raises(ValueError, match="Unsupported tensor framework 'numpy'"):
        grouped_gemm_swiglu_wrapper_sm100(
            a_tensor=a_np,
            b_tensor=None,
            sfa_tensor=None,
            sfb_tensor=None,
            padded_offsets=None,
            alpha_tensor=None,
        )
