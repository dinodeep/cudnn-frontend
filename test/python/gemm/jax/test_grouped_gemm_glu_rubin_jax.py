# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rubin dense MXFP8 grouped GLU through the JAX custom-call bridge."""

from functools import partial

import numpy as np
import pytest

jax = pytest.importorskip("jax")
ml_dtypes = pytest.importorskip("ml_dtypes")
import jax.numpy as jnp

pytestmark = pytest.mark.L0


def _inputs():
    dtype = ml_dtypes.float8_e4m3fn
    rows = cols = hidden = 256
    scale = jnp.full((1, 2, 2, 32, 4, 4), 127, jnp.uint8)
    return dict(
        a_tensor=jnp.full((rows, hidden), 0.125, dtype),
        b_tensor=jnp.full((1, cols, hidden), 0.125, dtype),
        sfa_tensor=scale,
        sfb_tensor=scale,
        padded_offsets=jnp.array([rows], jnp.int32),
        alpha_tensor=jnp.ones((1,), jnp.float32),
        prob_tensor=jnp.ones((rows,), jnp.float32),
        norm_const_tensor=jnp.ones((1,), jnp.float32),
    )


def _require_rubin():
    from cudnn.api_base import is_sm107_device

    if not is_sm107_device():
        pytest.skip("requires Rubin SM107")


@pytest.mark.parametrize("shared_wrapper", [False, True])
def test_dense_mxfp8_glu_jax_jit_and_replay(shared_wrapper):
    _require_rubin()
    import cudnn
    from cudnn.jax import grouped_gemm_glu

    fn = partial(cudnn.grouped_gemm_glu_wrapper_sm100, sf_vec_size=32, generate_c=True) if shared_wrapper else grouped_gemm_glu
    inputs = _inputs()
    compiled = jax.jit(fn)
    output = compiled(**inputs)
    assert output["c_tensor"].shape == (256, 256, 1)
    assert output["d_tensor"].shape == output["d_col_tensor"].shape == (256, 128, 1)
    assert output["sfd_row_tensor"].shape == (1, 2, 1, 32, 4, 4)
    np.testing.assert_array_equal(np.asarray(output["c_tensor"]), 4)
    assert np.isfinite(np.asarray(output["d_tensor"]).astype(np.float32)).all()
    dequantized = np.asarray(output["d_tensor"]).astype(np.float32) * np.asarray(output["sfd_row_tensor"])[0, 0, 0, 0, 0, 0].astype(np.float32)
    np.testing.assert_allclose(dequantized, float(jax.nn.silu(4.0) * 4.0), rtol=0.03)

    # A changed runtime alpha must reach the same compiled kernel.
    inputs["alpha_tensor"] = jnp.array([0.5], jnp.float32)
    replay = compiled(**inputs)
    np.testing.assert_array_equal(np.asarray(replay["c_tensor"]), 2)


def test_dense_mxfp8_glu_jax_rejects_short_scale():
    _require_rubin()
    from cudnn.jax import grouped_gemm_glu

    inputs = _inputs()
    inputs["sfa_tensor"] = inputs["sfa_tensor"].reshape(-1)[:-1]
    with pytest.raises(ValueError, match="SFA"):
        grouped_gemm_glu(**inputs)


def test_dense_mxfp8_glu_jax_uses_each_experts_weights():
    _require_rubin()
    from cudnn.jax import grouped_gemm_glu

    inputs = _inputs()
    inputs["a_tensor"] = jnp.tile(inputs["a_tensor"], (2, 1))
    inputs["b_tensor"] = jnp.concatenate((inputs["b_tensor"], jnp.full_like(inputs["b_tensor"], 0.25)), axis=0)
    inputs["sfa_tensor"] = jnp.tile(inputs["sfa_tensor"], (1, 2, 1, 1, 1, 1))
    inputs["sfb_tensor"] = jnp.tile(inputs["sfb_tensor"], (2, 1, 1, 1, 1, 1))
    inputs["padded_offsets"] = jnp.array([256, 512], jnp.int32)
    inputs["alpha_tensor"] = jnp.ones((2,), jnp.float32)
    inputs["prob_tensor"] = jnp.ones((512,), jnp.float32)
    output = jax.jit(grouped_gemm_glu)(**inputs)
    c = np.asarray(output["c_tensor"])[:, 0, 0]
    np.testing.assert_array_equal(c[:256], 4)
    np.testing.assert_array_equal(c[256:], 8)


def test_dense_mxfp8_glu_jax_initializes_unused_scale_rows():
    _require_rubin()
    from cudnn.jax import grouped_gemm_glu

    inputs = _inputs()
    inputs["a_tensor"] = jnp.tile(inputs["a_tensor"], (2, 1))
    inputs["sfa_tensor"] = jnp.tile(inputs["sfa_tensor"], (1, 2, 1, 1, 1, 1))
    inputs["prob_tensor"] = jnp.ones((512,), jnp.float32)
    output = jax.jit(grouped_gemm_glu)(**inputs)

    np.testing.assert_array_equal(np.asarray(output["c_tensor"])[:256], 4)
    np.testing.assert_array_equal(np.asarray(output["sfd_row_tensor"]).view(np.uint8)[:, 2:], 0)
    np.testing.assert_array_equal(np.asarray(output["sfd_col_tensor"]).view(np.uint8)[:, :, 2:], 0)
