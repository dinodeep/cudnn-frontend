# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rubin dense MXFP8 grouped GEMM + SwiGLU through the XLA-owned JAX bridge."""

import os
from math import prod

import cutlass
import cutlass.cute as cute
import cutlass.utils
import jax
import jax.numpy as jnp
from cutlass.cute.nvgpu import OperandMajorMode

from cudnn.api_base import TupleDict
from cudnn.datatypes import _convert_to_cutlass_data_type
from cudnn.jax import TensorSpec, call, gemm_operand_spec, zeros_init
from cudnn.tensor_adapter import detect_framework
from ..canonical_jax import output_type, sf_array, sf_shape, sf_zeros
from ..moe_utils import MoEWeightMode
from .moe_blockscaled_grouped_gemm_glu_rubin import BlockScaledMoEGroupedGemmGluKernel

_kernel_cache = {}
_fp8_dtypes = (cutlass.Float8E4M3FN, cutlass.Float8E5M2)


@cute.jit
def _rubin_glu_adapter(stream, a, b, sfa, sfb, offsets, alpha, prob, norm_const, c, d, d_col, sfd_row, sfd_col, workspace, *, kernel, mac):
    kernel(
        a=a,
        b=b,
        sfb=sfb,
        n=cutlass.Int32(0),
        k=cutlass.Int32(0),
        b_stride_size=cutlass.Int64(0),
        b_major_mode=OperandMajorMode.K,
        workspace_ptr=workspace.iterator,
        c=c,
        d=d,
        d_col=d_col,
        sfa=sfa,
        sfd_row_tensor=sfd_row,
        sfd_col_tensor=sfd_col,
        amax_tensor=None,
        norm_const_tensor=norm_const,
        padded_offsets=offsets,
        alpha=alpha,
        prob=prob,
        bias=None,
        max_active_clusters=mac,
        stream=stream,
    )


def grouped_gemm_glu(
    a_tensor,
    b_tensor,
    sfa_tensor,
    sfb_tensor,
    padded_offsets,
    alpha_tensor,
    prob_tensor,
    norm_const_tensor,
    c_dtype=cutlass.BFloat16,
    d_dtype=cutlass.Float8E4M3FN,
    mma_tiler_mn=(256, 256),
    cluster_shape_mn=None,
):
    """Run dense Rubin MXFP8 SwiGLU, eagerly or under ``jax.jit``.

    A is ``(m,k)`` and B is physical row-major ``(experts,n,k)``. Scale
    buffers contain E8M0 MMA atom bytes in physical row-major form. Offsets
    are cumulative 256-aligned expert ends. The returned keys match the
    PyTorch GLU wrapper; C is retained for the separate backward operation.
    """
    from cudnn.api_base import is_sm107_device

    if not is_sm107_device():
        raise RuntimeError("cudnn.jax.grouped_gemm_glu requires Rubin (SM107)")
    inputs = (a_tensor, b_tensor, sfa_tensor, sfb_tensor, padded_offsets, alpha_tensor, prob_tensor, norm_const_tensor)
    if any(detect_framework(x) != "jax" for x in inputs):
        raise ValueError("grouped_gemm_glu requires JAX arrays or tracers")
    if a_tensor.ndim != 2 or b_tensor.ndim != 3:
        raise ValueError("A must have shape (m,k) and B (experts,n,k)")
    m, k = a_tensor.shape
    experts, n, bk = b_tensor.shape
    if m <= 0 or m % 256 or experts <= 0 or experts > 1024 or k != bk or k % 128 or n % 256:
        raise ValueError("A/B require positive M padded to 256, matching 128-aligned K, 256-aligned N, and 1..1024 experts")
    if n % 64:
        raise ValueError("B N must contain paired 32-column SwiGLU blocks")
    a_dtype = _convert_to_cutlass_data_type(a_tensor.dtype)
    if a_dtype not in _fp8_dtypes or _convert_to_cutlass_data_type(b_tensor.dtype) is not a_dtype:
        raise ValueError("A and B must have the same MXFP8 dtype")
    for name, scale in (("SFA", sfa_tensor), ("SFB", sfb_tensor)):
        if _convert_to_cutlass_data_type(scale.dtype) not in (cutlass.Uint8, cutlass.Float8E8M0FNU):
            raise ValueError(f"{name} must contain E8M0 scale bytes")
    sfa_tensor = sf_array(sfa_tensor)
    sfb_tensor = sf_array(sfb_tensor)
    for name, scale, shape in (("SFA", sfa_tensor, sf_shape(m, k)), ("SFB", sfb_tensor, (experts, *sf_shape(n, k)[1:]))):
        if scale.size != prod(shape):
            raise ValueError(f"{name} must contain {shape} physical atom elements")
        scale = scale.reshape(shape)
        if name == "SFA":
            sfa_tensor = scale
        else:
            sfb_tensor = scale
    if padded_offsets.shape != (experts,) or _convert_to_cutlass_data_type(padded_offsets.dtype) is not cutlass.Int32:
        raise ValueError("padded_offsets must be (experts,) int32")
    if alpha_tensor.shape != (experts,) or _convert_to_cutlass_data_type(alpha_tensor.dtype) is not cutlass.Float32:
        raise ValueError("alpha_tensor must be (experts,) float32")
    if prob_tensor.shape != (m,) or _convert_to_cutlass_data_type(prob_tensor.dtype) is not cutlass.Float32:
        raise ValueError("prob_tensor must be (m,) float32")
    if norm_const_tensor.shape != (1,) or _convert_to_cutlass_data_type(norm_const_tensor.dtype) is not cutlass.Float32:
        raise ValueError("norm_const_tensor must be (1,) float32")
    c_dtype = _convert_to_cutlass_data_type(c_dtype)
    d_dtype = _convert_to_cutlass_data_type(d_dtype)
    if c_dtype not in (cutlass.BFloat16, cutlass.Float16) or d_dtype not in _fp8_dtypes:
        raise ValueError("C must be BF16/FP16 and D must be MXFP8")
    mma_tiler_mn = tuple(mma_tiler_mn)
    cluster_shape_mn = tuple(cluster_shape_mn or ((2, 1) if mma_tiler_mn[0] == 256 else (1, 1)))
    if mma_tiler_mn not in ((128, 256), (256, 256)) or cluster_shape_mn != ((2, 1) if mma_tiler_mn[0] == 256 else (1, 1)):
        raise ValueError("Unsupported Rubin GLU tile or cluster configuration")

    margin = int(os.getenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", "0"))
    key = (experts, mma_tiler_mn, cluster_shape_mn, margin)
    entry = _kernel_cache.get(key)
    if entry is None:
        kernel = BlockScaledMoEGroupedGemmGluKernel(
            sf_vec_size=32,
            acc_dtype=cutlass.Float32,
            use_2cta_instrs=mma_tiler_mn[0] == 256,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            vectorized_f32=False,
            generate_sfd=True,
            discrete_col_sfd=False,
            expert_cnt=experts,
            weight_mode=MoEWeightMode.DENSE,
            act_func="swiglu",
            enable_bias=False,
            generate_c=True,
        )
        mac = cutlass.utils.HardwareInfo().get_max_active_clusters(cluster_shape_mn[0] * cluster_shape_mn[1]) - margin
        if mac <= 0:
            raise ValueError("CUDNNFE_CLUSTER_OVERLAP_MARGIN leaves no active clusters")
        entry = (kernel, mac, max(kernel.get_workspace_bytes(), 1))
        _kernel_cache[key] = entry
    kernel, mac, workspace_bytes = entry
    outputs = (
        output_type((m, n, 1), c_dtype),
        output_type((m, n // 2, 1), d_dtype),
        output_type((m, n // 2, 1), d_dtype),
        output_type(sf_shape(m, n // 2), cutlass.Float8E8M0FNU),
        output_type(sf_shape(n // 2, m), cutlass.Float8E8M0FNU),
        jax.ShapeDtypeStruct((workspace_bytes,), jnp.uint8),
    )
    operand = gemm_operand_spec()
    result = call(
        _rubin_glu_adapter,
        output_shape_dtype=outputs,
        input_spec=(operand, TensorSpec(mode=(1, 2, 0)), None, None, None, None, None, None),
        output_spec=(operand, operand, operand, None, None, None),
        initialized_outputs={0: zeros_init, 1: zeros_init, 2: zeros_init, 3: sf_zeros, 4: sf_zeros},
        kernel=kernel,
        mac=mac,
    )(
        a_tensor.reshape(m, k, 1),
        b_tensor,
        sfa_tensor,
        sfb_tensor,
        padded_offsets,
        alpha_tensor,
        prob_tensor.reshape(m, 1, 1),
        norm_const_tensor,
    )
    return TupleDict(
        c_tensor=result[0],
        d_tensor=result[1],
        d_col_tensor=result[2],
        amax_tensor=None,
        sfd_row_tensor=result[3],
        sfd_col_tensor=result[4],
    )
