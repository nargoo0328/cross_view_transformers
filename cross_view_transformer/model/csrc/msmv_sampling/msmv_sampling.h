#pragma once

#include <torch/extension.h>

// ADD 1 & 2 & 3 layer number
at::Tensor ms_deform_attn_cuda_c2_forward(
    const at::Tensor& feat_c2,  // [B, N, H, W, C]
    const at::Tensor& sampling_loc,  // [B, Q, P, 3]
    const at::Tensor& attn_weight  // [B, Q, P, 4]
);

std::vector<at::Tensor> ms_deform_attn_cuda_c2_backward(
    const at::Tensor& feat_c2,  // [B, N, H, W, C]
    const at::Tensor& sampling_loc,  // [B, Q, P, 3]
    const at::Tensor& attn_weight,  // [B, Q, P, 4]
    const at::Tensor& grad_output
);

at::Tensor ms_deform_attn_cuda_c23_forward(
    const at::Tensor& feat_c2,  // [B, N, H, W, C]
    const at::Tensor& feat_c3,  // [B, N, H, W, C]
    const at::Tensor& sampling_loc,  // [B, Q, P, 3]
    const at::Tensor& attn_weight  // [B, Q, P, 4]
);

std::vector<at::Tensor> ms_deform_attn_cuda_c23_backward(
    const at::Tensor& feat_c2,  // [B, N, H, W, C]
    const at::Tensor& feat_c3,  // [B, N, H, W, C]
    const at::Tensor& sampling_loc,  // [B, Q, P, 3]
    const at::Tensor& attn_weight,  // [B, Q, P, 4]
    const at::Tensor& grad_output
);

at::Tensor ms_deform_attn_cuda_c234_forward(
    const at::Tensor& feat_c2,  // [B, N, H, W, C]
    const at::Tensor& feat_c3,  // [B, N, H, W, C]
    const at::Tensor& feat_c4,  // [B, N, H, W, C]
    const at::Tensor& sampling_loc,  // [B, Q, P, 3]
    const at::Tensor& attn_weight  // [B, Q, P, 4]
);

std::vector<at::Tensor> ms_deform_attn_cuda_c234_backward(
    const at::Tensor& feat_c2,  // [B, N, H, W, C]
    const at::Tensor& feat_c3,  // [B, N, H, W, C]
    const at::Tensor& feat_c4,  // [B, N, H, W, C]
    const at::Tensor& sampling_loc,  // [B, Q, P, 3]
    const at::Tensor& attn_weight,  // [B, Q, P, 4]
    const at::Tensor& grad_output
);

///////////

at::Tensor ms_deform_attn_cuda_c2345_forward(
    const at::Tensor& feat_c2,  // [B, N, H, W, C]
    const at::Tensor& feat_c3,  // [B, N, H, W, C]
    const at::Tensor& feat_c4,  // [B, N, H, W, C]
    const at::Tensor& feat_c5,  // [B, N, H, W, C]
    const at::Tensor& sampling_loc,  // [B, Q, P, 3]
    const at::Tensor& attn_weight  // [B, Q, P, 4]
);

std::vector<at::Tensor> ms_deform_attn_cuda_c2345_backward(
    const at::Tensor& feat_c2,  // [B, N, H, W, C]
    const at::Tensor& feat_c3,  // [B, N, H, W, C]
    const at::Tensor& feat_c4,  // [B, N, H, W, C]
    const at::Tensor& feat_c5,  // [B, N, H, W, C]
    const at::Tensor& sampling_loc,  // [B, Q, P, 3]
    const at::Tensor& attn_weight,  // [B, Q, P, 4]
    const at::Tensor& grad_output
);

at::Tensor ms_deform_attn_cuda_c23456_forward(
    const at::Tensor& feat_c2,  // [B, N, H, W, C]
    const at::Tensor& feat_c3,  // [B, N, H, W, C]
    const at::Tensor& feat_c4,  // [B, N, H, W, C]
    const at::Tensor& feat_c5,  // [B, N, H, W, C]
    const at::Tensor& feat_c6,  // [B, N, H, W, C]
    const at::Tensor& sampling_loc,  // [B, Q, P, 3]
    const at::Tensor& attn_weight  // [B, Q, P, 4]
);

std::vector<at::Tensor> ms_deform_attn_cuda_c23456_backward(
    const at::Tensor& grad_output,
    const at::Tensor& feat_c2,  // [B, N, H, W, C]
    const at::Tensor& feat_c3,  // [B, N, H, W, C]
    const at::Tensor& feat_c4,  // [B, N, H, W, C]
    const at::Tensor& feat_c5,  // [B, N, H, W, C]
    const at::Tensor& feat_c6,  // [B, N, H, W, C]
    const at::Tensor& sampling_loc,  // [B, Q, P, 3]
    const at::Tensor& attn_weight  // [B, Q, P, 4]
);