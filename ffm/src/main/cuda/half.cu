// Float16 activation kernels for the CUDA diffusion runtime. Storage is __half; arithmetic is Float32.
// Compiled to ffm/src/main/resources/scalatorch/half.ptx.
#include <cuda_fp16.h>
#include <mma.h>

#define GRID_INDEX(i, n) unsigned i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= (n)) return;

__device__ __forceinline__ float ld(const __half* p, unsigned i) { return __half2float(p[i]); }
__device__ __forceinline__ void st(__half* p, unsigned i, float v) { p[i] = __float2half_rn(v); }
__device__ __forceinline__ bool aligned4(const void* p) { return (((unsigned long long)p) & 3ULL) == 0ULL; }
__device__ __forceinline__ bool aligned8(const void* p) { return (((unsigned long long)p) & 7ULL) == 0ULL; }
__device__ __forceinline__ float fast_exp2f(float x) {
  float res;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(res) : "f"(x));
  return res;
}

__device__ float block_sum(float v, float* shared) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  unsigned lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  __syncthreads();
  if (lane == 0) shared[warp] = v;
  __syncthreads();
  float total = 0.f;
  for (unsigned w = 0; w < (blockDim.x >> 5); ++w) total += shared[w];
  return total;
}

__device__ float block_max(float v, float* shared) {
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  unsigned lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  __syncthreads();
  if (lane == 0) shared[warp] = v;
  __syncthreads();
  float best = -3.402823466e+38f;
  for (unsigned w = 0; w < (blockDim.x >> 5); ++w) best = fmaxf(best, shared[w]);
  return best;
}

extern "C" {

// Vectorized 2-elements-per-thread conversion/elementwise kernels (caller launches (n + 1) / 2 threads).
__global__ void h_f32_to_f16(const float* in, __half* out, unsigned n) {
  unsigned j = (blockIdx.x * blockDim.x + threadIdx.x) << 1;
  if (j >= n) return;
  if (j + 1u < n && aligned8(in) && aligned4(out)) {
    float2 f = ((const float2*)in)[j >> 1];
    ((__half2*)out)[j >> 1] = __floats2half2_rn(f.x, f.y);
  } else {
    out[j] = __float2half_rn(in[j]);
    if (j + 1u < n) out[j + 1u] = __float2half_rn(in[j + 1u]);
  }
}

__global__ void h_f16_to_f32(const __half* in, float* out, unsigned n) {
  unsigned j = (blockIdx.x * blockDim.x + threadIdx.x) << 1;
  if (j >= n) return;
  if (j + 1u < n && aligned4(in) && aligned8(out)) {
    float2 f = __half22float2(((const __half2*)in)[j >> 1]);
    ((float2*)out)[j >> 1] = f;
  } else {
    out[j] = __half2float(in[j]);
    if (j + 1u < n) out[j + 1u] = __half2float(in[j + 1u]);
  }
}

__global__ void h_fill(__half* out, unsigned n, float value) {
  GRID_INDEX(i, n) out[i] = __float2half_rn(value);
}

// out [rows, width] = bias [width] in every row; the matrix product is then accumulated on top (see linearWithBias).
__global__ void h_fill_rows(const __half* bias, __half* out, unsigned n, unsigned width) {
  unsigned j = (blockIdx.x * blockDim.x + threadIdx.x) << 1;
  if (j >= n) return;
  if (j + 1u < n && ((width & 1u) == 0u) && aligned4(bias) && aligned4(out)) {
    unsigned c = j % width;
    ((__half2*)out)[j >> 1] = ((const __half2*)bias)[c >> 1];
  } else {
    out[j] = bias[j % width];
    if (j + 1u < n) out[j + 1u] = bias[(j + 1u) % width];
  }
}

// out = a + bias along the last axis: the start of `residual + x W^T + bias`, on which cuBLAS accumulates the product.
__global__ void h_add_rows(const __half* a, const __half* bias, __half* out, unsigned n, unsigned width) {
  unsigned j = (blockIdx.x * blockDim.x + threadIdx.x) << 1;
  if (j >= n) return;
  if (j + 1u < n && ((width & 1u) == 0u) && aligned4(a) && aligned4(bias) && aligned4(out)) {
    unsigned c = j % width;
    float2 fa = __half22float2(((const __half2*)a)[j >> 1]);
    float2 fb = __half22float2(((const __half2*)bias)[c >> 1]);
    ((__half2*)out)[j >> 1] = __floats2half2_rn(fa.x + fb.x, fa.y + fb.y);
  } else {
    st(out, j, ld(a, j) + ld(bias, j % width));
    if (j + 1u < n) st(out, j + 1u, ld(a, j + 1u) + ld(bias, (j + 1u) % width));
  }
}

__global__ void h_add(const __half* a, const __half* b, __half* out, unsigned n) {
  unsigned j = (blockIdx.x * blockDim.x + threadIdx.x) << 1;
  if (j >= n) return;
  if (j + 1u < n && aligned4(a) && aligned4(b) && aligned4(out)) {
    float2 fa = __half22float2(((const __half2*)a)[j >> 1]);
    float2 fb = __half22float2(((const __half2*)b)[j >> 1]);
    ((__half2*)out)[j >> 1] = __floats2half2_rn(fa.x + fb.x, fa.y + fb.y);
  } else {
    st(out, j, ld(a, j) + ld(b, j));
    if (j + 1u < n) st(out, j + 1u, ld(a, j + 1u) + ld(b, j + 1u));
  }
}

// Broadcast add over up to five axes; d* are the output dims, sa*/sb* the operand strides (0 where broadcast).
__global__ void h_add5(const __half* a, const __half* b, __half* out, unsigned n,
                       unsigned d0, unsigned d1, unsigned d2, unsigned d3, unsigned d4,
                       unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned a4,
                       unsigned b0, unsigned b1, unsigned b2, unsigned b3, unsigned b4) {
  GRID_INDEX(i, n)
  // The usual case is a bias: `a` is the whole tensor and `b` varies along one axis only. One division pair instead of five.
  // (An axis of size 1 has stride 0 by convention and does not matter.)
  if ((d4 == 1 || a4 == 1) && (d3 == 1 || a3 == d4) && (d2 == 1 || a2 == d3 * d4) && (d1 == 1 || a1 == d2 * d3 * d4) &&
      (d0 == 1 || a0 == d1 * d2 * d3 * d4) &&
      (b0 != 0) + (b1 != 0) + (b2 != 0) + (b3 != 0) + (b4 != 0) <= 1) {
    unsigned inner, dim, stride;
    if (b4) { inner = 1; dim = d4; stride = b4; }
    else if (b3) { inner = d4; dim = d3; stride = b3; }
    else if (b2) { inner = d3 * d4; dim = d2; stride = b2; }
    else if (b1) { inner = d2 * d3 * d4; dim = d1; stride = b1; }
    else { inner = d1 * d2 * d3 * d4; dim = d0; stride = b0; }
    st(out, i, ld(a, i) + ld(b, ((i / inner) % dim) * stride));
    return;
  }
  unsigned t = i, ia = 0, ib = 0, c;
  c = t % d4; t /= d4; ia += c * a4; ib += c * b4;
  c = t % d3; t /= d3; ia += c * a3; ib += c * b3;
  c = t % d2; t /= d2; ia += c * a2; ib += c * b2;
  c = t % d1; t /= d1; ia += c * a1; ib += c * b1;
  c = t % d0; ia += c * a0; ib += c * b0;
  st(out, i, ld(a, ia) + ld(b, ib));
}

__global__ void h_scale(const __half* a, __half* out, unsigned n, float s) {
  unsigned j = (blockIdx.x * blockDim.x + threadIdx.x) << 1;
  if (j >= n) return;
  if (j + 1u < n && aligned4(a) && aligned4(out)) {
    float2 fa = __half22float2(((const __half2*)a)[j >> 1]);
    ((__half2*)out)[j >> 1] = __floats2half2_rn(fa.x * s, fa.y * s);
  } else {
    st(out, j, ld(a, j) * s);
    if (j + 1u < n) st(out, j + 1u, ld(a, j + 1u) * s);
  }
}

__global__ void h_silu(const __half* a, __half* out, unsigned n) {
  unsigned j = (blockIdx.x * blockDim.x + threadIdx.x) << 1;
  if (j >= n) return;
  if (j + 1u < n && aligned4(a) && aligned4(out)) {
    float2 fa = __half22float2(((const __half2*)a)[j >> 1]);
    ((__half2*)out)[j >> 1] = __floats2half2_rn(fa.x / (1.f + __expf(-fa.x)), fa.y / (1.f + __expf(-fa.y)));
  } else {
    float x0 = ld(a, j); st(out, j, x0 / (1.f + __expf(-x0)));
    if (j + 1u < n) { float x1 = ld(a, j + 1u); st(out, j + 1u, x1 / (1.f + __expf(-x1))); }
  }
}

// Fast NCHW ResNet fused epilogues: replace separate h_add5 + h_add + h_scale passes.
__global__ void h_add_nchw_bias(const __half* a, const __half* bias, __half* out,
                                unsigned n, unsigned channels, unsigned spatial) {
  unsigned j = (blockIdx.x * blockDim.x + threadIdx.x) << 1;
  if (j >= n) return;
  if (j + 1u < n && ((spatial & 1u) == 0u) && aligned4(a) && aligned4(out)) {
    float bval = ld(bias, (j / spatial) % channels);
    float2 fa = __half22float2(((const __half2*)a)[j >> 1]);
    ((__half2*)out)[j >> 1] = __floats2half2_rn(fa.x + bval, fa.y + bval);
  } else {
    st(out, j, ld(a, j) + ld(bias, (j / spatial) % channels));
    if (j + 1u < n) st(out, j + 1u, ld(a, j + 1u) + ld(bias, ((j + 1u) / spatial) % channels));
  }
}

__global__ void h_add_nchw_bias_time(const __half* a, const __half* bias, const __half* time, __half* out,
                                     unsigned n, unsigned channels, unsigned spatial) {
  unsigned j = (blockIdx.x * blockDim.x + threadIdx.x) << 1;
  if (j >= n) return;
  if (j + 1u < n && ((spatial & 1u) == 0u) && aligned4(a) && aligned4(out)) {
    unsigned sc = j / spatial;
    float add = (bias ? ld(bias, sc % channels) : 0.f) + ld(time, sc);
    float2 fa = __half22float2(((const __half2*)a)[j >> 1]);
    ((__half2*)out)[j >> 1] = __floats2half2_rn(fa.x + add, fa.y + add);
  } else {
    unsigned sc0 = j / spatial;
    st(out, j, ld(a, j) + (bias ? ld(bias, sc0 % channels) : 0.f) + ld(time, sc0));
    if (j + 1u < n) {
      unsigned sc1 = (j + 1u) / spatial;
      st(out, j + 1u, ld(a, j + 1u) + (bias ? ld(bias, sc1 % channels) : 0.f) + ld(time, sc1));
    }
  }
}

__global__ void h_add_nchw_bias_res(const __half* a, const __half* bias, const __half* shortcut, __half* out,
                                    unsigned n, unsigned channels, unsigned spatial, float scale) {
  unsigned j = (blockIdx.x * blockDim.x + threadIdx.x) << 1;
  if (j >= n) return;
  if (j + 1u < n && ((spatial & 1u) == 0u) && aligned4(a) && aligned4(shortcut) && aligned4(out)) {
    float bval = bias ? ld(bias, (j / spatial) % channels) : 0.f;
    float2 fa = __half22float2(((const __half2*)a)[j >> 1]);
    float2 fs = __half22float2(((const __half2*)shortcut)[j >> 1]);
    float r0 = (fa.x + bval) + fs.x;
    float r1 = (fa.y + bval) + fs.y;
    if (scale != 1.f) { r0 *= scale; r1 *= scale; }
    ((__half2*)out)[j >> 1] = __floats2half2_rn(r0, r1);
  } else {
    float b0 = bias ? ld(bias, (j / spatial) % channels) : 0.f;
    float r0 = (ld(a, j) + b0) + ld(shortcut, j);
    if (scale != 1.f) r0 *= scale;
    st(out, j, r0);
    if (j + 1u < n) {
      float b1 = bias ? ld(bias, ((j + 1u) / spatial) % channels) : 0.f;
      float r1 = (ld(a, j + 1u) + b1) + ld(shortcut, j + 1u);
      if (scale != 1.f) r1 *= scale;
      st(out, j + 1u, r1);
    }
  }
}

// input [..., 2*width]: out = input[..., :width] * gelu(input[..., width:]) with exact (erf) GELU.
__global__ void h_geglu(const __half* in, __half* out, unsigned n, unsigned width) {
  GRID_INDEX(i, n)
  unsigned row = i / width, col = i % width;
  float v = ld(in, row * 2 * width + col);
  float g = ld(in, row * 2 * width + width + col);
  float gelu = 0.5f * g * (1.f + erff(g * 0.70710678118654752f));
  st(out, i, v * gelu);
}

// GEGLU of `in + bias` (bias [2 * width]): the bias of the projection before it is added here, not in a pass of its own.
__global__ void h_geglu_bias(const __half* in, const __half* bias, __half* out, unsigned n, unsigned width) {
  GRID_INDEX(i, n)
  unsigned row = i / width, col = i % width;
  float v = ld(in, row * 2 * width + col) + ld(bias, col);
  float g = ld(in, row * 2 * width + width + col) + ld(bias, width + col);
  float gelu = 0.5f * g * (1.f + erff(g * 0.70710678118654752f));
  st(out, i, v * gelu);
}

// Swap axes lo<hi of [outer, X, mid, Y, inner] -> [outer, Y, mid, X, inner]; sizes only.
__global__ void h_transpose5(const unsigned short* in, unsigned short* out, unsigned n,
                             unsigned x, unsigned mid, unsigned y, unsigned inner) {
  GRID_INDEX(i, n)
  unsigned t = i, e = t % inner; t /= inner;
  unsigned xi = t % x; t /= x;
  unsigned m = t % mid; t /= mid;
  unsigned yi = t % y; unsigned o = t / y;
  unsigned idx = (((o * x + xi) * mid + m) * y + yi) * inner + e;
  out[i] = in[idx];
}

// Copy one input of a concatenation: input viewed [outer, rowIn], output [outer, rowOut], placed at offset.
__global__ void h_concat(const unsigned short* in, unsigned short* out, unsigned n,
                         unsigned rowIn, unsigned rowOut, unsigned offset) {
  GRID_INDEX(i, n)
  unsigned o = i / rowIn, r = i % rowIn;
  out[o * rowOut + offset + r] = in[i];
}

__global__ void h_resize_nearest(const unsigned short* in, unsigned short* out, unsigned n,
                                 unsigned ih, unsigned iw, unsigned oh, unsigned ow) {
  GRID_INDEX(i, n)
  unsigned x = i % ow, t = i / ow, y = t % oh, plane = t / oh;
  unsigned sy = (unsigned)(((unsigned long long)y * ih) / oh);
  unsigned sx = (unsigned)(((unsigned long long)x * iw) / ow);
  out[i] = in[(plane * ih + sy) * iw + sx];
}

// One block per row; two passes so the variance does not suffer cancellation. stats[row] = (mean, inverse std).
__global__ void h_groupnorm_stats(const __half* x, float2* stats, unsigned rows, unsigned width, float eps) {
  __shared__ float shared[32];
  unsigned row = blockIdx.x;
  if (row >= rows) return;
  const __half* base = x + (unsigned long long)row * width;
  float s = 0.f;
  if (aligned4(base) && ((width & 1u) == 0u)) {
    const __half2* b2 = (const __half2*)base;
    unsigned w2 = width >> 1;
    for (unsigned i = threadIdx.x; i < w2; i += blockDim.x) {
      float2 f = __half22float2(b2[i]);
      s += f.x + f.y;
    }
    float mean = block_sum(s, shared) / (float)width;
    float q = 0.f;
    for (unsigned i = threadIdx.x; i < w2; i += blockDim.x) {
      float2 f = __half22float2(b2[i]);
      float d0 = f.x - mean, d1 = f.y - mean;
      q += d0 * d0 + d1 * d1;
    }
    float var = block_sum(q, shared) / (float)width;
    if (threadIdx.x == 0) stats[row] = make_float2(mean, rsqrtf(var + eps));
    return;
  }
  for (unsigned i = threadIdx.x; i < width; i += blockDim.x) s += ld(base, i);
  float mean = block_sum(s, shared) / (float)width;
  float q = 0.f;
  for (unsigned i = threadIdx.x; i < width; i += blockDim.x) { float d = ld(base, i) - mean; q += d * d; }
  float var = block_sum(q, shared) / (float)width;
  if (threadIdx.x == 0) stats[row] = make_float2(mean, rsqrtf(var + eps));
}

// LayerNorm of rows up to 8 * blockDim wide in one kernel: a block reads its row once, keeps it in registers, and writes the
// result (h_groupnorm_stats reads the row twice and h_groupnorm_apply a third time). Same sums in the same order as those two.
__global__ void h_norm_rows(const __half* x, const __half* weight, const __half* bias, __half* out,
                            unsigned rows, unsigned width, float eps) {
  __shared__ float shared[32];
  unsigned row = blockIdx.x;
  if (row >= rows) return;
  const __half* base = x + (unsigned long long)row * width;
  __half* o = out + (unsigned long long)row * width;
  float v[8];
  float s = 0.f;
  #pragma unroll
  for (int j = 0; j < 8; ++j) {
    unsigned i = threadIdx.x + j * blockDim.x;
    v[j] = i < width ? ld(base, i) : 0.f;
    s += v[j];
  }
  float mean = block_sum(s, shared) / (float)width;
  float q = 0.f;
  #pragma unroll
  for (int j = 0; j < 8; ++j) {
    unsigned i = threadIdx.x + j * blockDim.x;
    if (i < width) { float d = v[j] - mean; q += d * d; }
  }
  float inv = rsqrtf(block_sum(q, shared) / (float)width + eps);
  #pragma unroll
  for (int j = 0; j < 8; ++j) {
    unsigned i = threadIdx.x + j * blockDim.x;
    if (i < width) {
      float r = (v[j] - mean) * inv;
      if (weight) r *= ld(weight, i);
      if (bias) r += ld(bias, i);
      st(o, i, r);
    }
  }
}

__global__ void h_groupnorm_apply(const __half* x, const __half* weight, const __half* bias, __half* out,
                                  const float2* stats, unsigned n, unsigned width, unsigned channels, unsigned spatial) {
  unsigned j = (blockIdx.x * blockDim.x + threadIdx.x) << 1;
  if (j >= n) return;
  if (j + 1u < n && ((spatial & 1u) == 0u) && ((width & 1u) == 0u) && aligned4(x) && aligned4(out)) {
    float2 s = stats[j / width];
    unsigned c = (j / spatial) % channels;
    float w = weight ? ld(weight, c) : 1.f;
    float b = bias ? ld(bias, c) : 0.f;
    float2 fx = __half22float2(((const __half2*)x)[j >> 1]);
    float v0 = (fx.x - s.x) * s.y;
    float v1 = (fx.y - s.x) * s.y;
    if (weight) { v0 *= w; v1 *= w; }
    if (bias) { v0 += b; v1 += b; }
    ((__half2*)out)[j >> 1] = __floats2half2_rn(v0, v1);
  } else {
    float2 s0 = stats[j / width];
    float v0 = (ld(x, j) - s0.x) * s0.y;
    unsigned c0 = (j / spatial) % channels;
    if (weight) v0 *= ld(weight, c0);
    if (bias) v0 += ld(bias, c0);
    st(out, j, v0);
    if (j + 1u < n) {
      float2 s1 = stats[(j + 1u) / width];
      float v1 = (ld(x, j + 1u) - s1.x) * s1.y;
      unsigned c1 = ((j + 1u) / spatial) % channels;
      if (weight) v1 *= ld(weight, c1);
      if (bias) v1 += ld(bias, c1);
      st(out, j + 1u, v1);
    }
  }
}

__global__ void h_groupnorm_apply_silu(const __half* x, const __half* weight, const __half* bias, __half* out,
                                       const float2* stats, unsigned n, unsigned width, unsigned channels, unsigned spatial) {
  unsigned j = (blockIdx.x * blockDim.x + threadIdx.x) << 1;
  if (j >= n) return;
  if (j + 1u < n && ((spatial & 1u) == 0u) && ((width & 1u) == 0u) && aligned4(x) && aligned4(out)) {
    float2 s = stats[j / width];
    unsigned c = (j / spatial) % channels;
    float w = weight ? ld(weight, c) : 1.f;
    float b = bias ? ld(bias, c) : 0.f;
    float2 fx = __half22float2(((const __half2*)x)[j >> 1]);
    float v0 = (fx.x - s.x) * s.y;
    float v1 = (fx.y - s.x) * s.y;
    if (weight) { v0 *= w; v1 *= w; }
    if (bias) { v0 += b; v1 += b; }
    v0 = v0 / (1.f + __expf(-v0));
    v1 = v1 / (1.f + __expf(-v1));
    ((__half2*)out)[j >> 1] = __floats2half2_rn(v0, v1);
  } else {
    float2 s0 = stats[j / width];
    float v0 = (ld(x, j) - s0.x) * s0.y;
    unsigned c0 = (j / spatial) % channels;
    if (weight) v0 *= ld(weight, c0);
    if (bias) v0 += ld(bias, c0);
    st(out, j, v0 / (1.f + __expf(-v0)));
    if (j + 1u < n) {
      float2 s1 = stats[(j + 1u) / width];
      float v1 = (ld(x, j + 1u) - s1.x) * s1.y;
      unsigned c1 = ((j + 1u) / spatial) % channels;
      if (weight) v1 *= ld(weight, c1);
      if (bias) v1 += ld(bias, c1);
      st(out, j + 1u, v1 / (1.f + __expf(-v1)));
    }
  }
}

// im2col for one (batch, group) chunk: col[r][p], r in [0, cg*kh*kw), p in [0, positions).
__global__ void h_im2col(const __half* x, __half* col, unsigned count,
                         unsigned ci, unsigned h, unsigned w, unsigned cg, unsigned kh, unsigned kw,
                         unsigned oh, unsigned ow, int sh, int sw, int ph, int pw, int dh, int dw,
                         unsigned batch, unsigned group, unsigned positionStart, unsigned positions) {
  GRID_INDEX(i, count)
  unsigned r = i / positions, p = positionStart + i % positions;
  unsigned kx = r % kw, t = r / kw, ky = t % kh, c = t / kh;
  unsigned oy = p / ow, ox = p % ow;
  int iy = (int)oy * sh - ph + (int)ky * dh;
  int ix = (int)ox * sw - pw + (int)kx * dw;
  __half v = __float2half_rn(0.f);
  if (iy >= 0 && iy < (int)h && ix >= 0 && ix < (int)w)
    v = x[(((unsigned long long)batch * ci + group * cg + c) * h + iy) * w + ix];
  col[i] = v;
}

// im2col for `samples` batch entries at once (one group): col[r][s * positions + p], r in [0, ci*kh*kw), all output positions.
__global__ void h_im2col_batch(const __half* x, __half* col, unsigned count,
                               unsigned ci, unsigned h, unsigned w, unsigned kh, unsigned kw,
                               unsigned oh, unsigned ow, int sh, int sw, int ph, int pw, int dh, int dw,
                               unsigned batchStart, unsigned samples) {
  GRID_INDEX(i, count)
  unsigned positions = oh * ow, rowLength = samples * positions;
  unsigned r = i / rowLength, q = i % rowLength;
  unsigned s = q / positions, p = q % positions;
  unsigned kx = r % kw, t = r / kw, ky = t % kh, c = t / kh;
  unsigned oy = p / ow, ox = p % ow;
  int iy = (int)oy * sh - ph + (int)ky * dh;
  int ix = (int)ox * sw - pw + (int)kx * dw;
  __half v = __float2half_rn(0.f);
  if (iy >= 0 && iy < (int)h && ix >= 0 && ix < (int)w)
    v = x[(((unsigned long long)(batchStart + s) * ci + c) * h + iy) * w + ix];
  col[i] = v;
}

// Division-free shared-memory 3x3 (stride=1, padding=1, dilation=1, groups=1) im2col:
// grid = (tilesX * tilesY, ci, samples), block = 256 (32x8 threads covering a 32x16 spatial tile).
__global__ void h_im2col_3x3_s1_p1_batch(const __half* x, __half* col,
                                         unsigned ci, unsigned h, unsigned w,
                                         unsigned batchStart, unsigned samples) {
  __shared__ __half tile[18][34];
  unsigned tilesX = (w + 31u) >> 5;
  unsigned tileIdx = blockIdx.x, c = blockIdx.y, s = blockIdx.z;
  unsigned tileY0 = (tileIdx / tilesX) << 4;
  unsigned tileX0 = (tileIdx % tilesX) << 5;
  const __half* srcPlane = x + (((unsigned long long)(batchStart + s) * ci + c) * h) * w;
  __half zero = __float2half_rn(0.f);
  for (unsigned idx = threadIdx.x; idx < 18u * 34u; idx += 256u) {
    unsigned sy = idx / 34u, sx = idx % 34u;
    int iy = (int)tileY0 + (int)sy - 1;
    int ix = (int)tileX0 + (int)sx - 1;
    tile[sy][sx] = (iy >= 0 && iy < (int)h && ix >= 0 && ix < (int)w)
      ? srcPlane[(unsigned long long)iy * w + (unsigned)ix]
      : zero;
  }
  __syncthreads();
  unsigned tx = threadIdx.x & 31u, ty = threadIdx.x >> 5;
  unsigned ox = tileX0 + tx, oy0 = tileY0 + ty, oy1 = oy0 + 8u;
  if (ox < w) {
    unsigned long long hw = (unsigned long long)h * w;
    unsigned long long rowLength = (unsigned long long)samples * hw;
    unsigned long long sampleOff = (unsigned long long)s * hw;
    __half* colBase = col + (unsigned long long)c * 9ULL * rowLength + sampleOff;
    bool valid0 = oy0 < h;
    bool valid1 = oy1 < h;
    unsigned long long p0 = (unsigned long long)oy0 * w + ox;
    unsigned long long p1 = (unsigned long long)oy1 * w + ox;
    #pragma unroll
    for (int ky = 0; ky < 3; ++ky) {
      #pragma unroll
      for (int kx = 0; kx < 3; ++kx) {
        unsigned long long kOff = (unsigned long long)(ky * 3 + kx) * rowLength;
        if (valid0) colBase[kOff + p0] = tile[ty + ky][tx + kx];
        if (valid1) colBase[kOff + p1] = tile[ty + 8u + ky][tx + kx];
      }
    }
  }
}

// In-place softmax over the last axis of [rows, width], one block per row.
__global__ void h_softmax(__half* x, unsigned rows, unsigned width) {
  __shared__ float shared[32];
  unsigned row = blockIdx.x;
  if (row >= rows) return;
  __half* base = x + (unsigned long long)row * width;
  unsigned per = width / blockDim.x;
  if (width % blockDim.x == 0 && per >= 8 && per <= 16) { // shorter rows measured slower this way
    float local[16];
    float tm = -3.402823466e+38f;
    #pragma unroll
    for (unsigned c = 0; c < 16; ++c) if (c < per) { float v = ld(base, threadIdx.x + c * blockDim.x); local[c] = v; tm = fmaxf(tm, v); }
    float m = block_max(tm, shared);
    float ts = 0.f;
    #pragma unroll
    for (unsigned c = 0; c < 16; ++c) if (c < per) { float e = __expf(local[c] - m); local[c] = e; ts += e; }
    float inv = 1.f / block_sum(ts, shared);
    #pragma unroll
    for (unsigned c = 0; c < 16; ++c) if (c < per) st(base, threadIdx.x + c * blockDim.x, local[c] * inv);
    return;
  }
  float tm = -3.402823466e+38f, ts = 0.f;
  for (unsigned i = threadIdx.x; i < width; i += blockDim.x) {
    float v = ld(base, i);
    if (v > tm) { ts = ts * __expf(tm - v) + 1.f; tm = v; }
    else ts += __expf(v - tm);
  }
  float m = block_max(tm, shared);
  float s = block_sum(ts * __expf(tm - m), shared);
  float inv = 1.f / s;
  for (unsigned i = threadIdx.x; i < width; i += blockDim.x) st(base, i, __expf(ld(base, i) - m) * inv);
}

__global__ void h_gelu_erf(const __half* a, __half* out, unsigned n) {
  GRID_INDEX(i, n) { float x = ld(a, i); st(out, i, 0.5f * x * (1.f + erff(x * 0.70710678118654752f))); }
}

__global__ void h_quick_gelu(const __half* a, __half* out, unsigned n) {
  GRID_INDEX(i, n) { float x = ld(a, i); st(out, i, x / (1.f + __expf(-1.702f * x))); }
}

// GPU-native nearest-exact token downsampling matching CPU TokenDownsampling.resampleTokens.
// Grid: batch * dstH * dstW blocks, 256 threads per block.
__global__ void h_downsample_tokens_exact(const __half* in, __half* out,
                                          unsigned batch, unsigned srcH, unsigned srcW,
                                          unsigned dstH, unsigned dstW, unsigned channels) {
  unsigned dstSeq = dstH * dstW;
  unsigned tok = blockIdx.x;
  if (tok >= batch * dstSeq) return;
  unsigned b = tok / dstSeq, rem = tok % dstSeq;
  unsigned ty = rem / dstW, tx = rem % dstW;
  unsigned sy = min((unsigned)floor(((double)ty + 0.5) * (double)srcH / (double)dstH), srcH - 1u);
  unsigned sx = min((unsigned)floor(((double)tx + 0.5) * (double)srcW / (double)dstW), srcW - 1u);
  unsigned srcSeq = srcH * srcW;
  const __half* src = in + (((unsigned long long)b * srcSeq + (unsigned long long)sy * srcW + sx) * channels);
  __half* dst = out + ((unsigned long long)tok * channels);
  if (aligned4(src) && aligned4(dst) && ((channels & 1u) == 0u)) {
    const __half2* s2 = (const __half2*)src;
    __half2* d2 = (__half2*)dst;
    unsigned c2 = channels >> 1;
    for (unsigned i = threadIdx.x; i < c2; i += blockDim.x) d2[i] = s2[i];
  } else {
    for (unsigned i = threadIdx.x; i < channels; i += blockDim.x) dst[i] = src[i];
  }
}

// TaylorSeer finite-difference kernels (caller launches (n + 1) / 2 threads).
__global__ void h_taylor_diff(const __half* cur, const __half* prev, __half* out, unsigned n, float invStep) {
  unsigned j = (blockIdx.x * blockDim.x + threadIdx.x) << 1;
  if (j >= n) return;
  if (j + 1u < n && aligned4(cur) && aligned4(prev) && aligned4(out)) {
    float2 fc = __half22float2(((const __half2*)cur)[j >> 1]);
    float2 fp = __half22float2(((const __half2*)prev)[j >> 1]);
    ((__half2*)out)[j >> 1] = __floats2half2_rn((fc.x - fp.x) * invStep, (fc.y - fp.y) * invStep);
  } else {
    st(out, j, (ld(cur, j) - ld(prev, j)) * invStep);
    if (j + 1u < n) st(out, j + 1u, (ld(cur, j + 1u) - ld(prev, j + 1u)) * invStep);
  }
}

__global__ void h_taylor_forecast1(const __half* f0, const __half* d1, __half* out, unsigned n, float c1) {
  unsigned j = (blockIdx.x * blockDim.x + threadIdx.x) << 1;
  if (j >= n) return;
  if (j + 1u < n && aligned4(f0) && aligned4(d1) && aligned4(out)) {
    float2 a = __half22float2(((const __half2*)f0)[j >> 1]);
    float2 b = __half22float2(((const __half2*)d1)[j >> 1]);
    ((__half2*)out)[j >> 1] = __floats2half2_rn(a.x + c1 * b.x, a.y + c1 * b.y);
  } else {
    st(out, j, ld(f0, j) + c1 * ld(d1, j));
    if (j + 1u < n) st(out, j + 1u, ld(f0, j + 1u) + c1 * ld(d1, j + 1u));
  }
}

__global__ void h_taylor_forecast2(const __half* f0, const __half* d1, const __half* d2, __half* out,
                                   unsigned n, float c1, float c2) {
  unsigned j = (blockIdx.x * blockDim.x + threadIdx.x) << 1;
  if (j >= n) return;
  if (j + 1u < n && aligned4(f0) && aligned4(d1) && aligned4(d2) && aligned4(out)) {
    float2 a = __half22float2(((const __half2*)f0)[j >> 1]);
    float2 b = __half22float2(((const __half2*)d1)[j >> 1]);
    float2 d = __half22float2(((const __half2*)d2)[j >> 1]);
    ((__half2*)out)[j >> 1] = __floats2half2_rn(a.x + c1 * b.x + c2 * d.x, a.y + c1 * b.y + c2 * d.y);
  } else {
    st(out, j, ld(f0, j) + c1 * ld(d1, j) + c2 * ld(d2, j));
    if (j + 1u < n) st(out, j + 1u, ld(f0, j + 1u) + c1 * ld(d1, j + 1u) + c2 * ld(d2, j + 1u));
  }
}

// T-GATE cross-attention cache formation and broadcast residual add (caller launches (outElems + 1) / 2 or (n + 1) / 2).
__global__ void h_tgate_avg_half(const __half* in, __half* out, unsigned outElems, unsigned hasPair) {
  unsigned j = (blockIdx.x * blockDim.x + threadIdx.x) << 1;
  if (j >= outElems) return;
  if (!hasPair) {
    if (j + 1u < outElems && aligned4(in) && aligned4(out)) {
      ((__half2*)out)[j >> 1] = ((const __half2*)in)[j >> 1];
    } else {
      out[j] = in[j];
      if (j + 1u < outElems) out[j + 1u] = in[j + 1u];
    }
    return;
  }
  const __half* in2 = in + outElems;
  if (j + 1u < outElems && aligned4(in) && aligned4(in2) && aligned4(out)) {
    float2 f0 = __half22float2(((const __half2*)in)[j >> 1]);
    float2 f1 = __half22float2(((const __half2*)in2)[j >> 1]);
    ((__half2*)out)[j >> 1] = __floats2half2_rn(0.5f * (f0.x + f1.x), 0.5f * (f0.y + f1.y));
  } else {
    st(out, j, 0.5f * (ld(in, j) + ld(in2, j)));
    if (j + 1u < outElems) st(out, j + 1u, 0.5f * (ld(in, j + 1u) + ld(in2, j + 1u)));
  }
}

__global__ void h_tgate_add_broadcast(const __half* self, const __half* cached, __half* out,
                                      unsigned n, unsigned cachedElems) {
  unsigned j = (blockIdx.x * blockDim.x + threadIdx.x) << 1;
  if (j >= n) return;
  if (j + 1u < n && ((cachedElems & 1u) == 0u) && aligned4(self) && aligned4(cached) && aligned4(out)) {
    unsigned cj = (j >= cachedElems) ? (j % cachedElems) : j;
    float2 fs = __half22float2(((const __half2*)self)[j >> 1]);
    float2 fc = __half22float2(((const __half2*)cached)[cj >> 1]);
    ((__half2*)out)[j >> 1] = __floats2half2_rn(fs.x + fc.x, fs.y + fc.y);
  } else {
    unsigned c0 = (j >= cachedElems) ? (j % cachedElems) : j;
    st(out, j, ld(self, j) + ld(cached, c0));
    if (j + 1u < n) {
      unsigned j1 = j + 1u;
      unsigned c1 = (j1 >= cachedElems) ? (j1 % cachedElems) : j1;
      st(out, j1, ld(self, j1) + ld(cached, c1));
    }
  }
}

// Attention without a score matrix in device memory (tensor cores, one pass over the keys, everything in registers).
} // extern "C": the template below needs C++ linkage
__device__ __forceinline__ float att_group_max(float v) {
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
  return fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
}
__device__ __forceinline__ float att_group_sum(float v) {
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  return v + __shfl_xor_sync(0xffffffffu, v, 2);
}
template <int R, int C>
__device__ __forceinline__ void att_body(const __half* q, const __half* k, const __half* v, __half* out,
                                         unsigned heads, unsigned T, unsigned S, unsigned D, unsigned Dv, float scale,
                                         unsigned H, unsigned strideIn, unsigned strideOut, float* scores, __half* tile) {
  using namespace nvcuda;
  unsigned groupsPerHead = T / (16 * R);
  unsigned head = blockIdx.x / groupsPerHead, row0 = (blockIdx.x % groupsPerHead) * 16 * R;
  if (head >= heads) return;
  unsigned lane = threadIdx.x & 31, row = lane & 15, col0 = (lane >> 4) * 8;
  unsigned strideQ = H ? (strideIn ? strideIn : H * D) : D, strideV = H ? (strideIn ? strideIn : H * Dv) : Dv;
  unsigned strideO = H ? (strideOut ? strideOut : H * Dv) : Dv;
  unsigned long long member = H ? head / H : 0, within = H ? head % H : 0;
  const __half* kh = H ? k + member * S * strideQ + within * D : k + (unsigned long long)head * S * D;
  const __half* vh = H ? v + member * S * strideV + within * Dv : v + (unsigned long long)head * S * Dv;
  const __half* qBase = H ? q + member * T * strideQ + within * D : q + (unsigned long long)head * T * D;
  __half* outBase = H ? out + member * T * strideO + within * Dv : out + (unsigned long long)head * T * Dv;
  unsigned tail = D % 16, tailV = Dv % 16;

  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> qa[R][C];
  #pragma unroll
  for (int r = 0; r < R; ++r) {
    const __half* qh = qBase + (unsigned long long)(row0 + r * 16) * strideQ;
    #pragma unroll
    for (int c = 0; c < C; ++c) {
      if (c == C - 1 && tail != 0) {
        for (unsigned j = 0; j < 8; ++j) {
          unsigned cc = col0 + j;
          tile[row * 16 + cc] = cc < 16 - tail ? __float2half_rn(0.f) : qh[row * strideQ + (D - 16) + cc];
        }
        __syncwarp();
        wmma::load_matrix_sync(qa[r][c], tile, 16);
        __syncwarp();
      } else wmma::load_matrix_sync(qa[r][c], qh + c * 16, strideQ);
    }
  }

  wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> kb[C];
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> vb[C];
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> pa;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> result[R][C];
  float maxLow[R], maxHigh[R], sumLow[R], sumHigh[R];
  #pragma unroll
  for (int r = 0; r < R; ++r) {
    #pragma unroll
    for (int c = 0; c < C; ++c) wmma::fill_fragment(result[r][c], 0.f);
    maxLow[r] = -3.402823466e+38f; maxHigh[r] = -3.402823466e+38f; sumLow[r] = 0.f; sumHigh[r] = 0.f;
  }

  for (unsigned key0 = 0; key0 < S; key0 += 16) {
    bool partial = key0 + 16 > S;
    #pragma unroll
    for (int c = 0; c < C; ++c) {
      unsigned d0 = (c == C - 1 && tail != 0) ? D - 16 : c * 16;
      if (partial) {
        for (unsigned j = 0; j < 8; ++j) {
          unsigned cc = col0 + j;
          tile[row * 16 + cc] = key0 + row < S ? kh[(unsigned long long)(key0 + row) * strideQ + d0 + cc] : __float2half_rn(0.f);
        }
        __syncwarp();
        wmma::load_matrix_sync(kb[c], tile, 16);
        __syncwarp();
      } else wmma::load_matrix_sync(kb[c], kh + (unsigned long long)key0 * strideQ + d0, strideQ);
    }
    #pragma unroll
    for (int c = 0; c < C; ++c) {
      unsigned e0v = (c == C - 1 && tailV != 0) ? Dv - 16 : c * 16;
      if (partial) {
        for (unsigned j = 0; j < 8; ++j) {
          unsigned cc = col0 + j;
          tile[row * 16 + cc] = key0 + row < S ? vh[(unsigned long long)(key0 + row) * strideV + e0v + cc] : __float2half_rn(0.f);
        }
        __syncwarp();
        wmma::load_matrix_sync(vb[c], tile, 16);
        __syncwarp();
      } else wmma::load_matrix_sync(vb[c], vh + (unsigned long long)key0 * strideV + e0v, strideV);
    }
    #pragma unroll
    for (int r = 0; r < R; ++r) {
      wmma::fill_fragment(acc, 0.f);
      #pragma unroll
      for (int c = 0; c < C; ++c) wmma::mma_sync(acc, qa[r][c], kb[c], acc);
      float s0 = acc.x[0] * scale, s1 = acc.x[1] * scale, s2 = acc.x[2] * scale, s3 = acc.x[3] * scale;
      float s4 = acc.x[4] * scale, s5 = acc.x[5] * scale, s6 = acc.x[6] * scale, s7 = acc.x[7] * scale;
      if (partial) {
        unsigned valid = S - key0, t2 = (lane & 3) * 2;
        const float none = -3.402823466e+38f;
        if (t2 >= valid) { s0 = none; s2 = none; }
        if (t2 + 1 >= valid) { s1 = none; s3 = none; }
        if (t2 + 8 >= valid) { s4 = none; s6 = none; }
        if (t2 + 9 >= valid) { s5 = none; s7 = none; }
      }
      float newLow = fmaxf(maxLow[r], att_group_max(fmaxf(fmaxf(s0, s1), fmaxf(s4, s5))));
      float newHigh = fmaxf(maxHigh[r], att_group_max(fmaxf(fmaxf(s2, s3), fmaxf(s6, s7))));
      float downLow = __expf(maxLow[r] - newLow), downHigh = __expf(maxHigh[r] - newHigh);
      maxLow[r] = newLow; maxHigh[r] = newHigh;
      sumLow[r] *= downLow; sumHigh[r] *= downHigh;
      #pragma unroll
      for (int c = 0; c < C; ++c) {
        result[r][c].x[0] *= downLow; result[r][c].x[1] *= downLow; result[r][c].x[4] *= downLow; result[r][c].x[5] *= downLow;
        result[r][c].x[2] *= downHigh; result[r][c].x[3] *= downHigh; result[r][c].x[6] *= downHigh; result[r][c].x[7] *= downHigh;
      }
      __half e0 = __float2half_rn(__expf(s0 - newLow)), e1 = __float2half_rn(__expf(s1 - newLow));
      __half e4 = __float2half_rn(__expf(s4 - newLow)), e5 = __float2half_rn(__expf(s5 - newLow));
      __half e2 = __float2half_rn(__expf(s2 - newHigh)), e3 = __float2half_rn(__expf(s3 - newHigh));
      __half e6 = __float2half_rn(__expf(s6 - newHigh)), e7 = __float2half_rn(__expf(s7 - newHigh));
      pa.x[0] = e0; pa.x[1] = e1; pa.x[2] = e2; pa.x[3] = e3; pa.x[4] = e4; pa.x[5] = e5; pa.x[6] = e6; pa.x[7] = e7;
      sumLow[r] += __half2float(e0) + __half2float(e1) + __half2float(e4) + __half2float(e5);
      sumHigh[r] += __half2float(e2) + __half2float(e3) + __half2float(e6) + __half2float(e7);
      #pragma unroll
      for (int c = 0; c < C; ++c) wmma::mma_sync(result[r][c], pa, vb[c], result[r][c]);
    }
  }

  #pragma unroll
  for (int r = 0; r < R; ++r) {
    float invLow = 1.f / att_group_sum(sumLow[r]), invHigh = 1.f / att_group_sum(sumHigh[r]);
    __half* oh = outBase + (unsigned long long)(row0 + r * 16) * strideO;
    #pragma unroll
    for (int c = 0; c < C; ++c) {
      bool last = c == C - 1 && tailV != 0;
      unsigned e0v = last ? Dv - 16 : c * 16, skip = last ? 16 - tailV : 0;
      result[r][c].x[0] *= invLow; result[r][c].x[1] *= invLow; result[r][c].x[4] *= invLow; result[r][c].x[5] *= invLow;
      result[r][c].x[2] *= invHigh; result[r][c].x[3] *= invHigh; result[r][c].x[6] *= invHigh; result[r][c].x[7] *= invHigh;
      wmma::store_matrix_sync(scores, result[r][c], 16, wmma::mem_row_major);
      __syncwarp();
      for (unsigned j = 0; j < 8; ++j) {
        unsigned cc = col0 + j;
        if (cc >= skip) oh[row * strideO + e0v + cc] = __float2half_rn(scores[row * 16 + cc]);
      }
      __syncwarp();
    }
  }
}

// FlashAttention-2 style multi-warp Q-split WMMA body (Dao, ICLR 2024 Algorithm 1):
// 4 warps share cooperative coalesced K/V shared-memory tiles with padded stride, exp2f scaling,
// max-unchanged rescale guard, and deferred partial-key boundary handling.
template <int R, int C, int WARPS, int LDK, bool PARTIAL>
__device__ __forceinline__ void fa2_step(
    const __half* kh, const __half* vh, unsigned S, unsigned D, unsigned Dv,
    unsigned strideQ, unsigned strideV, unsigned tail, unsigned tailV,
    float scaleLog2, unsigned key0, unsigned tid, unsigned lane,
    __half* smemK, __half* smemV,
    const nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> (&qa)[R][C],
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> (&result)[R][C],
    float (&maxLow)[R], float (&maxHigh)[R], float (&sumLow)[R], float (&sumHigh)[R]) {
  using namespace nvcuda;
  unsigned kr = tid >> 3, colStart = tid & 7;
  bool validRow = !PARTIAL || (key0 + kr < S);
  if (aligned4(kh) && ((strideQ & 1u) == 0u) && ((D & 1u) == 0u)) {
    unsigned d2 = D >> 1;
    const __half2* kSrc2 = (const __half2*)(kh + (unsigned long long)(key0 + kr) * strideQ);
    __half2* kDst2 = (__half2*)(smemK + kr * LDK);
    __half2 zero2 = __halves2half2(__float2half_rn(0.f), __float2half_rn(0.f));
    for (unsigned c2 = colStart; c2 < d2; c2 += 8u) kDst2[c2] = validRow ? kSrc2[c2] : zero2;
  } else {
    const __half* kSrc = kh + (unsigned long long)(key0 + kr) * strideQ;
    __half* kDst = smemK + kr * LDK;
    __half z = __float2half_rn(0.f);
    for (unsigned c0 = colStart; c0 < D; c0 += 8u) kDst[c0] = validRow ? kSrc[c0] : z;
  }
  if (aligned4(vh) && ((strideV & 1u) == 0u) && ((Dv & 1u) == 0u)) {
    unsigned dv2 = Dv >> 1;
    const __half2* vSrc2 = (const __half2*)(vh + (unsigned long long)(key0 + kr) * strideV);
    __half2* vDst2 = (__half2*)(smemV + kr * LDK);
    __half2 zero2 = __halves2half2(__float2half_rn(0.f), __float2half_rn(0.f));
    for (unsigned c2 = colStart; c2 < dv2; c2 += 8u) vDst2[c2] = validRow ? vSrc2[c2] : zero2;
  } else {
    const __half* vSrc = vh + (unsigned long long)(key0 + kr) * strideV;
    __half* vDst = smemV + kr * LDK;
    __half z = __float2half_rn(0.f);
    for (unsigned c0 = colStart; c0 < Dv; c0 += 8u) vDst[c0] = validRow ? vSrc[c0] : z;
  }
  __syncthreads();

  wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> kb[C];
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> vb[C];
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> pa;
  #pragma unroll
  for (int c = 0; c < C; ++c) {
    unsigned d0 = (c == C - 1 && tail != 0) ? D - 16 : c * 16;
    wmma::load_matrix_sync(kb[c], smemK + d0, LDK);
  }
  #pragma unroll
  for (int c = 0; c < C; ++c) {
    unsigned e0v = (c == C - 1 && tailV != 0) ? Dv - 16 : c * 16;
    wmma::load_matrix_sync(vb[c], smemV + e0v, LDK);
  }

  #pragma unroll
  for (int r = 0; r < R; ++r) {
    wmma::fill_fragment(acc, 0.f);
    #pragma unroll
    for (int c = 0; c < C; ++c) wmma::mma_sync(acc, qa[r][c], kb[c], acc);
    float s0 = acc.x[0] * scaleLog2, s1 = acc.x[1] * scaleLog2, s2 = acc.x[2] * scaleLog2, s3 = acc.x[3] * scaleLog2;
    float s4 = acc.x[4] * scaleLog2, s5 = acc.x[5] * scaleLog2, s6 = acc.x[6] * scaleLog2, s7 = acc.x[7] * scaleLog2;
    if (PARTIAL) {
      unsigned valid = S - key0, t2 = (lane & 3) * 2;
      const float none = -3.402823466e+38f;
      if (t2 >= valid) { s0 = none; s2 = none; }
      if (t2 + 1 >= valid) { s1 = none; s3 = none; }
      if (t2 + 8 >= valid) { s4 = none; s6 = none; }
      if (t2 + 9 >= valid) { s5 = none; s7 = none; }
    }
    float tileMaxLow = att_group_max(fmaxf(fmaxf(s0, s1), fmaxf(s4, s5)));
    if (tileMaxLow > maxLow[r]) {
      float downLow = fast_exp2f(maxLow[r] - tileMaxLow);
      maxLow[r] = tileMaxLow;
      sumLow[r] *= downLow;
      #pragma unroll
      for (int c = 0; c < C; ++c) {
        result[r][c].x[0] *= downLow; result[r][c].x[1] *= downLow;
        result[r][c].x[4] *= downLow; result[r][c].x[5] *= downLow;
      }
    }
    float tileMaxHigh = att_group_max(fmaxf(fmaxf(s2, s3), fmaxf(s6, s7)));
    if (tileMaxHigh > maxHigh[r]) {
      float downHigh = fast_exp2f(maxHigh[r] - tileMaxHigh);
      maxHigh[r] = tileMaxHigh;
      sumHigh[r] *= downHigh;
      #pragma unroll
      for (int c = 0; c < C; ++c) {
        result[r][c].x[2] *= downHigh; result[r][c].x[3] *= downHigh;
        result[r][c].x[6] *= downHigh; result[r][c].x[7] *= downHigh;
      }
    }
    float ml = maxLow[r], mh = maxHigh[r];
    __half e0 = __float2half_rn(fast_exp2f(s0 - ml)), e1 = __float2half_rn(fast_exp2f(s1 - ml));
    __half e4 = __float2half_rn(fast_exp2f(s4 - ml)), e5 = __float2half_rn(fast_exp2f(s5 - ml));
    __half e2 = __float2half_rn(fast_exp2f(s2 - mh)), e3 = __float2half_rn(fast_exp2f(s3 - mh));
    __half e6 = __float2half_rn(fast_exp2f(s6 - mh)), e7 = __float2half_rn(fast_exp2f(s7 - mh));
    pa.x[0] = e0; pa.x[1] = e1; pa.x[2] = e2; pa.x[3] = e3; pa.x[4] = e4; pa.x[5] = e5; pa.x[6] = e6; pa.x[7] = e7;
    sumLow[r] += __half2float(e0) + __half2float(e1) + __half2float(e4) + __half2float(e5);
    sumHigh[r] += __half2float(e2) + __half2float(e3) + __half2float(e6) + __half2float(e7);
    #pragma unroll
    for (int c = 0; c < C; ++c) wmma::mma_sync(result[r][c], pa, vb[c], result[r][c]);
  }
  __syncthreads();
}

template <int R, int C, int WARPS>
__device__ __forceinline__ void fa2_body(const __half* q, const __half* k, const __half* v, __half* out,
                                         unsigned heads, unsigned T, unsigned S, unsigned D, unsigned Dv, float scale,
                                         unsigned H, unsigned strideIn, unsigned strideOut) {
  using namespace nvcuda;
  constexpr int LDK = C * 16 + 8;
  __shared__ __align__(16) __half smemK[16 * LDK];
  __shared__ __align__(16) __half smemV[16 * LDK];
  __shared__ __align__(16) float warpScores[WARPS][256];

  unsigned groupsPerHead = T / (16u * WARPS * R);
  unsigned head = blockIdx.x / groupsPerHead;
  if (head >= heads) return;
  unsigned warpId = threadIdx.x >> 5, lane = threadIdx.x & 31, row = lane & 15, col0 = (lane >> 4) * 8;
  unsigned row0 = ((blockIdx.x % groupsPerHead) * WARPS + warpId) * 16u * R;
  unsigned strideQ = H ? (strideIn ? strideIn : H * D) : D, strideV = H ? (strideIn ? strideIn : H * Dv) : Dv;
  unsigned strideO = H ? (strideOut ? strideOut : H * Dv) : Dv;
  unsigned long long member = H ? head / H : 0, within = H ? head % H : 0;
  const __half* kh = H ? k + member * S * strideQ + within * D : k + (unsigned long long)head * S * D;
  const __half* vh = H ? v + member * S * strideV + within * Dv : v + (unsigned long long)head * S * Dv;
  const __half* qBase = H ? q + member * T * strideQ + within * D : q + (unsigned long long)head * T * D;
  __half* outBase = H ? out + member * T * strideO + within * Dv : out + (unsigned long long)head * T * Dv;
  unsigned tail = D % 16, tailV = Dv % 16;

  __half* warpTile = reinterpret_cast<__half*>(warpScores[warpId]);
  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> qa[R][C];
  #pragma unroll
  for (int r = 0; r < R; ++r) {
    const __half* qh = qBase + (unsigned long long)(row0 + r * 16) * strideQ;
    #pragma unroll
    for (int c = 0; c < C; ++c) {
      if (c == C - 1 && tail != 0) {
        for (unsigned j = 0; j < 8; ++j) {
          unsigned cc = col0 + j;
          warpTile[row * 16 + cc] = cc < 16 - tail ? __float2half_rn(0.f) : qh[row * strideQ + (D - 16) + cc];
        }
        __syncwarp();
        wmma::load_matrix_sync(qa[r][c], warpTile, 16);
        __syncwarp();
      } else wmma::load_matrix_sync(qa[r][c], qh + c * 16, strideQ);
    }
  }

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> result[R][C];
  float maxLow[R], maxHigh[R], sumLow[R], sumHigh[R];
  #pragma unroll
  for (int r = 0; r < R; ++r) {
    #pragma unroll
    for (int c = 0; c < C; ++c) wmma::fill_fragment(result[r][c], 0.f);
    maxLow[r] = -3.402823466e+38f; maxHigh[r] = -3.402823466e+38f; sumLow[r] = 0.f; sumHigh[r] = 0.f;
  }

  float scaleLog2 = scale * 1.4426950408889634f;
  unsigned key0 = 0;
  for (; key0 + 16u <= S; key0 += 16u) {
    fa2_step<R, C, WARPS, LDK, false>(kh, vh, S, D, Dv, strideQ, strideV, tail, tailV, scaleLog2,
      key0, threadIdx.x, lane, smemK, smemV, qa, result, maxLow, maxHigh, sumLow, sumHigh);
  }
  if (key0 < S) {
    fa2_step<R, C, WARPS, LDK, true>(kh, vh, S, D, Dv, strideQ, strideV, tail, tailV, scaleLog2,
      key0, threadIdx.x, lane, smemK, smemV, qa, result, maxLow, maxHigh, sumLow, sumHigh);
  }

  float* myScores = warpScores[warpId];
  #pragma unroll
  for (int r = 0; r < R; ++r) {
    float invLow = 1.f / att_group_sum(sumLow[r]), invHigh = 1.f / att_group_sum(sumHigh[r]);
    __half* oh = outBase + (unsigned long long)(row0 + r * 16) * strideO;
    #pragma unroll
    for (int c = 0; c < C; ++c) {
      bool last = c == C - 1 && tailV != 0;
      unsigned e0v = last ? Dv - 16 : c * 16, skip = last ? 16 - tailV : 0;
      result[r][c].x[0] *= invLow; result[r][c].x[1] *= invLow; result[r][c].x[4] *= invLow; result[r][c].x[5] *= invLow;
      result[r][c].x[2] *= invHigh; result[r][c].x[3] *= invHigh; result[r][c].x[6] *= invHigh; result[r][c].x[7] *= invHigh;
      wmma::store_matrix_sync(myScores, result[r][c], 16, wmma::mem_row_major);
      __syncwarp();
      for (unsigned j = 0; j < 8; ++j) {
        unsigned cc = col0 + j;
        if (cc >= skip) oh[row * strideO + e0v + cc] = __float2half_rn(myScores[row * 16 + cc]);
      }
      __syncwarp();
    }
  }
}

// SageAttention (ICLR 2025) + SpargeAttn (ICML 2025) multi-warp WMMA step:
// - Q/K quantized to INT8 with per-16-row scale -> s8*s8->s32 WMMA tensor cores (2x-4x on Ampere)
// - Warp-level online SpargeAttn PV-skip when all rows in the warp tile fall `skipDelta` below running max (log2 scale)
// - PV matrix multiply uses FP16-accumulator WMMA (2x throughput on RTX 3090) summed into FP32 running accumulators
template <int R, int C, int WARPS, int LDK_I8, int LDK_V, bool PARTIAL>
__device__ __forceinline__ void sage_step(
    const signed char* kI8Head, const __half* vh, unsigned S, unsigned S_pad, unsigned D, unsigned Dv,
    unsigned strideV, unsigned tailV, float scaleLog2, float kScaleVal, float skipDelta,
    unsigned key0, unsigned tid, unsigned lane,
    signed char* smemK, __half* smemV,
    const float (&qScale)[R],
    const nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, signed char, nvcuda::wmma::row_major> (&qa)[R][C],
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> (&result)[R][C],
    float (&maxLow)[R], float (&maxHigh)[R], float (&sumLow)[R], float (&sumHigh)[R]) {
  using namespace nvcuda;
  constexpr int D_PAD = C * 16;
  unsigned kr = tid >> 3, colStart = tid & 7;
  bool validRow = !PARTIAL || (key0 + kr < S);

  const uint32_t* kSrc4 = (const uint32_t*)(kI8Head + (unsigned long long)(key0 + kr) * (unsigned)D_PAD);
  uint32_t* kDst4 = (uint32_t*)(smemK + kr * LDK_I8);
  constexpr unsigned d4 = D_PAD >> 2;
  #pragma unroll
  for (unsigned c4 = colStart; c4 < d4; c4 += 8u) kDst4[c4] = validRow ? kSrc4[c4] : 0u;

  if (aligned4(vh) && ((strideV & 1u) == 0u) && ((Dv & 1u) == 0u)) {
    unsigned dv2 = Dv >> 1;
    const __half2* vSrc2 = (const __half2*)(vh + (unsigned long long)(key0 + kr) * strideV);
    __half2* vDst2 = (__half2*)(smemV + kr * LDK_V);
    __half2 zero2 = __halves2half2(__float2half_rn(0.f), __float2half_rn(0.f));
    for (unsigned c2 = colStart; c2 < dv2; c2 += 8u) vDst2[c2] = validRow ? vSrc2[c2] : zero2;
  } else {
    const __half* vSrc = vh + (unsigned long long)(key0 + kr) * strideV;
    __half* vDst = smemV + kr * LDK_V;
    __half z = __float2half_rn(0.f);
    for (unsigned c0 = colStart; c0 < Dv; c0 += 8u) vDst[c0] = validRow ? vSrc[c0] : z;
  }
  __syncthreads();

  wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::col_major> kb[C];
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> vb[C];
  wmma::fragment<wmma::accumulator, 16, 16, 16, int> acci;
  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> pa;
  wmma::fragment<wmma::accumulator, 16, 16, 16, __half> pvt;
  #pragma unroll
  for (int c = 0; c < C; ++c) wmma::load_matrix_sync(kb[c], smemK + c * 16, LDK_I8);
  #pragma unroll
  for (int c = 0; c < C; ++c) {
    unsigned e0v = (c == C - 1 && tailV != 0) ? Dv - 16 : c * 16;
    wmma::load_matrix_sync(vb[c], smemV + e0v, LDK_V);
  }

  float kScaleLog2 = kScaleVal * scaleLog2;
  #pragma unroll
  for (int r = 0; r < R; ++r) {
    wmma::fill_fragment(acci, 0);
    #pragma unroll
    for (int c = 0; c < C; ++c) wmma::mma_sync(acci, qa[r][c], kb[c], acci);
    float qkScale = qScale[r] * kScaleLog2;
    float s0 = (float)acci.x[0] * qkScale, s1 = (float)acci.x[1] * qkScale;
    float s2 = (float)acci.x[2] * qkScale, s3 = (float)acci.x[3] * qkScale;
    float s4 = (float)acci.x[4] * qkScale, s5 = (float)acci.x[5] * qkScale;
    float s6 = (float)acci.x[6] * qkScale, s7 = (float)acci.x[7] * qkScale;
    if (PARTIAL) {
      unsigned valid = S - key0, t2 = (lane & 3) * 2;
      const float none = -3.402823466e+38f;
      if (t2 >= valid) { s0 = none; s2 = none; }
      if (t2 + 1 >= valid) { s1 = none; s3 = none; }
      if (t2 + 8 >= valid) { s4 = none; s6 = none; }
      if (t2 + 9 >= valid) { s5 = none; s7 = none; }
    }
    float tileMaxLow = att_group_max(fmaxf(fmaxf(s0, s1), fmaxf(s4, s5)));
    float tileMaxHigh = att_group_max(fmaxf(fmaxf(s2, s3), fmaxf(s6, s7)));
    if (skipDelta > 0.f && __all_sync(0xffffffffu, tileMaxLow < maxLow[r] - skipDelta && tileMaxHigh < maxHigh[r] - skipDelta)) {
      continue;
    }
    if (tileMaxLow > maxLow[r]) {
      float downLow = fast_exp2f(maxLow[r] - tileMaxLow);
      maxLow[r] = tileMaxLow;
      sumLow[r] *= downLow;
      #pragma unroll
      for (int c = 0; c < C; ++c) {
        result[r][c].x[0] *= downLow; result[r][c].x[1] *= downLow;
        result[r][c].x[4] *= downLow; result[r][c].x[5] *= downLow;
      }
    }
    if (tileMaxHigh > maxHigh[r]) {
      float downHigh = fast_exp2f(maxHigh[r] - tileMaxHigh);
      maxHigh[r] = tileMaxHigh;
      sumHigh[r] *= downHigh;
      #pragma unroll
      for (int c = 0; c < C; ++c) {
        result[r][c].x[2] *= downHigh; result[r][c].x[3] *= downHigh;
        result[r][c].x[6] *= downHigh; result[r][c].x[7] *= downHigh;
      }
    }
    float ml = maxLow[r], mh = maxHigh[r];
    __half e0 = __float2half_rn(fast_exp2f(s0 - ml)), e1 = __float2half_rn(fast_exp2f(s1 - ml));
    __half e4 = __float2half_rn(fast_exp2f(s4 - ml)), e5 = __float2half_rn(fast_exp2f(s5 - ml));
    __half e2 = __float2half_rn(fast_exp2f(s2 - mh)), e3 = __float2half_rn(fast_exp2f(s3 - mh));
    __half e6 = __float2half_rn(fast_exp2f(s6 - mh)), e7 = __float2half_rn(fast_exp2f(s7 - mh));
    pa.x[0] = e0; pa.x[1] = e1; pa.x[2] = e2; pa.x[3] = e3; pa.x[4] = e4; pa.x[5] = e5; pa.x[6] = e6; pa.x[7] = e7;
    sumLow[r] += __half2float(e0) + __half2float(e1) + __half2float(e4) + __half2float(e5);
    sumHigh[r] += __half2float(e2) + __half2float(e3) + __half2float(e6) + __half2float(e7);
    #pragma unroll
    for (int c = 0; c < C; ++c) {
      wmma::fill_fragment(pvt, __float2half_rn(0.f));
      wmma::mma_sync(pvt, pa, vb[c], pvt);
      result[r][c].x[0] += __half2float(pvt.x[0]); result[r][c].x[1] += __half2float(pvt.x[1]);
      result[r][c].x[2] += __half2float(pvt.x[2]); result[r][c].x[3] += __half2float(pvt.x[3]);
      result[r][c].x[4] += __half2float(pvt.x[4]); result[r][c].x[5] += __half2float(pvt.x[5]);
      result[r][c].x[6] += __half2float(pvt.x[6]); result[r][c].x[7] += __half2float(pvt.x[7]);
    }
  }
  __syncthreads();
}

template <int R, int C, int WARPS>
__device__ __forceinline__ void sage_body(const __half* q, const signed char* kI8, const float* kScale, const __half* v, __half* out,
                                          unsigned heads, unsigned T, unsigned S, unsigned S_pad, unsigned D, unsigned Dv, float scale,
                                          float skipDelta, unsigned H, unsigned strideIn, unsigned strideOut) {
  using namespace nvcuda;
  constexpr int D_PAD = C * 16;
  constexpr int LDK_I8 = C * 16 + 16;
  constexpr int LDK_V = C * 16 + 8;
  __shared__ __align__(16) signed char smemK[16 * LDK_I8];
  __shared__ __align__(16) __half smemV[16 * LDK_V];
  __shared__ __align__(16) float warpScores[WARPS][320];

  unsigned groupsPerHead = T / (16u * WARPS * R);
  unsigned head = blockIdx.x / groupsPerHead;
  if (head >= heads) return;
  unsigned warpId = threadIdx.x >> 5, lane = threadIdx.x & 31, row = lane & 15, col0 = (lane >> 4) * 8;
  unsigned row0 = ((blockIdx.x % groupsPerHead) * WARPS + warpId) * 16u * R;
  unsigned strideQ = H ? (strideIn ? strideIn : H * D) : D, strideV = H ? (strideIn ? strideIn : H * Dv) : Dv;
  unsigned strideO = H ? (strideOut ? strideOut : H * Dv) : Dv;
  unsigned long long member = H ? head / H : 0, within = H ? head % H : 0;
  const __half* qBase = H ? q + member * T * strideQ + within * D : q + (unsigned long long)head * T * D;
  const signed char* kI8Head = kI8 + (unsigned long long)head * S_pad * (unsigned)D_PAD;
  const float* kScaleHead = kScale + (unsigned long long)head * (S_pad >> 4);
  const __half* vh = H ? v + member * S * strideV + within * Dv : v + (unsigned long long)head * S * Dv;
  __half* outBase = H ? out + member * T * strideO + within * Dv : out + (unsigned long long)head * T * Dv;
  unsigned tailV = Dv % 16;

  signed char* warpTileI8 = reinterpret_cast<signed char*>(warpScores[warpId]);
  float qScale[R];
  wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> qa[R][C];
  unsigned tr = lane >> 1, tc = lane & 1;
  #pragma unroll
  for (int r = 0; r < R; ++r) {
    const __half* qh = qBase + (unsigned long long)(row0 + r * 16) * strideQ;
    float localMax = 0.f;
    for (unsigned c = tc; c < D; c += 2u) {
      localMax = fmaxf(localMax, fabsf(__half2float(qh[(unsigned long long)tr * strideQ + c])));
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) localMax = fmaxf(localMax, __shfl_xor_sync(0xffffffffu, localMax, off));
    float qs = (localMax > 0.f) ? (localMax / 127.f) : 1.f;
    float qInv = (localMax > 0.f) ? (127.f / localMax) : 0.f;
    qScale[r] = qs;
    for (unsigned c = tc; c < (unsigned)D_PAD; c += 2u) {
      signed char qv = 0;
      if (c < D) {
        int iv = __float2int_rn(__half2float(qh[(unsigned long long)tr * strideQ + c]) * qInv);
        if (iv > 127) iv = 127; else if (iv < -127) iv = -127;
        qv = (signed char)iv;
      }
      warpTileI8[tr * (unsigned)D_PAD + c] = qv;
    }
    __syncwarp();
    #pragma unroll
    for (int c = 0; c < C; ++c) wmma::load_matrix_sync(qa[r][c], warpTileI8 + c * 16, D_PAD);
    __syncwarp();
  }

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> result[R][C];
  float maxLow[R], maxHigh[R], sumLow[R], sumHigh[R];
  #pragma unroll
  for (int r = 0; r < R; ++r) {
    #pragma unroll
    for (int c = 0; c < C; ++c) wmma::fill_fragment(result[r][c], 0.f);
    maxLow[r] = -3.402823466e+38f; maxHigh[r] = -3.402823466e+38f; sumLow[r] = 0.f; sumHigh[r] = 0.f;
  }

  float scaleLog2 = scale * 1.4426950408889634f;
  unsigned key0 = 0;
  for (; key0 + 16u <= S; key0 += 16u) {
    float kScaleVal = kScaleHead[key0 >> 4];
    sage_step<R, C, WARPS, LDK_I8, LDK_V, false>(kI8Head, vh, S, S_pad, D, Dv, strideV, tailV, scaleLog2,
      kScaleVal, skipDelta, key0, threadIdx.x, lane, smemK, smemV, qScale, qa, result, maxLow, maxHigh, sumLow, sumHigh);
  }
  if (key0 < S) {
    float kScaleVal = kScaleHead[key0 >> 4];
    sage_step<R, C, WARPS, LDK_I8, LDK_V, true>(kI8Head, vh, S, S_pad, D, Dv, strideV, tailV, scaleLog2,
      kScaleVal, skipDelta, key0, threadIdx.x, lane, smemK, smemV, qScale, qa, result, maxLow, maxHigh, sumLow, sumHigh);
  }

  float* myScores = warpScores[warpId];
  #pragma unroll
  for (int r = 0; r < R; ++r) {
    float invLow = 1.f / att_group_sum(sumLow[r]), invHigh = 1.f / att_group_sum(sumHigh[r]);
    __half* oh = outBase + (unsigned long long)(row0 + r * 16) * strideO;
    #pragma unroll
    for (int c = 0; c < C; ++c) {
      bool last = c == C - 1 && tailV != 0;
      unsigned e0v = last ? Dv - 16 : c * 16, skip = last ? 16 - tailV : 0;
      result[r][c].x[0] *= invLow; result[r][c].x[1] *= invLow; result[r][c].x[4] *= invLow; result[r][c].x[5] *= invLow;
      result[r][c].x[2] *= invHigh; result[r][c].x[3] *= invHigh; result[r][c].x[6] *= invHigh; result[r][c].x[7] *= invHigh;
      wmma::store_matrix_sync(myScores, result[r][c], 16, wmma::mem_row_major);
      __syncwarp();
      for (unsigned j = 0; j < 8; ++j) {
        unsigned cc = col0 + j;
        if (cc >= skip) oh[row * strideO + e0v + cc] = __float2half_rn(myScores[row * 16 + cc]);
      }
      __syncwarp();
    }
  }
}

extern "C" {
// One warp per block. Variants: query tiles per warp x largest number of 16-wide chunks (width up to 16 * chunks).
#define ATT_KERNEL(NAME, R, C) \
  __global__ void NAME(const __half* q, const __half* k, const __half* v, __half* out, \
                       unsigned heads, unsigned T, unsigned S, unsigned D, unsigned Dv, float scale, unsigned H, \
                       unsigned strideIn, unsigned strideOut) { \
    __shared__ float scores[256]; \
    __shared__ __half tile[256]; \
    att_body<R, C>(q, k, v, out, heads, T, S, D, Dv, scale, H, strideIn, strideOut, scores, tile); \
  }
ATT_KERNEL(h_attention_fused_r1c3, 1, 3)
ATT_KERNEL(h_attention_fused_r2c3, 2, 3)
ATT_KERNEL(h_attention_fused_r4c3, 4, 3)
ATT_KERNEL(h_attention_fused_r1c4, 1, 4)
ATT_KERNEL(h_attention_fused_r2c4, 2, 4)
ATT_KERNEL(h_attention_fused_r4c4, 4, 4)
ATT_KERNEL(h_attention_fused_r1c5, 1, 5)
ATT_KERNEL(h_attention_fused_r2c5, 2, 5)

// Multi-warp FlashAttention-2 variants (4 warps = 128 threads per CTA).
#define FA2_KERNEL(NAME, R, C, WARPS) \
  __global__ void NAME(const __half* q, const __half* k, const __half* v, __half* out, \
                       unsigned heads, unsigned T, unsigned S, unsigned D, unsigned Dv, float scale, unsigned H, \
                       unsigned strideIn, unsigned strideOut) { \
    fa2_body<R, C, WARPS>(q, k, v, out, heads, T, S, D, Dv, scale, H, strideIn, strideOut); \
  }
FA2_KERNEL(h_fa2_w4r1c3, 1, 3, 4)
FA2_KERNEL(h_fa2_w4r1c4, 1, 4, 4)
FA2_KERNEL(h_fa2_w4r1c5, 1, 5, 4)
FA2_KERNEL(h_fa2_w4r2c3, 2, 3, 4)
FA2_KERNEL(h_fa2_w4r2c4, 2, 4, 4)

#define SAGE_KERNEL(NAME, R, C, WARPS) \
  __global__ void NAME(const __half* q, const signed char* kI8, const float* kScale, const __half* v, __half* out, \
                       unsigned heads, unsigned T, unsigned S, unsigned S_pad, unsigned D, unsigned Dv, float scale, \
                       float skipDelta, unsigned H, unsigned strideIn, unsigned strideOut) { \
    sage_body<R, C, WARPS>(q, kI8, kScale, v, out, heads, T, S, S_pad, D, Dv, scale, skipDelta, H, strideIn, strideOut); \
  }
SAGE_KERNEL(h_sage_w4r1c3, 1, 3, 4)
SAGE_KERNEL(h_sage_w4r1c4, 1, 4, 4)
SAGE_KERNEL(h_sage_w4r1c5, 1, 5, 4)
SAGE_KERNEL(h_sage_w4r2c4, 2, 4, 4)

// SageAttention channel-wise K smoothing mean (one 256-thread block per head, coalesced along head dim D <= 80).
__global__ void h_sage_k_mean(const __half* k, float* kMean, unsigned heads, unsigned S, unsigned D,
                              unsigned H, unsigned strideIn) {
  __shared__ float warpAcc[8][80];
  unsigned head = blockIdx.x;
  if (head >= heads) return;
  unsigned warpId = threadIdx.x >> 5, lane = threadIdx.x & 31;
  unsigned strideQ = H ? (strideIn ? strideIn : H * D) : D;
  unsigned long long member = H ? head / H : 0, within = H ? head % H : 0;
  const __half* kh = H ? k + member * S * strideQ + within * D : k + (unsigned long long)head * S * D;
  for (unsigned pass = 0; pass * 32u < D; ++pass) {
    unsigned c = pass * 32u + lane;
    float sum = 0.f;
    if (c < D) {
      for (unsigned s = warpId; s < S; s += 8u) sum += ld(kh, (unsigned long long)s * strideQ + c);
      warpAcc[warpId][c] = sum;
    }
  }
  __syncthreads();
  float invS = S > 0u ? (1.f / (float)S) : 0.f;
  for (unsigned c = threadIdx.x; c < D; c += blockDim.x) {
    float tot = 0.f;
    #pragma unroll
    for (int w = 0; w < 8; ++w) tot += warpAcc[w][c];
    kMean[(unsigned long long)head * D + c] = tot * invS;
  }
}

// Subtract K channel mean and quantize 16-row tiles to contiguous [heads, S_pad, D_pad] INT8 + [heads, S_pad/16] scale.
__global__ void h_sage_quant_k(const __half* k, const float* kMean, signed char* kOut, float* kScale,
                               unsigned heads, unsigned S, unsigned S_pad, unsigned D, unsigned D_pad,
                               unsigned H, unsigned strideIn) {
  __shared__ float svals[16][80];
  __shared__ float tileMax[128];
  unsigned tileIdx = blockIdx.x, head = blockIdx.y;
  if (head >= heads) return;
  unsigned strideQ = H ? (strideIn ? strideIn : H * D) : D;
  unsigned long long member = H ? head / H : 0, within = H ? head % H : 0;
  const __half* kh = H ? k + member * S * strideQ + within * D : k + (unsigned long long)head * S * D;
  const float* meanH = kMean ? kMean + (unsigned long long)head * D : nullptr;
  unsigned tid = threadIdx.x;
  unsigned rInTile = tid >> 3, cLane = tid & 7;
  unsigned row = tileIdx * 16u + rInTile;
  float amax = 0.f;
  for (unsigned c = cLane; c < D; c += 8u) {
    float v = 0.f;
    if (row < S) {
      v = ld(kh, (unsigned long long)row * strideQ + c) - (meanH ? meanH[c] : 0.f);
      amax = fmaxf(amax, fabsf(v));
    }
    svals[rInTile][c] = v;
  }
  tileMax[tid] = amax;
  __syncthreads();
  for (unsigned s = 64u; s > 0u; s >>= 1) {
    if (tid < s) tileMax[tid] = fmaxf(tileMax[tid], tileMax[tid + s]);
    __syncthreads();
  }
  float gMax = tileMax[0];
  float scale = (gMax > 0.f) ? (gMax / 127.f) : 1.f;
  float invScale = (gMax > 0.f) ? (127.f / gMax) : 0.f;
  if (tid == 0) kScale[(unsigned long long)head * (S_pad >> 4) + tileIdx] = scale;
  signed char* dst = kOut + (((unsigned long long)head * S_pad + row) * D_pad);
  for (unsigned c = cLane; c < D_pad; c += 8u) {
    if (row < S && c < D) {
      int qv = __float2int_rn(svals[rInTile][c] * invScale);
      if (qv > 127) qv = 127; else if (qv < -127) qv = -127;
      dst[c] = (signed char)qv;
    } else {
      dst[c] = (signed char)0;
    }
  }
}

// W8A8 per-row symmetric INT8 activation quantization (`rows` blocks x 256 threads).
__global__ void h_quantize_rows_i8(const __half* x, signed char* q, float* scales, unsigned rows, unsigned cols) {
  __shared__ float shared[32];
  unsigned row = blockIdx.x;
  if (row >= rows) return;
  const __half* base = x + (unsigned long long)row * cols;
  signed char* dst = q + (unsigned long long)row * cols;
  float amax = 0.f;
  if (aligned4(base) && ((cols & 1u) == 0u)) {
    const __half2* b2 = (const __half2*)base;
    unsigned c2 = cols >> 1;
    for (unsigned i = threadIdx.x; i < c2; i += blockDim.x) {
      float2 f = __half22float2(b2[i]);
      amax = fmaxf(amax, fmaxf(fabsf(f.x), fabsf(f.y)));
    }
    float rowMax = block_max(amax, shared);
    float scale = (rowMax > 0.f) ? (rowMax / 127.f) : 1.f;
    float invScale = (rowMax > 0.f) ? (127.f / rowMax) : 0.f;
    if (threadIdx.x == 0) scales[row] = scale;
    if (aligned4(dst) && ((cols & 3u) == 0u)) {
      const __half2* b2p = (const __half2*)base;
      char2* d2 = (char2*)dst;
      for (unsigned i = threadIdx.x; i < c2; i += blockDim.x) {
        float2 f = __half22float2(b2p[i]);
        int q0 = __float2int_rn(f.x * invScale); if (q0 > 127) q0 = 127; else if (q0 < -127) q0 = -127;
        int q1 = __float2int_rn(f.y * invScale); if (q1 > 127) q1 = 127; else if (q1 < -127) q1 = -127;
        char2 pack; pack.x = (signed char)q0; pack.y = (signed char)q1;
        d2[i] = pack;
      }
    } else {
      for (unsigned i = threadIdx.x; i < cols; i += blockDim.x) {
        int qv = __float2int_rn(ld(base, i) * invScale);
        if (qv > 127) qv = 127; else if (qv < -127) qv = -127;
        dst[i] = (signed char)qv;
      }
    }
    return;
  }
  for (unsigned i = threadIdx.x; i < cols; i += blockDim.x) amax = fmaxf(amax, fabsf(ld(base, i)));
  float rowMax = block_max(amax, shared);
  float scale = (rowMax > 0.f) ? (rowMax / 127.f) : 1.f;
  float invScale = (rowMax > 0.f) ? (127.f / rowMax) : 0.f;
  if (threadIdx.x == 0) scales[row] = scale;
  for (unsigned i = threadIdx.x; i < cols; i += blockDim.x) {
    int qv = __float2int_rn(ld(base, i) * invScale);
    if (qv > 127) qv = 127; else if (qv < -127) qv = -127;
    dst[i] = (signed char)qv;
  }
}

// Dequantize Int32 GEMM output `acc [rows, cols]`, fold optional bias [cols] and residual [rows, cols].
__global__ void h_dequant_i32_bias_res(const int* acc, const float* scaleX, const float* scaleW,
                                       const __half* bias, const __half* residual, __half* out,
                                       unsigned n, unsigned cols) {
  unsigned j = (blockIdx.x * blockDim.x + threadIdx.x) << 1;
  if (j >= n) return;
  if (j + 1u < n && ((cols & 1u) == 0u) && aligned8(acc) && aligned8(scaleW) && aligned4(out) &&
      (!bias || aligned4(bias)) && (!residual || aligned4(residual))) {
    unsigned row = j / cols, col = j % cols;
    float sx = scaleX[row];
    int2 a2 = ((const int2*)acc)[j >> 1];
    float2 sw2 = ((const float2*)scaleW)[col >> 1];
    float r0 = (float)a2.x * sx * sw2.x;
    float r1 = (float)a2.y * sx * sw2.y;
    if (bias) {
      float2 b2 = __half22float2(((const __half2*)bias)[col >> 1]);
      r0 += b2.x; r1 += b2.y;
    }
    if (residual) {
      float2 res2 = __half22float2(((const __half2*)residual)[j >> 1]);
      r0 += res2.x; r1 += res2.y;
    }
    ((__half2*)out)[j >> 1] = __floats2half2_rn(r0, r1);
  } else {
    unsigned row0 = j / cols, col0 = j % cols;
    float r0 = (float)acc[j] * scaleX[row0] * scaleW[col0] + (bias ? ld(bias, col0) : 0.f) + (residual ? ld(residual, j) : 0.f);
    st(out, j, r0);
    if (j + 1u < n) {
      unsigned j1 = j + 1u, row1 = j1 / cols, col1 = j1 % cols;
      float r1 = (float)acc[j1] * scaleX[row1] * scaleW[col1] + (bias ? ld(bias, col1) : 0.f) + (residual ? ld(residual, j1) : 0.f);
      st(out, j1, r1);
    }
  }
}

// Dequantize Int32 FFN up-projection `acc [rows, 2*width]` + bias + GEGLU into `out [rows, width]`.
__global__ void h_dequant_i32_geglu_bias(const int* acc, const float* scaleX, const float* scaleW,
                                         const __half* bias, __half* out, unsigned n, unsigned width) {
  GRID_INDEX(i, n)
  unsigned row = i / width, col = i % width;
  float sx = scaleX[row];
  unsigned idxV = row * 2u * width + col;
  unsigned idxG = idxV + width;
  float v = (float)acc[idxV] * sx * scaleW[col] + (bias ? ld(bias, col) : 0.f);
  float g = (float)acc[idxG] * sx * scaleW[width + col] + (bias ? ld(bias, width + col) : 0.f);
  float gelu = 0.5f * g * (1.f + erff(g * 0.70710678118654752f));
  st(out, i, v * gelu);
}

// Where the elements of 16x16 fragments live: out[kind * 256 + lane * 8 + slot] = row * 16 + column, for kind 0 = accumulator
// (float), 1 = matrix_a (half, row major), 2 = matrix_b (half, column major), 3 = matrix_b (half, row major). One warp.
__global__ void h_wmma_layout(float* out) {
  using namespace nvcuda;
  __shared__ float indexF[256];
  __shared__ __half indexH[256];
  for (unsigned i = threadIdx.x; i < 256; i += 32) { indexF[i] = (float)i; indexH[i] = __float2half_rn((float)i); }
  __syncwarp();
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> bc;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> br;
  wmma::load_matrix_sync(c, indexF, 16, wmma::mem_row_major);
  wmma::load_matrix_sync(a, indexH, 16);
  wmma::load_matrix_sync(bc, indexH, 16);
  wmma::load_matrix_sync(br, indexH, 16);
  for (unsigned i = 0; i < 8; ++i) {
    out[0 * 256 + threadIdx.x * 8 + i] = c.x[i];
    out[1 * 256 + threadIdx.x * 8 + i] = __half2float(a.x[i]);
    out[2 * 256 + threadIdx.x * 8 + i] = __half2float(bc.x[i]);
    out[3 * 256 + threadIdx.x * 8 + i] = __half2float(br.x[i]);
  }
}

// Causal softmax over [rows, width]: row r attends to keys j <= (r % queries) + queryStart; the rest become 0.
__global__ void h_softmax_causal(__half* x, unsigned rows, unsigned width, unsigned queries, unsigned queryStart) {
  __shared__ float shared[32];
  unsigned row = blockIdx.x;
  if (row >= rows) return;
  __half* base = x + (unsigned long long)row * width;
  unsigned limit = min(width, (row % queries) + queryStart + 1);
  float m = -3.402823466e+38f;
  for (unsigned i = threadIdx.x; i < limit; i += blockDim.x) m = fmaxf(m, ld(base, i));
  m = block_max(m, shared);
  float s = 0.f;
  for (unsigned i = threadIdx.x; i < limit; i += blockDim.x) s += __expf(ld(base, i) - m);
  s = block_sum(s, shared);
  float inv = 1.f / s;
  for (unsigned i = threadIdx.x; i < width; i += blockDim.x)
    st(base, i, i < limit ? __expf(ld(base, i) - m) * inv : 0.f);
}

}
