// Float16 activation kernels for the CUDA diffusion runtime. Storage is __half; arithmetic is Float32.
// Compiled to ffm/src/main/resources/scalatorch/half.ptx by build_ptx.py.
#include <cuda_fp16.h>

#define GRID_INDEX(i, n) unsigned i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= (n)) return;

__device__ __forceinline__ float ld(const __half* p, unsigned i) { return __half2float(p[i]); }
__device__ __forceinline__ void st(__half* p, unsigned i, float v) { p[i] = __float2half_rn(v); }

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

__global__ void h_f32_to_f16(const float* in, __half* out, unsigned n) {
  GRID_INDEX(i, n) out[i] = __float2half_rn(in[i]);
}

__global__ void h_f16_to_f32(const __half* in, float* out, unsigned n) {
  GRID_INDEX(i, n) out[i] = __half2float(in[i]);
}

__global__ void h_fill(__half* out, unsigned n, float value) {
  GRID_INDEX(i, n) out[i] = __float2half_rn(value);
}

__global__ void h_add(const __half* a, const __half* b, __half* out, unsigned n) {
  GRID_INDEX(i, n) st(out, i, ld(a, i) + ld(b, i));
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
  GRID_INDEX(i, n) st(out, i, ld(a, i) * s);
}

__global__ void h_silu(const __half* a, __half* out, unsigned n) {
  GRID_INDEX(i, n) { float x = ld(a, i); st(out, i, x / (1.f + __expf(-x))); }
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
  for (unsigned i = threadIdx.x; i < width; i += blockDim.x) s += ld(base, i);
  float mean = block_sum(s, shared) / (float)width;
  float q = 0.f;
  for (unsigned i = threadIdx.x; i < width; i += blockDim.x) { float d = ld(base, i) - mean; q += d * d; }
  float var = block_sum(q, shared) / (float)width;
  if (threadIdx.x == 0) stats[row] = make_float2(mean, rsqrtf(var + eps));
}

__global__ void h_groupnorm_apply(const __half* x, const __half* weight, const __half* bias, __half* out,
                                  const float2* stats, unsigned n, unsigned width, unsigned channels, unsigned spatial) {
  GRID_INDEX(i, n)
  float2 s = stats[i / width];
  float v = (ld(x, i) - s.x) * s.y;
  unsigned c = (i / spatial) % channels;
  if (weight) v *= ld(weight, c);
  if (bias) v += ld(bias, c);
  st(out, i, v);
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

// In-place softmax over the last axis of [rows, width], one block per row.
__global__ void h_softmax(__half* x, unsigned rows, unsigned width) {
  __shared__ float shared[32];
  unsigned row = blockIdx.x;
  if (row >= rows) return;
  __half* base = x + (unsigned long long)row * width;
  // One read for the maximum and the sum together: each thread keeps a running maximum and rescales its sum when the maximum grows;
  // the per-thread sums are brought to the common maximum afterwards. The score matrix is far larger than any cache, so a read less
  // is time saved.
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
