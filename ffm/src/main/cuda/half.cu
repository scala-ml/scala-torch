// Float16 activation kernels for the CUDA diffusion runtime. Storage is __half; arithmetic is Float32.
// Compiled to ffm/src/main/resources/scalatorch/half.ptx by build_ptx.py.
#include <cuda_fp16.h>
#include <mma.h>

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

// out [rows, width] = bias [width] in every row; the matrix product is then accumulated on top (see linearWithBias).
__global__ void h_fill_rows(const __half* bias, __half* out, unsigned n, unsigned width) {
  GRID_INDEX(i, n) out[i] = bias[i % width];
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

// In-place softmax over the last axis of [rows, width], one block per row.
__global__ void h_softmax(__half* x, unsigned rows, unsigned width) {
  __shared__ float shared[32];
  unsigned row = blockIdx.x;
  if (row >= rows) return;
  __half* base = x + (unsigned long long)row * width;
  unsigned per = width / blockDim.x;
  if (width % blockDim.x == 0 && per >= 8 && per <= 16) { // shorter rows measured slower this way
    // The score matrix is far larger than any cache, so every pass over it costs device-memory bandwidth. Each thread reads its
    // share of the row once and keeps it in registers (the loops have a constant bound for that): one read and one write of the
    // row, and one exponential per element.
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
  // Other widths: one read for the maximum and the sum together (running maximum per thread, sums brought to the common maximum).
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

// Attention without a score matrix in device memory (tensor cores, one pass over the keys, everything in registers).
// q [heads, T, D], k [heads, S, D], v [heads, S, Dv], out [heads, T, Dv]; S is a multiple of 16, T of 16 * R, 16 <= D, Dv.
// A warp takes R tiles of 16 query rows of one head and visits the keys once, in tiles of 16. For every key tile the key and
// value fragments are loaded once and used for all R query tiles: scores = Q K^T (tensor cores), then an online softmax (the
// running row maximum is raised, the sums and results gathered so far are scaled down by exp(old - new), the scores become
// exponentials relative to the maximum) and the product with the values (tensor cores). The division by the row sum comes last.
// Reading the keys and values is what limits this kernel, so R query tiles per read cut its cost by R.
//
// This relies on where the 16x16 fragments keep their elements (measured by h_wmma_layout and checked by the caller before this
// kernel is used): accumulator and matrix_a use the same positions; lane L holds, with g = L / 4 and t = L % 4, row g in slots
// 0, 1, 4, 5 and row g + 8 in slots 2, 3, 6, 7 (columns 2t, 2t+1, 2t+8, 2t+9).
// A width that is not a multiple of 16 takes its last chunk from the last 16 columns; the columns that an earlier chunk already
// covered are zeroed on the query side (scores) or not written (output).
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
                                         unsigned H, float* scores, __half* tile) {
  using namespace nvcuda;
  // H == 0: heads major, q [heads, T, D] (and so on). H > 0: tokens major, q [heads / H, T, H, D], as the projections of a
  // transformer block leave them; the fragments are then loaded with a row stride of H * D and no transposed copy is needed.
  unsigned groupsPerHead = T / (16 * R);
  unsigned head = blockIdx.x / groupsPerHead, row0 = (blockIdx.x % groupsPerHead) * 16 * R;
  if (head >= heads) return;
  unsigned lane = threadIdx.x & 31, row = lane & 15, col0 = (lane >> 4) * 8;
  unsigned strideQ = H ? H * D : D, strideV = H ? H * Dv : Dv;
  unsigned long long member = H ? head / H : 0, within = H ? head % H : 0;
  const __half* kh = H ? k + (member * S * H + within) * D : k + (unsigned long long)head * S * D;
  const __half* vh = H ? v + (member * S * H + within) * Dv : v + (unsigned long long)head * S * Dv;
  const __half* qBase = H ? q + (member * T * H + within) * D : q + (unsigned long long)head * T * D;
  __half* outBase = H ? out + (member * T * H + within) * Dv : out + (unsigned long long)head * T * Dv;
  // D and Dv both have exactly C chunks of 16 (the caller picks the kernel). Every loop over chunks and query tiles has a
  // constant bound and is unrolled, so the fragments stay in registers; with run-time bounds they were kept in local memory
  // and every variant took the same 6.5 ms whatever it did.
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
  // "low" = the row of slots 0, 1, 4, 5; "high" = the row of slots 2, 3, 6, 7
  float maxLow[R], maxHigh[R], sumLow[R], sumHigh[R];
  #pragma unroll
  for (int r = 0; r < R; ++r) {
    #pragma unroll
    for (int c = 0; c < C; ++c) wmma::fill_fragment(result[r][c], 0.f);
    maxLow[r] = -3.402823466e+38f; maxHigh[r] = -3.402823466e+38f; sumLow[r] = 0.f; sumHigh[r] = 0.f;
  }

  for (unsigned key0 = 0; key0 < S; key0 += 16) {
    #pragma unroll
    for (int c = 0; c < C; ++c) {
      unsigned d0 = (c == C - 1 && tail != 0) ? D - 16 : c * 16;
      wmma::load_matrix_sync(kb[c], kh + (unsigned long long)key0 * strideQ + d0, strideQ);
    }
    #pragma unroll
    for (int c = 0; c < C; ++c) {
      unsigned e0v = (c == C - 1 && tailV != 0) ? Dv - 16 : c * 16;
      wmma::load_matrix_sync(vb[c], vh + (unsigned long long)key0 * strideV + e0v, strideV);
    }
    #pragma unroll
    for (int r = 0; r < R; ++r) {
      wmma::fill_fragment(acc, 0.f);
      #pragma unroll
      for (int c = 0; c < C; ++c) wmma::mma_sync(acc, qa[r][c], kb[c], acc);
      float s0 = acc.x[0] * scale, s1 = acc.x[1] * scale, s2 = acc.x[2] * scale, s3 = acc.x[3] * scale;
      float s4 = acc.x[4] * scale, s5 = acc.x[5] * scale, s6 = acc.x[6] * scale, s7 = acc.x[7] * scale;
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
    __half* oh = outBase + (unsigned long long)(row0 + r * 16) * strideV;
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
        if (cc >= skip) oh[row * strideV + e0v + cc] = __float2half_rn(scores[row * 16 + cc]);
      }
      __syncwarp();
    }
  }
}
extern "C" {
// One warp per block. Variants: query tiles per warp x largest number of 16-wide chunks (width up to 16 * chunks).
#define ATT_KERNEL(NAME, R, C) \
  __global__ void NAME(const __half* q, const __half* k, const __half* v, __half* out, \
                       unsigned heads, unsigned T, unsigned S, unsigned D, unsigned Dv, float scale, unsigned H) { \
    __shared__ float scores[256]; \
    __shared__ __half tile[256]; \
    att_body<R, C>(q, k, v, out, heads, T, S, D, Dv, scale, H, scores, tile); \
  }
// name: r<query tiles per warp>c<chunks>
ATT_KERNEL(h_attention_fused_r1c3, 1, 3)
ATT_KERNEL(h_attention_fused_r2c3, 2, 3)
ATT_KERNEL(h_attention_fused_r4c3, 4, 3)
ATT_KERNEL(h_attention_fused_r1c4, 1, 4)
ATT_KERNEL(h_attention_fused_r2c4, 2, 4)
ATT_KERNEL(h_attention_fused_r4c4, 4, 4)
ATT_KERNEL(h_attention_fused_r1c5, 1, 5)
ATT_KERNEL(h_attention_fused_r2c5, 2, 5)

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
