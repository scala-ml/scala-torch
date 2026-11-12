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

// Attention without a score matrix in device memory (tensor cores, one warp per head and 16 query rows).
// q [heads, T, D], k [heads, S, D], v [heads, S, Dv], out [heads, T, Dv]; T and S are multiples of 16, 16 <= D, Dv <= 160.
// The keys are visited twice in tiles of 16: first for the row maximum, then the scores are computed again, turned into
// exponentials relative to that maximum and multiplied with the values; the division by the row sum comes last. Scores live in shared memory only; the keys and
// values of one head fit the L2 cache, so nothing of the size T x S is ever written to or read from device memory.
// A width that is not a multiple of 16 takes its last chunk from the last 16 columns; the columns that an earlier chunk
// already covered are zeroed on the query side (scores) or not written (output).
#define ATT_MAX_CHUNKS 10
__global__ void h_attention_fused(const __half* q, const __half* k, const __half* v, __half* out,
                                  unsigned heads, unsigned T, unsigned S, unsigned D, unsigned Dv, float scale) {
  using namespace nvcuda;
  __shared__ float scores[256];
  __shared__ __half tile[256];
  unsigned tilesPerHead = T / 16;
  unsigned head = blockIdx.x / tilesPerHead, row0 = (blockIdx.x % tilesPerHead) * 16;
  if (head >= heads) return;
  unsigned lane = threadIdx.x, row = lane & 15, col0 = (lane >> 4) * 8;
  const __half* qh = q + ((unsigned long long)head * T + row0) * D;
  const __half* kh = k + (unsigned long long)head * S * D;
  const __half* vh = v + (unsigned long long)head * S * Dv;
  unsigned chunks = (D + 15) / 16, tail = D % 16;
  unsigned chunksV = (Dv + 15) / 16, tailV = Dv % 16;

  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> qa[ATT_MAX_CHUNKS];
  for (unsigned c = 0; c < chunks; ++c) {
    if (c == chunks - 1 && tail != 0) {
      // last 16 columns, with the columns of the previous chunk set to zero
      for (unsigned j = 0; j < 8; ++j) {
        unsigned cc = col0 + j;
        tile[row * 16 + cc] = cc < 16 - tail ? __float2half_rn(0.f) : qh[row * D + (D - 16) + cc];
      }
      __syncwarp();
      wmma::load_matrix_sync(qa[c], tile, 16);
      __syncwarp();
    } else wmma::load_matrix_sync(qa[c], qh + c * 16, D);
  }

  wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> kb;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;

  // pass 1: running maximum and sum of exponentials for this lane's half of its row
  float m = -3.402823466e+38f;
  for (unsigned key0 = 0; key0 < S; key0 += 16) {
    wmma::fill_fragment(acc, 0.f);
    for (unsigned c = 0; c < chunks; ++c) {
      unsigned d0 = (c == chunks - 1 && tail != 0) ? D - 16 : c * 16;
      wmma::load_matrix_sync(kb, kh + (unsigned long long)key0 * D + d0, D);
      wmma::mma_sync(acc, qa[c], kb, acc);
    }
    wmma::store_matrix_sync(scores, acc, 16, wmma::mem_row_major);
    __syncwarp();
    for (unsigned j = 0; j < 8; ++j) m = fmaxf(m, scores[row * 16 + col0 + j]);
    __syncwarp();
  }
  float rowMax = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 16)) * scale;
  float sum = 0.f;

  // pass 2: probabilities times values
  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> pa;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> vb;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> result[ATT_MAX_CHUNKS];
  for (unsigned c = 0; c < chunksV; ++c) wmma::fill_fragment(result[c], 0.f);
  for (unsigned key0 = 0; key0 < S; key0 += 16) {
    wmma::fill_fragment(acc, 0.f);
    for (unsigned c = 0; c < chunks; ++c) {
      unsigned d0 = (c == chunks - 1 && tail != 0) ? D - 16 : c * 16;
      wmma::load_matrix_sync(kb, kh + (unsigned long long)key0 * D + d0, D);
      wmma::mma_sync(acc, qa[c], kb, acc);
    }
    wmma::store_matrix_sync(scores, acc, 16, wmma::mem_row_major);
    __syncwarp();
    for (unsigned j = 0; j < 8; ++j) {
      unsigned at = row * 16 + col0 + j;
      __half e = __float2half_rn(__expf(scores[at] * scale - rowMax));
      tile[at] = e;
      sum += __half2float(e);
    }
    __syncwarp();
    wmma::load_matrix_sync(pa, tile, 16);
    for (unsigned c = 0; c < chunksV; ++c) {
      unsigned e0 = (c == chunksV - 1 && tailV != 0) ? Dv - 16 : c * 16;
      wmma::load_matrix_sync(vb, vh + (unsigned long long)key0 * Dv + e0, Dv);
      wmma::mma_sync(result[c], pa, vb, result[c]);
    }
    __syncwarp();
  }

  float inv = 1.f / (sum + __shfl_xor_sync(0xffffffffu, sum, 16));
  __half* oh = out + ((unsigned long long)head * T + row0) * Dv;
  for (unsigned c = 0; c < chunksV; ++c) {
    bool last = c == chunksV - 1 && tailV != 0;
    unsigned e0 = last ? Dv - 16 : c * 16, skip = last ? 16 - tailV : 0;
    wmma::store_matrix_sync(scores, result[c], 16, wmma::mem_row_major);
    __syncwarp();
    for (unsigned j = 0; j < 8; ++j) {
      unsigned cc = col0 + j;
      if (cc >= skip) oh[row * Dv + e0 + cc] = __float2half_rn(scores[row * 16 + cc] * inv);
    }
    __syncwarp();
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
