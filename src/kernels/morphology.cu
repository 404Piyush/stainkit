// stainkit/src/kernels/morphology.cu
//
// Binary morphology: erosion, dilation, opening and closing with a flat
// disc structuring element. We use a separable approach: a 1D horizontal
// pass followed by a 1D vertical pass, with shared memory used as a tile
// for the disc SE.

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>

#include "stainkit/kernels/morphology.h"

namespace stainkit {
namespace kernels {
namespace {

// Compute the half-width of the disc for a given pixel distance.
// We pre-bake a 1D mask of (2*radius+1) entries: 1 if the cell is inside
// the disc, 0 otherwise.
__device__ inline bool InsideDisc(int dx, int dy, int radius) {
  return (dx * dx + dy * dy) <= (radius * radius);
}

// Separable 1D pass along the row axis.
__global__ void MorphRowKernel(const std::uint8_t* __restrict__ d_in,
                               std::uint8_t* __restrict__ d_out, std::size_t width,
                               std::size_t height, int radius, bool is_max) {
  extern __shared__ std::uint8_t s_tile[];
  const int tile_w = blockDim.x + 2 * 64;  // assume radius <= 64
  const int x_block_start = blockIdx.x * blockDim.x;
  const int y             = blockIdx.y * blockDim.y + threadIdx.y;
  if (y >= static_cast<int>(height)) return;

  // Cooperatively load the tile (with halo) into shared memory.
  for (int i = threadIdx.x; i < tile_w; i += blockDim.x) {
    const int gx = x_block_start - 64 + i;
    const int sx = gx;
    if (gx >= 0 && gx < static_cast<int>(width)) {
      s_tile[i] = d_in[y * width + gx];
    } else {
      s_tile[i] = 0;
    }
  }
  __syncthreads();
  if (x_block_start + threadIdx.x >= static_cast<int>(width)) return;
  const int x = x_block_start + threadIdx.x;
  std::uint8_t acc = is_max ? 0 : 255;
  for (int dx = -radius; dx <= radius; ++dx) {
    if (!InsideDisc(dx, 0, radius)) continue;
    const int sx = threadIdx.x + 64 + dx;
    const std::uint8_t v = s_tile[sx];
    acc = is_max ? max(acc, v) : min(acc, v);
  }
  d_out[y * width + x] = acc;
}

// Separable 1D pass along the column axis.
__global__ void MorphColKernel(const std::uint8_t* __restrict__ d_in,
                               std::uint8_t* __restrict__ d_out, std::size_t width,
                               std::size_t height, int radius, bool is_max) {
  extern __shared__ std::uint8_t s_tile[];
  const int tile_h = blockDim.y + 2 * 64;
  const int x               = blockIdx.x * blockDim.x + threadIdx.x;
  const int y_block_start  = blockIdx.y * blockDim.y;
  if (x >= static_cast<int>(width)) return;

  for (int i = threadIdx.y; i < tile_h; i += blockDim.y) {
    const int gy = y_block_start - 64 + i;
    if (gy >= 0 && gy < static_cast<int>(height)) {
      s_tile[i] = d_in[gy * width + x];
    } else {
      s_tile[i] = 0;
    }
  }
  __syncthreads();
  if (y_block_start + threadIdx.y >= static_cast<int>(height)) return;
  const int y = y_block_start + threadIdx.y;
  std::uint8_t acc = is_max ? 0 : 255;
  for (int dy = -radius; dy <= radius; ++dy) {
    if (!InsideDisc(0, dy, radius)) continue;
    const int sy = threadIdx.y + 64 + dy;
    const std::uint8_t v = s_tile[sy];
    acc = is_max ? max(acc, v) : min(acc, v);
  }
  d_out[y * width + x] = acc;
}

inline cudaStream_t AsStream(void* s) {
  return (s == nullptr) ? 0 : *reinterpret_cast<cudaStream_t*>(&s);
}

void DispatchRow(const std::uint8_t* d_in, std::uint8_t* d_out, std::size_t w,
                 std::size_t h, int radius, bool is_max, void* stream) {
  if (radius <= 0) {
    cudaMemcpyAsync(d_out, d_in, w * h * sizeof(std::uint8_t),
                    cudaMemcpyDeviceToDevice, AsStream(stream));
    return;
  }
  if (radius > 64) {
    throw std::invalid_argument("DispatchRow: radius > 64 not supported");
  }
  dim3 block(128, 4);
  dim3 grid(static_cast<unsigned int>((w + block.x - 1) / block.x),
            static_cast<unsigned int>((h + block.y - 1) / block.y));
  size_t smem = (block.x + 128) * sizeof(std::uint8_t);
  MorphRowKernel<<<grid, block, smem, AsStream(stream)>>>(d_in, d_out, w, h,
                                                          radius, is_max);
}

void DispatchCol(const std::uint8_t* d_in, std::uint8_t* d_out, std::size_t w,
                 std::size_t h, int radius, bool is_max, void* stream) {
  if (radius <= 0) {
    cudaMemcpyAsync(d_out, d_in, w * h * sizeof(std::uint8_t),
                    cudaMemcpyDeviceToDevice, AsStream(stream));
    return;
  }
  if (radius > 64) {
    throw std::invalid_argument("DispatchCol: radius > 64 not supported");
  }
  dim3 block(32, 8);
  dim3 grid(static_cast<unsigned int>((w + block.x - 1) / block.x),
            static_cast<unsigned int>((h + block.y - 1) / block.y));
  size_t smem = (block.y + 128) * sizeof(std::uint8_t);
  MorphColKernel<<<grid, block, smem, AsStream(stream)>>>(d_in, d_out, w, h,
                                                          radius, is_max);
}

}  // namespace

void Erode(std::uint8_t* d_io, std::size_t width, std::size_t height,
           std::size_t radius, void* stream) {
  if (d_io == nullptr) return;
  // Erosion = min over SE. We ping-pong through a scratch buffer.
  std::uint8_t* d_scratch = nullptr;
  cudaMalloc(&d_scratch, width * height * sizeof(std::uint8_t));
  DispatchCol(d_io, d_scratch, width, height, static_cast<int>(radius), false,
              stream);
  DispatchRow(d_scratch, d_io, width, height, static_cast<int>(radius), false,
              stream);
  cudaFree(d_scratch);
}

void Dilate(std::uint8_t* d_io, std::size_t width, std::size_t height,
            std::size_t radius, void* stream) {
  if (d_io == nullptr) return;
  std::uint8_t* d_scratch = nullptr;
  cudaMalloc(&d_scratch, width * height * sizeof(std::uint8_t));
  DispatchCol(d_io, d_scratch, width, height, static_cast<int>(radius), true,
              stream);
  DispatchRow(d_scratch, d_io, width, height, static_cast<int>(radius), true,
              stream);
  cudaFree(d_scratch);
}

void Open(std::uint8_t* d_io, std::size_t width, std::size_t height,
          std::size_t radius, void* stream) {
  if (radius == 0) return;
  Erode(d_io, width, height, radius, stream);
  Dilate(d_io, width, height, radius, stream);
}

void Close(std::uint8_t* d_io, std::size_t width, std::size_t height,
           std::size_t radius, void* stream) {
  if (radius == 0) return;
  Dilate(d_io, width, height, radius, stream);
  Erode(d_io, width, height, radius, stream);
}

}  // namespace kernels
}  // namespace stainkit
