// stainkit/src/io.cpp
//
// Host-side image I/O built on stb_image / stb_image_write. The implementation
// is intentionally kept inside a single TU so the macros defined by stb
// (which are *_IMPLEMENTATION) only expand once per binary.

#include "stainkit/io.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace stainkit {

namespace {

// Extensions we accept on input. We keep PNG + JPEG + BMP + TGA on the
// host side; TIFF and whole-slide formats go through openslide at the
// pipeline level (see include/stainkit/pipeline.h).
constexpr const char* kInputExtensions[] = {
    ".png",  ".jpg",  ".jpeg", ".bmp", ".tga", ".hdr",
};

constexpr const char* kOutputExtensions[] = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tga",
};

std::string ToLower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return s;
}

}  // namespace

bool IsSupportedImage(const std::filesystem::path& path) {
  const std::string ext = ToLower(path.extension().string());
  for (const auto* candidate : kInputExtensions) {
    if (ext == candidate) return true;
  }
  return false;
}

bool IsSupportedOutputImage(const std::filesystem::path& path) {
  const std::string ext = ToLower(path.extension().string());
  for (const auto* candidate : kOutputExtensions) {
    if (ext == candidate) return true;
  }
  return false;
}

Image ReadImage(const std::filesystem::path& path, int force_channels) {
  if (!std::filesystem::exists(path)) {
    std::ostringstream oss;
    oss << "ReadImage: file does not exist: " << path;
    throw std::runtime_error(oss.str());
  }

  int            width = 0;
  int            height = 0;
  int            channels_in_file = 0;
  stbi_uc*       raw = stbi_load(path.string().c_str(), &width, &height,
                                 &channels_in_file, force_channels);
  if (raw == nullptr) {
    std::ostringstream oss;
    oss << "ReadImage: stbi_load failed for " << path << " : "
        << stbi_failure_reason();
    throw std::runtime_error(oss.str());
  }

  const int channels = (force_channels == 0) ? channels_in_file
                                              : force_channels;
  if (channels != 3 && channels != 4) {
    stbi_image_free(raw);
    std::ostringstream oss;
    oss << "ReadImage: unsupported channel count " << channels
        << " for " << path;
    throw std::runtime_error(oss.str());
  }

  Image img;
  img.width  = static_cast<std::size_t>(width);
  img.height = static_cast<std::size_t>(height);
  img.layout = (channels == 4) ? PixelLayout::kRgba : PixelLayout::kRgb;
  img.stride = ComputeStride(img.width, channels);

  // stb gives us a tightly-packed buffer; expand into our row-padded one.
  const std::size_t tight_row = img.width * channels;
  img.pixels.assign(img.stride * img.height, 0);
  for (std::size_t y = 0; y < img.height; ++y) {
    const auto* src = raw + y * tight_row;
    auto*       dst = img.pixels.data() + y * img.stride;
    std::memcpy(dst, src, tight_row);
  }

  stbi_image_free(raw);
  return img;
}

namespace {

void WritePng(const Image& image, const std::filesystem::path& path) {
  const int stride_in_bytes = static_cast<int>(image.stride);
  const int ok = stbi_write_png(path.string().c_str(),
                                static_cast<int>(image.width),
                                static_cast<int>(image.height),
                                static_cast<int>(image.channels()),
                                image.pixels.data(), stride_in_bytes);
  if (ok == 0) {
    throw std::runtime_error("WriteImage: stbi_write_png failed for " +
                             path.string());
  }
}

void WriteJpg(const Image& image, const std::filesystem::path& path,
              int quality) {
  if (quality < 1) quality = 1;
  if (quality > 100) quality = 100;
  const int ok = stbi_write_jpg(path.string().c_str(),
                                static_cast<int>(image.width),
                                static_cast<int>(image.height),
                                static_cast<int>(image.channels()),
                                image.pixels.data(), quality);
  if (ok == 0) {
    throw std::runtime_error("WriteImage: stbi_write_jpg failed for " +
                             path.string());
  }
}

void WriteBmp(const Image& image, const std::filesystem::path& path) {
  const int ok = stbi_write_bmp(path.string().c_str(),
                                static_cast<int>(image.width),
                                static_cast<int>(image.height),
                                static_cast<int>(image.channels()),
                                image.pixels.data());
  if (ok == 0) {
    throw std::runtime_error("WriteImage: stbi_write_bmp failed for " +
                             path.string());
  }
}

void WriteTga(const Image& image, const std::filesystem::path& path) {
  const int ok = stbi_write_tga(path.string().c_str(),
                                static_cast<int>(image.width),
                                static_cast<int>(image.height),
                                static_cast<int>(image.channels()),
                                image.pixels.data());
  if (ok == 0) {
    throw std::runtime_error("WriteImage: stbi_write_tga failed for " +
                             path.string());
  }
}

}  // namespace

void WriteImage(const Image& image, const std::filesystem::path& path,
                int jpeg_quality) {
  if (image.empty()) {
    throw std::runtime_error("WriteImage: image is empty");
  }
  if (!IsSupportedOutputImage(path)) {
    throw std::runtime_error("WriteImage: unsupported output extension: " +
                             path.extension().string());
  }
  const std::string ext = ToLower(path.extension().string());
  if (ext == ".png") {
    WritePng(image, path);
  } else if (ext == ".jpg" || ext == ".jpeg") {
    WriteJpg(image, path, jpeg_quality);
  } else if (ext == ".bmp") {
    WriteBmp(image, path);
  } else if (ext == ".tga") {
    WriteTga(image, path);
  } else {
    throw std::runtime_error("WriteImage: no writer registered for " + ext);
  }
}

std::vector<std::filesystem::path> ListImagesIn(
    const std::filesystem::path& dir) {
  std::vector<std::filesystem::path> out;
  if (!std::filesystem::exists(dir) || !std::filesystem::is_directory(dir)) {
    return out;
  }
  for (const auto& entry : std::filesystem::directory_iterator(dir)) {
    if (!entry.is_regular_file()) continue;
    if (IsSupportedImage(entry.path())) {
      out.push_back(entry.path());
    }
  }
  std::sort(out.begin(), out.end());
  return out;
}

void WriteVisualisationPanel(const Image& input, const Image& normalised,
                             const Image& mask,
                             const std::filesystem::path& path) {
  if (input.empty() || normalised.empty() || mask.empty()) {
    throw std::runtime_error(
        "WriteVisualisationPanel: one of the inputs is empty");
  }
  if (input.width != normalised.width || input.height != normalised.height ||
      input.width != mask.width || input.height != mask.height) {
    throw std::runtime_error(
        "WriteVisualisationPanel: input/normalised/mask dimensions mismatch");
  }

  const std::size_t w = input.width;
  const std::size_t h = input.height;
  const std::size_t gap = std::max<std::size_t>(2, w / 200);
  const std::size_t panel_w = 3 * w + 2 * gap;
  const std::size_t panel_h = h;
  Image panel = MakeImage(panel_w, panel_h, PixelLayout::kRgb);

  auto blit = [&](const Image& src, std::size_t dx) {
    for (std::size_t y = 0; y < h; ++y) {
      const auto* src_row = src.pixels.data() + y * src.stride;
      auto*       dst_row = panel.pixels.data() + y * panel.stride + dx * 3;
      std::memcpy(dst_row, src_row, w * 3);
    }
  };
  blit(input, 0);
  blit(normalised, w + gap);
  // Convert mask to a 3-channel visualisation (white = tissue).
  Image mask_rgb = MakeImage(w, h, PixelLayout::kRgb);
  for (std::size_t y = 0; y < h; ++y) {
    for (std::size_t x = 0; x < w; ++x) {
      const byte v = mask.pixels[y * mask.stride + x];
      mask_rgb.pixels[y * mask_rgb.stride + 3 * x + 0] = v;
      mask_rgb.pixels[y * mask_rgb.stride + 3 * x + 1] = v;
      mask_rgb.pixels[y * mask_rgb.stride + 3 * x + 2] = v;
    }
  }
  blit(mask_rgb, 2 * (w + gap));

  WriteImage(panel, path);
}

}  // namespace stainkit
