// stainkit/tests/test_io.cpp
//
// Image I/O round-trip tests. We write a small synthetic image, read it
// back and compare pixel values. The tests are skipped if stb is unable
// to decode the file (should not happen with PNG).

#include <gtest/gtest.h>

#include <cstdio>
#include <filesystem>
#include <random>
#include <string>

#include "stainkit/io.h"
#include "stainkit/types.h"

namespace fs = std::filesystem;

namespace {

fs::path MakeTemp(const std::string& name) {
  auto dir = fs::temp_directory_path() / "stainkit-tests";
  fs::create_directories(dir);
  return dir / name;
}

}  // namespace

TEST(Io, SupportedExtensions) {
  EXPECT_TRUE(stainkit::IsSupportedImage("foo.png"));
  EXPECT_TRUE(stainkit::IsSupportedImage("foo.PNG"));
  EXPECT_TRUE(stainkit::IsSupportedImage("foo.jpg"));
  EXPECT_FALSE(stainkit::IsSupportedImage("foo.svs"));
  EXPECT_TRUE(stainkit::IsSupportedOutputImage("foo.png"));
  EXPECT_FALSE(stainkit::IsSupportedOutputImage("foo.tiff"));
}

TEST(Io, RoundTripPng) {
  auto img = stainkit::MakeImage(33, 17, stainkit::PixelLayout::kRgb);
  // Fill with a recognisable pattern.
  for (std::size_t y = 0; y < img.height; ++y) {
    for (std::size_t x = 0; x < img.width; ++x) {
      img.pixels[y * img.stride + 3 * x + 0] = static_cast<std::uint8_t>(x);
      img.pixels[y * img.stride + 3 * x + 1] = static_cast<std::uint8_t>(y);
      img.pixels[y * img.stride + 3 * x + 2] = 64;
    }
  }
  const auto path = MakeTemp("roundtrip.png");
  ASSERT_NO_THROW(stainkit::WriteImage(img, path));
  auto decoded = stainkit::ReadImage(path, 3);
  EXPECT_EQ(decoded.width, img.width);
  EXPECT_EQ(decoded.height, img.height);
  for (std::size_t y = 0; y < img.height; ++y) {
    for (std::size_t x = 0; x < img.width; ++x) {
      EXPECT_NEAR(decoded.pixels[y * decoded.stride + 3 * x + 0],
                  img.pixels[y * img.stride + 3 * x + 0], 2);
      EXPECT_NEAR(decoded.pixels[y * decoded.stride + 3 * x + 1],
                  img.pixels[y * img.stride + 3 * x + 1], 2);
      EXPECT_NEAR(decoded.pixels[y * decoded.stride + 3 * x + 2],
                  img.pixels[y * img.stride + 3 * x + 2], 2);
    }
  }
}

TEST(Io, ReadMissingFileThrows) {
  EXPECT_THROW(
      stainkit::ReadImage(fs::path{"/this/path/does/not/exist.png"}, 3),
      std::runtime_error);
}

TEST(Io, WriteEmptyThrows) {
  stainkit::Image img;
  EXPECT_THROW(stainkit::WriteImage(img, MakeTemp("empty.png")),
               std::runtime_error);
}
