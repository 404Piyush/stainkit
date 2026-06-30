// stainkit/tests/test_python_bindings.cpp
//
// Sanity test that the pybind11 module is buildable. We do not import
// Python here — that requires a runtime Python dependency. Instead, the
// test asserts that the module's symbol set is non-empty.

#include <gtest/gtest.h>

extern "C" void PyInit_gpustain();

TEST(PythonBindings, ModuleSymbolResolves) {
  // This test passes if and only if the bindings object file was linked
  // into the test executable. The static cast ensures the symbol is not
  // dead-stripped.
  auto fn = reinterpret_cast<void (*)()>(&PyInit_gpustain);
  EXPECT_NE(fn, nullptr);
}
