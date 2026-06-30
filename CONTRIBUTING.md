# Contributing to stainkit

Thanks for taking the time to contribute. The project follows a
small set of conventions so that the codebase stays uniform and easy
to navigate.

## Workflow

1. Fork the repo, create a feature branch.
2. Make your change.
3. Run `./install.sh` and `./run.sh` end-to-end to confirm nothing is
   broken.
4. Add a test in `tests/` if your change introduces new behaviour or
   fixes a bug.
5. Run `clang-format` (Google C++ style, see `.clang-format`).
6. Open a pull request; the CI will run the build, the unit tests
   and the formatter check.

## Style

* The codebase is `clang-formatted` against the Google C++ style.
  Run `clang-format -i <file>` before committing.
* Headers live in `include/stainkit/`. Implementations live in
  `src/` and `src/kernels/`.
* Public APIs follow the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html).
* Avoid raw pointers in public APIs; use `std::unique_ptr`,
  `std::shared_ptr` or references.
* CUDA kernels must be device-callable only. Host-side launchers
  live next to the kernel in the same `.cu` file.
* Tests use GoogleTest. Run them with `ctest`.

## Adding a new kernel

1. Declare the host-side launcher in a header under
   `include/stainkit/kernels/`.
2. Implement the device kernel in a matching `src/kernels/*.cu` file.
3. Add a unit test in `tests/`.
4. Add the source file to `CMakeLists.txt` (`STK_KERNEL_SOURCES`).
5. Update `docs/algorithms.md` if the math changed.

## Commit messages

We use the [Conventional Commits](https://www.conventionalcommits.org/)
style:

```
feat: add NPP-backed resize kernel
fix: handle empty image in WriteVisualisationPanel
docs: add macenko math to docs/algorithms.md
test: add multi-stream test for RunBatch
```

## License

By contributing you agree that your work will be released under the
[Apache 2.0 License](LICENSE).
