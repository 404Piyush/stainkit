# stainkit/tests/CMakeLists.txt
# ---------------------------------------------------------------------------
# Test suite. The `add_subdirectory` in the root CMakeLists.txt includes
# this file. Each test executable is registered with CTest.

add_executable(test_types           test_types.cpp)
add_executable(test_io              test_io.cpp)
add_executable(test_cpu_reference   test_cpu_reference.cpp)
add_executable(test_kernels         test_kernels.cu)
add_executable(test_pipeline        test_pipeline.cpp)

target_link_libraries(test_types          PRIVATE stainkit::core)
target_link_libraries(test_io             PRIVATE stainkit::core)
target_link_libraries(test_cpu_reference  PRIVATE stainkit::core)
target_link_libraries(test_kernels        PRIVATE stainkit::core stainkit::cuda)
target_link_libraries(test_pipeline       PRIVATE stainkit::core stainkit::cuda)

if(STK_BUILD_PYTHON)
    add_executable(test_python_bindings test_python_bindings.cpp)
    target_link_libraries(test_python_bindings PRIVATE stainkit::core)
endif()

foreach(t test_types test_io test_cpu_reference test_kernels test_pipeline)
    add_test(NAME ${t} COMMAND ${t})
    set_tests_properties(${t} PROPERTIES TIMEOUT 120)
endforeach()

if(STK_BUILD_PYTHON)
    add_test(NAME test_python_bindings COMMAND test_python_bindings)
    set_tests_properties(test_python_bindings PROPERTIES TIMEOUT 60)
endif()
