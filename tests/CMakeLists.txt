enable_testing()

# C++ unit tests use GoogleTest if available, else a header-only fallback.
find_package(GTest QUIET)

if(GTest_FOUND)
  add_subdirectory(unit)
  add_subdirectory(integration)
else()
  message(STATUS "GTest not found; skipping C++ tests")
endif()

# Python tests are always registered (pytest is discovered at runtime)
add_test(NAME python_tests
  COMMAND ${Python3_EXECUTABLE} -m pytest -xvs ${CMAKE_CURRENT_SOURCE_DIR}/python
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
