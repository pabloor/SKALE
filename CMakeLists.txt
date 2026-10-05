cmake_minimum_required(VERSION 3.20)
project(skale VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(SKALE_BUILD_TESTS "Construir los tests" ON)

add_library(skale_core STATIC
  src/analysis/Fft.cpp
  src/analysis/Chromagram.cpp
  src/analysis/Theory.cpp
  src/analysis/KeyDetector.cpp
  src/analysis/ChordDetector.cpp
  src/analysis/Analyzer.cpp
  src/analysis/AudioFile.cpp
)
target_include_directories(skale_core PUBLIC src)
target_include_directories(skale_core SYSTEM PRIVATE third_party/dr_libs)
if(NOT MSVC)
  target_compile_options(skale_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(skale-cli src/cli/main.cpp)
target_link_libraries(skale-cli PRIVATE skale_core)
if(NOT MSVC)
  target_compile_options(skale-cli PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(SKALE_BUILD_TESTS)
  enable_testing()
  add_executable(skale_tests tests/test_main.cpp)
  target_link_libraries(skale_tests PRIVATE skale_core)
  if(NOT MSVC)
    target_compile_options(skale_tests PRIVATE -Wall -Wextra -Wpedantic)
  endif()
  add_test(NAME skale_tests COMMAND skale_tests)
endif()
