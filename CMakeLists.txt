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
  src/analysis/KeyTracker.cpp
  src/analysis/ChordDetector.cpp
  src/analysis/Analyzer.cpp
  src/analysis/AudioFile.cpp
)
set_target_properties(skale_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
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

# Plugin (VST3 y aplicación independiente). JUCE se descarga al configurar; no
# está en el repo. OJO: JUCE es AGPLv3 o licencia comercial; para distribuir el
# plugin con la licencia propietaria de Skale hace falta la licencia comercial.
option(SKALE_BUILD_PLUGIN "Construir el plugin JUCE (descarga JUCE)" OFF)
if(SKALE_BUILD_PLUGIN)
  enable_language(C)
  include(FetchContent)
  FetchContent_Declare(JUCE
    GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
    GIT_TAG 8.0.15
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(JUCE)

  juce_add_plugin(SkalePlugin
    PRODUCT_NAME "Skale"
    BUNDLE_ID com.pabloolivares.skale
    COMPANY_NAME "Pablo Olivares"
    PLUGIN_MANUFACTURER_CODE Pabo
    PLUGIN_CODE Skal
    FORMATS VST3 Standalone
    IS_SYNTH FALSE
    NEEDS_MIDI_INPUT FALSE
    NEEDS_MIDI_OUTPUT FALSE
    COPY_PLUGIN_AFTER_BUILD FALSE)
  target_sources(SkalePlugin PRIVATE
    src/plugin/PluginProcessor.cpp
    src/plugin/PluginEditor.cpp)
  target_compile_definitions(SkalePlugin PUBLIC
    JUCE_WEB_BROWSER=0 JUCE_USE_CURL=0 JUCE_VST3_CAN_REPLACE_VST2=0 JUCE_USE_MP3AUDIOFORMAT=1)
  target_link_libraries(SkalePlugin PRIVATE
    skale_core
    juce::juce_audio_utils
    juce::juce_audio_formats
    juce::juce_recommended_config_flags)

  juce_add_console_app(SkalePluginSelfTest PRODUCT_NAME "SkalePluginSelfTest")
  target_sources(SkalePluginSelfTest PRIVATE
    tools/plugin_selftest.cpp
    src/plugin/PluginProcessor.cpp
    src/plugin/PluginEditor.cpp)
  target_compile_definitions(SkalePluginSelfTest PRIVATE
    JUCE_WEB_BROWSER=0 JUCE_USE_CURL=0 JUCE_STANDALONE_APPLICATION=1 JUCE_USE_MP3AUDIOFORMAT=1)
  target_link_libraries(SkalePluginSelfTest PRIVATE
    skale_core juce::juce_audio_utils juce::juce_audio_formats juce::juce_recommended_config_flags)
endif()
