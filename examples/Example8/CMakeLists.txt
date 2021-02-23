# SPDX-FileCopyrightText: 2021 CERN
# SPDX-License-Identifier: Apache-2.0

if(NOT TARGET g4HepEm)
  message(STATUS "Disabling example8 (needs G4HepEm)")
  return()
endif()

# Example 8 of particle transportation with GPUs: Tracks are stored with a
# monotonic slot number, slots are not reused. The example includes geometry,
# magnetic field with constant Bz, and physics processes for e-/e+. Photons
# immediately pair-produce if allowed or deposit their energy. Results are
# reproducible using one RANLUX++ state per track.
add_executable(example8 example8.cpp example8.cu)
target_include_directories(example8 PUBLIC
  $<TARGET_PROPERTY:g4HepEmRun,INCLUDE_DIRECTORIES>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/base/inc>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/physics/processes/inc>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/tracking/inc>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/magneticfield/inc>
  $<INSTALL_INTERFACE:base>
)
target_compile_definitions(example8 PRIVATE G4HepEm_CUDA_BUILD)
target_link_libraries(example8 PRIVATE VecCore::VecCore VecGeom::vecgeom VecGeom::vecgeomcuda_static VecGeom::vgdml ${Geant4_LIBRARIES} g4HepEmData g4HepEmInit CopCore::CopCore)
target_compile_options(example8 PRIVATE "$<$<AND:$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>")
set_target_properties(example8 PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
