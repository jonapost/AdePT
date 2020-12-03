# SPDX-FileCopyrightText: 2020 CERN
# SPDX-License-Identifier: Apache-2.0

# Noddy example 1 of particle processing with GPU
add_executable(field_step field_step.cu)
target_include_directories(field_step PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/base/inc>
  $<INSTALL_INTERFACE:base>
)
target_link_libraries(field_step PRIVATE CUDA::curand VecCore::VecCore CopCore::CopCore)

