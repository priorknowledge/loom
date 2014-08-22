include_directories(${CMAKE_SOURCE_DIR})
include_directories(${DISTRIBUTIONS_INCLUDE_DIR})

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_LIST_DIR}/schema.pb.cc
  COMMAND protoc ${CMAKE_CURRENT_LIST_DIR}/schema.proto
    --proto_path=${CMAKE_CURRENT_LIST_DIR}
    --proto_path=${DISTRIBUTIONS_INCLUDE_DIR}
    --cpp_out=${CMAKE_CURRENT_LIST_DIR}
    --python_out=${CMAKE_CURRENT_LIST_DIR}/../loom
  DEPENDS ${CMAKE_CURRENT_LIST_DIR}/schema.proto
)

set_source_files_properties(schema.pb.cc
  PROPERTIES
  COMPILE_FLAGS "-Wno-unused -Wno-unused-parameter"
)

add_library(loom
  loom.cc
  multi_loom.cc
  logger.cc
  product_value.cc
  product_model.cc
  product_mixture.cc
  cross_cat.cc
  assignments.cc
  cat_pipeline.cc
  hyper_kernel.cc
  kind_kernel.cc
  kind_proposer.cc
  kind_pipeline.cc
  query_server.cc
  differ.cc
  schema.pb.cc
  #${DISTRIBUTIONS_INCLUDE_DIR}/distributions/io/schema.pb.cc
)

set(LOOM_LIBRARIES
  loom
  ${DISTRIBUTIONS_LIBRARIES}
  protobuf
  pthread
  tcmalloc
)

# unset dynamic flags
set(CMAKE_EXE_LINK_DYNAMIC_C_FLAGS)
set(CMAKE_EXE_LINK_DYNAMIC_CXX_FLAGS)

add_executable(loom_tare tare.cc)
target_link_libraries(loom_tare ${LOOM_LIBRARIES})

add_executable(loom_sparsify sparsify.cc)
target_link_libraries(loom_sparsify ${LOOM_LIBRARIES})

add_executable(loom_shuffle shuffle.cc)
target_link_libraries(loom_shuffle ${LOOM_LIBRARIES})

add_executable(loom_infer infer.cc)
target_link_libraries(loom_infer ${LOOM_LIBRARIES})

add_executable(loom_posterior_enum posterior_enum.cc)
target_link_libraries(loom_posterior_enum ${LOOM_LIBRARIES})

add_executable(loom_generate generate.cc)
target_link_libraries(loom_generate ${LOOM_LIBRARIES})

add_executable(loom_mix mix.cc)
target_link_libraries(loom_mix ${LOOM_LIBRARIES})

add_executable(loom_query query.cc)
target_link_libraries(loom_query ${LOOM_LIBRARIES})

install(TARGETS
  loom_tare
  loom_sparsify
  loom_shuffle
  loom_infer
  loom_posterior_enum
  loom_generate
  loom_mix
  loom_query
  RUNTIME DESTINATION bin
)
