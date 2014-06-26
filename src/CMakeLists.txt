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
  logger.cc
  product_model.cc
  cross_cat.cc
  assignments.cc
  cat_pipeline.cc
  hyper_kernel.cc
  kind_kernel.cc
  kind_proposer.cc
  kind_pipeline.cc
  schema.pb.cc
  ${DISTRIBUTIONS_INCLUDE_DIR}/distributions/io/schema.pb.cc
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

add_executable(shuffle shuffle.cc)
target_link_libraries(shuffle ${LOOM_LIBRARIES})

add_executable(infer infer.cc)
target_link_libraries(infer ${LOOM_LIBRARIES})

add_executable(posterior_enum posterior_enum.cc)
target_link_libraries(posterior_enum ${LOOM_LIBRARIES})

add_executable(generate generate.cc)
target_link_libraries(generate ${LOOM_LIBRARIES})

add_executable(query query.cc)
target_link_libraries(query ${LOOM_LIBRARIES})

install(TARGETS shuffle infer posterior_enum generate query
  RUNTIME DESTINATION bin
)
