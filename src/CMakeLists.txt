include_directories(${CMAKE_SOURCE_DIR})

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_LIST_DIR}/schema.pb.cc
  COMMAND protoc ${CMAKE_CURRENT_LIST_DIR}/schema.proto
    --proto_path=${CMAKE_CURRENT_LIST_DIR}
    --proto_path=${DISTRIBUTIONS_PATH}
    --cpp_out=${CMAKE_CURRENT_LIST_DIR}
    --python_out=${CMAKE_CURRENT_LIST_DIR}
  DEPENDS ${CMAKE_CURRENT_LIST_DIR}/schema.proto
)

add_library(loom
  loom.cc
  logger.cc
  product_model.cc
  cross_cat.cc
  assignments.cc
  cat_kernel.cc
  hyper_kernel.cc
  kind_kernel.cc
  kind_proposer.cc
  schema.pb.cc
  ${DISTRIBUTIONS_PATH}/src/io/schema.pb.cc
)

set(LOOM_LIBRARIES
  loom
  distributions_shared
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

add_executable(predict predict.cc)
target_link_libraries(predict ${LOOM_LIBRARIES})
