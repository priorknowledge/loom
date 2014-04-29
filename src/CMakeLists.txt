
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
  product_model.cc
  cross_cat.cc
  algorithm8.cc
  assignments.cc
  schema.pb.cc
  ${DISTRIBUTIONS_PATH}/src/io/schema.pb.cc
)

set(LOOM_LIBRARIES
  loom
  distributions_shared
  protobuf
  pthread
)

# unset dynamic flags
set(CMAKE_EXE_LINK_DYNAMIC_C_FLAGS)
set(CMAKE_EXE_LINK_DYNAMIC_CXX_FLAGS)

add_executable(infer infer.cc)
target_link_libraries(infer ${LOOM_LIBRARIES})

add_executable(posterior_enum posterior_enum.cc)
target_link_libraries(posterior_enum ${LOOM_LIBRARIES})

add_executable(predict predict.cc)
target_link_libraries(predict ${LOOM_LIBRARIES})
