
add_executable(loom
  loom.cc
  product_model.cc
  cross_cat.cc
  schema.pb.cc
)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_LIST_DIR}/schema.pb.cc
  COMMAND protoc ${CMAKE_CURRENT_LIST_DIR}/schema.proto
    --proto_path=${CMAKE_CURRENT_LIST_DIR}
    --proto_path=${DISTRIBUTIONS_PATH}
    --cpp_out=${CMAKE_CURRENT_LIST_DIR}
    --python_out=${CMAKE_CURRENT_LIST_DIR}
  DEPENDS ${CMAKE_CURRENT_LIST_DIR}/schema.proto
)

target_link_libraries(loom
  distributions_shared
  protobuf
)
