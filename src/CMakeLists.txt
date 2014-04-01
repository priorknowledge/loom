
add_executable(loom loom.cc)

target_link_libraries(loom
  distributions_shared
  protobuf
)
