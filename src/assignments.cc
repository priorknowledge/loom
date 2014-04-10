#include "protobuf.hpp"
#include "assignments.hpp"

namespace loom
{

void Assignments::load (const char * filename)
{
    protobuf::InFile file(filename);
    protobuf::Assignment assignment;

    while (file.try_read_stream(assignment)) {
        LOOM_ASSERT_EQ(assignment.groupids_size(), dim_);
        auto rowid = assignment.rowid();
        auto * groupid = try_add(rowid);
        LOOM_ASSERT(groupid, "duplicate rowid: " << rowid);
        for (size_t i = 0; i < dim_; ++i) {
            groupid[i] = assignment.groupids(i);
        }
    }
}

void Assignments::dump (const char * filename) const
{
    protobuf::OutFile file(filename);
    protobuf::Assignment assignment;

    for (const auto & pair : map_) {
        assignment.Clear();
        assignment.set_rowid(pair.first);
        for (size_t i = 0; i < dim_; ++i) {
            assignment.add_groupids(pair.second[i]);
        }
        file.write_stream(assignment);
    }
}

} // namespace loom
