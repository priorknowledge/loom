#include "protobuf.hpp"
#include "assignments.hpp"

namespace loom
{

void Assignments::clear ()
{
    keys_.clear();
    for (auto & values : values_) {
        values.clear();
    }
}

void Assignments::load (const char * filename)
{
    clear();

    protobuf::InFile file(filename);
    protobuf::Assignment assignment;

    const size_t dim = this->dim();
    while (file.try_read_stream(assignment)) {
        LOOM_ASSERT_EQ(assignment.groupids_size(), dim);
        auto rowid = assignment.rowid();
        keys_.push(rowid);
        for (size_t i = 0; i < dim; ++i) {
            values_[i].push(assignment.groupids(i));
        }
    }
}

void Assignments::dump (const char * filename) const
{
    protobuf::OutFile file(filename);
    protobuf::Assignment assignment;

    const size_t size = this->size();
    for (size_t i = 0; i < size; ++i) {
        assignment.Clear();
        assignment.set_rowid(keys_[i]);
        for (const auto & values : values_) {
            assignment.add_groupids(values[i]);
        }
        file.write_stream(assignment);
    }
}

} // namespace loom
