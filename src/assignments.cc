#include <loom/assignments.hpp>
#include <unordered_map>
#include <distributions/trivial_hash.hpp>
#include <loom/protobuf.hpp>

namespace loom
{

void Assignments::init (size_t kind_count)
{
    clear();
    values_.resize(kind_count);
}

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
    protobuf::Row assignment;

    const size_t kind_count = this->kind_count();
    while (file.try_read_stream(assignment)) {
        keys_.push(assignment.id());
        const auto & groupids = assignment.data().counts();
        LOOM_ASSERT_EQ(groupids.size(), kind_count);
        for (size_t i = 0; i < kind_count; ++i) {
            values_[i].push(groupids.Get(i));
        }
    }
}

void Assignments::dump (
        const char * filename,
        const std::vector<std::vector<uint32_t>> & sorted_to_globals) const
{
    const size_t row_count = this->row_count();
    const size_t kind_count = this->kind_count();

    typedef distributions::TrivialHash<Value> Hash;
    typedef std::unordered_map<Value, Value, Hash> Map;
    std::vector<Map> global_to_sorteds(kind_count);
    for (size_t k = 0; k < kind_count; ++k) {
        Map & global_to_sorted = global_to_sorteds[k];
        const auto & sorted_to_global = sorted_to_globals[k];
        const size_t group_count = sorted_to_global.size();
        for (size_t g = 0; g < group_count; ++g) {
            global_to_sorted[sorted_to_global[g]] = g;
        }
    }

    protobuf::OutFile file(filename);
    protobuf::Row assignment;
    for (size_t r = 0; r < row_count; ++r) {
        assignment.Clear();
        assignment.set_id(keys_[r]);
        auto & groupids = * assignment.mutable_data()->mutable_counts();
        for (size_t k = 0; k < kind_count; ++k) {
            const Map & global_to_sorted = global_to_sorteds[k];
            uint32_t global = values_[k][r];
            auto i = global_to_sorted.find(global);
            LOOM_ASSERT1(i != global_to_sorted.end(), "bad id: " << global);
            groupids.Add(i->second);
        }
        file.write_stream(assignment);
    }
}

} // namespace loom
