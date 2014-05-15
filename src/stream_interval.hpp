#include <loom/common.hpp>
#include <loom/protobuf.hpp>
#include <loom/assignments.hpp>
#include <loom/timer.hpp>
#include <loom/logger.hpp>

namespace loom
{

class StreamInterval : noncopyable
{
public:

    StreamInterval (const char * rows_in) :
        unassigned_(rows_in),
        assigned_(rows_in)
    {
    }

    void init_from_file_offsets (
            size_t first_unassigned,
            size_t first_assigned)
    {
        LOOM_ASSERT_NE(first_unassigned, first_assigned);
        TODO("position read heads from offsets");
    }

    void init_and_read_assigned (
            protobuf::SparseRow & row,
            Assignments & assignments)
    {
        Timer::Scope timer(timer_);
        LOOM_ASSERT(assignments.row_count(), "nothing to initialize");
        LOOM_ASSERT(assigned_.is_file(), "only files support StreamInterval");

        // point unassigned at first unassigned row
        const auto last_assigned_rowid = assignments.rowids().back();
        do {
            read_unassigned(row);
        } while (row.id() != last_assigned_rowid);

        // point rows_assigned at first assigned row
        const auto first_assigned_rowid = assignments.rowids().front();
        do {
            read_assigned(row);
        } while (row.id() != first_assigned_rowid);
    }

    void read_unassigned (protobuf::SparseRow & row)
    {
        Timer::Scope timer(timer_);
        unassigned_.cyclic_read_stream(row);
    }

    void read_assigned (protobuf::SparseRow & row)
    {
        Timer::Scope timer(timer_);
        assigned_.cyclic_read_stream(row);
    }

    void log_metrics (Logger::Message & message)
    {
        auto & status = * message.mutable_kernel_status()->mutable_reader();
        status.set_total_time(timer_.total());
        timer_.clear();
    }

private:

    protobuf::InFile unassigned_;
    protobuf::InFile assigned_;
    Timer timer_;
};

} // namespace loom
