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

    void init_from_assignments (const Assignments & assignments)
    {
        Timer::Scope timer(timer_);
        LOOM_ASSERT(assignments.row_count(), "nothing to initialize");
        LOOM_ASSERT(assigned_.is_file(), "only files support StreamInterval");

        #pragma omp parallel sections
        {
            #pragma omp section
            seek_first_unassigned_row(assignments);

            #pragma omp section
            seek_first_assigned_row(assignments);
        }
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

    void seek_first_unassigned_row (const Assignments & assignments)
    {
        const auto last_assigned_rowid = assignments.rowids().back();
        protobuf::SparseRow row;

        while (true) {
            bool success = unassigned_.try_read_stream(row);
            LOOM_ASSERT(success, "row.id not found: " << last_assigned_rowid);
            if (row.id() == last_assigned_rowid) {
                break;
            }
        }
    }

    void seek_first_assigned_row (const Assignments & assignments)
    {
        const auto first_assigned_rowid = assignments.rowids().front();
        protobuf::InFile peeker(unassigned_.filename());
        protobuf::SparseRow row;
        std::vector<char> unused;

        while (true) {
            bool success = peeker.try_read_stream(row);
            LOOM_ASSERT(success, "row.id not found: " << first_assigned_rowid);
            if (row.id() == first_assigned_rowid) {
                break;
            }
            assigned_.try_read_stream(unused);
        }
    }

    protobuf::InFile unassigned_;
    protobuf::InFile assigned_;
    Timer timer_;
};

} // namespace loom
