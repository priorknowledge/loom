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

    void load (const protobuf::Checkpoint::StreamInterval & rows)
    {
        Timer::Scope timer(timer_);
        #pragma omp parallel sections
        {
            #pragma omp section
            unassigned_.set_position(rows.unassigned_pos());

            #pragma omp section
            assigned_.set_position(rows.assigned_pos());
        }
    }

    void dump (protobuf::Checkpoint::StreamInterval & rows)
    {
        rows.set_unassigned_pos(unassigned_.position());
        rows.set_assigned_pos(assigned_.position());
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

    template<class Message>
    void read_unassigned (Message & message)
    {
        Timer::Scope timer(timer_);
        unassigned_.cyclic_read_stream(message);
    }

    template<class Message>
    void read_assigned (Message & message)
    {
        Timer::Scope timer(timer_);
        assigned_.cyclic_read_stream(message);
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
