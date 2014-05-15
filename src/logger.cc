#include <loom/logger.hpp>
#include <sys/resource.h>
#include <loom/timer.hpp>

namespace loom
{

Logger global_logger;

void Logger::log ()
{
    LOOM_ASSERT(file_, "logger is not open");

    rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    auto & rusage = * message_.mutable_rusage();
    rusage.set_max_resident_size_kb(usage.ru_maxrss);
    rusage.set_user_time_sec(get_time_sec(usage.ru_utime));
    rusage.set_sys_time_sec(get_time_sec(usage.ru_stime));

    message_.set_timestamp_usec(current_time_usec());

    file_->write_stream(message_);
    file_->flush();
}

} // namespace loom
