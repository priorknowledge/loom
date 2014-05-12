#include "logger.hpp"
#include <sys/resource.h>
#include "timer.hpp"

namespace loom
{

Logger global_logger;

void Logger::log (Dict && args)
{
    rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    args("rusage", Dict
        ("max_resident_size_kb", usage.ru_maxrss)
        ("user_time_sec", get_time_sec(usage.ru_utime))
        ("sys_time_sec", get_time_sec(usage.ru_stime))
    );

    file_ << Dict
        ("name", "metrics.loom.runner")
        ("timestamp", current_time_usec())
        ("args", args)
        << std::endl;
}

} // namespace loom
