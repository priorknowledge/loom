#pragma once

#include <loom/common.hpp>
#include <loom/protobuf.hpp>
#include <loom/protobuf_stream.hpp>

namespace loom
{

class Logger
{
public:

    Logger () : file_(nullptr) {}
    ~Logger () { delete file_; }

    operator bool () const { return file_; }

    void open (const char * filename)
    {
        LOOM_ASSERT(not file_, "logger is already open");
        file_ = new protobuf::OutFile(filename);
    }

    typedef protobuf::InferLog::Args Message;

    template<class Writer>
    void log (const Writer & writer)
    {
        if (file_) {
            Message & args = * message_.mutable_args();
            args.Clear();
            writer(args);
            log();
        }
    }

private:

    void log ();

    protobuf::OutFile * file_;
    protobuf::InferLog message_;
};

extern Logger global_logger;

} // namespace loom
