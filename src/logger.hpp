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

    typedef protobuf::LogMessage::Args Message;

    template<class Writer>
    void operator() (const Writer & writer)
    {
        if (file_) {
            Message & args = * message_.mutable_args();
            args.Clear();
            writer(args);
            write_message();
        }
    }

private:

    void write_message ();

    protobuf::OutFile * file_;
    protobuf::LogMessage message_;
};

extern Logger logger;

} // namespace loom
