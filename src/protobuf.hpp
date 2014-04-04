#pragma once

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/gzip_stream.h>
#include "common.hpp"
#include "schema.pb.h"

namespace loom
{
namespace protobuf
{

using namespace ::protobuf::loom;

inline bool startswith (const char * filename, const char * prefix)
{
    return strlen(filename) >= strlen(prefix) and
        strncmp(filename, prefix, strlen(prefix)) == 0;
}

inline bool endswith (const char * filename, const char * suffix)
{
    return strlen(filename) >= strlen(suffix) and
        strcmp(filename + strlen(filename) - strlen(suffix), suffix) == 0;
}

// References:
// https://groups.google.com/forum/#!topic/protobuf/UBZJXJxR7QY
// https://developers.google.com/protocol-buffers/docs/reference/cpp

class InFile
{
public:

    InFile (const char * filename) : filename_(filename)
    {
        if (startswith(filename, "-")) {
            fid_ = -1;
            raw_ = new google::protobuf::io::IstreamInputStream(& std::cin);
        } else {
            fid_ = open(filename, O_RDONLY | O_NOATIME);
            LOOM_ASSERT(fid_ != -1, "failed to open values file");
            raw_ = new google::protobuf::io::FileInputStream(fid_);
        }

        if (endswith(filename, ".gz")) {
            gzip_ = new google::protobuf::io::GzipInputStream(raw_);
            coded_ = new google::protobuf::io::CodedInputStream(gzip_);
        } else {
            gzip_ = nullptr;
            coded_ = new google::protobuf::io::CodedInputStream(raw_);
        }
    }

    ~InFile ()
    {
        delete coded_;
        delete gzip_;
        delete raw_;
        if (fid_ != -1) {
            close(fid_);
        }
    }

    template<class Message>
    void read (Message & message)
    {
        bool success = message.ParseFromCodedStream(coded_);
        LOOM_ASSERT(success, "failed to parse message from " << filename_);
    }

    template<class Message>
    bool try_read_stream (Message & message)
    {
        uint32_t message_size = 0;
        if (likely(coded_->ReadLittleEndian32(& message_size))) {
            auto old_limit = coded_->PushLimit(message_size);
            bool success = message.ParseFromCodedStream(coded_);
            LOOM_ASSERT(success, "failed to parse message from " << filename_);
            coded_->PopLimit(old_limit);
            return true;
        } else {
            return false;
        }
    }

private:

    const std::string filename_;
    int fid_;
    google::protobuf::io::ZeroCopyInputStream * raw_;
    google::protobuf::io::GzipInputStream * gzip_;
    google::protobuf::io::CodedInputStream * coded_;
};


class OutFile
{
public:

    OutFile (const char * filename) : filename_(filename)
    {
        if (startswith(filename, "-")) {
            fid_ = -1;
            raw_ = new google::protobuf::io::OstreamOutputStream(& std::cout);
        } else {
            fid_ = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
            LOOM_ASSERT(fid_ != -1, "failed to open file " << filename_);
            raw_ = new google::protobuf::io::FileOutputStream(fid_);
        }

        if (endswith(filename, ".gz")) {
            gzip_ = new google::protobuf::io::GzipOutputStream(raw_);
            coded_ = new google::protobuf::io::CodedOutputStream(gzip_);
        } else {
            gzip_ = nullptr;
            coded_ = new google::protobuf::io::CodedOutputStream(raw_);
        }
    }

    ~OutFile ()
    {
        delete coded_;
        delete gzip_;
        delete raw_;
        if (fid_ != -1) {
            close(fid_);
        }
    }

    template<class Message>
    void write (Message & message)
    {
        LOOM_ASSERT1(message.IsInitialized(), "message not initialized");
        message.ByteSize();
        bool success = message.SerializeWithCachedSizes(coded_);
        LOOM_ASSERT(success, "failed to serialize message to " << filename_);
    }

    template<class Message>
    void write_stream (Message & message)
    {
        LOOM_ASSERT1(message.IsInitialized(), "message not initialized");
        uint32_t message_size = message.ByteSize();
        LOOM_ASSERT1(message_size > 0, "zero sized message");
        coded_->WriteLittleEndian32(message_size);
        message.SerializeWithCachedSizes(coded_);
    }

private:

    const std::string filename_;
    int fid_;
    google::protobuf::io::ZeroCopyOutputStream * raw_;
    google::protobuf::io::GzipOutputStream * gzip_;
    google::protobuf::io::CodedOutputStream * coded_;
};

} // namespace protobuf
} // namespace loom
