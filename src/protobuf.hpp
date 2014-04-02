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

class InFileStream
{
public:

    InFileStream (const char * filename)
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

    ~InFileStream ()
    {
        delete coded_;
        delete gzip_;
        delete raw_;
        if (fid_ != -1) {
            close(fid_);
        }
    }

    template<class Message>
    bool try_read (Message & message)
    {
        uint32_t message_size = 0;
        auto old_limit = coded_->PushLimit(sizeof(uint32_t));
        coded_->ReadLittleEndian32(& message_size);
        coded_->PopLimit(old_limit);

        if (message_size == 0) {
            return false;
        } else {
            auto old_limit = coded_->PushLimit(message_size);
            message.ParseFromCodedStream(coded_);
            coded_->PopLimit(old_limit);
            return true;
        }
    }

private:

    int fid_;
    google::protobuf::io::ZeroCopyInputStream * raw_;
    google::protobuf::io::GzipInputStream * gzip_;
    google::protobuf::io::CodedInputStream * coded_;
};


class OutFileStream
{
public:

    OutFileStream (const char * filename)
    {
        if (startswith(filename, "-")) {
            fid_ = -1;
            raw_ = new google::protobuf::io::OstreamOutputStream(& std::cout);
        } else {
            fid_ = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
            LOOM_ASSERT(fid_ != -1, "failed to open values file");
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

    ~OutFileStream ()
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
        uint32_t message_size = message.ByteSize();
        LOOM_ASSERT1(message_size > 0, "zero sized message");
        coded_->WriteLittleEndian32(message_size);
        message.SerializeWithCachedSizes(coded_);
    }

private:

    int fid_;
    google::protobuf::io::ZeroCopyOutputStream * raw_;
    google::protobuf::io::GzipOutputStream * gzip_;
    google::protobuf::io::CodedOutputStream * coded_;
};

} // namespace protobuf
} // namespace loom
