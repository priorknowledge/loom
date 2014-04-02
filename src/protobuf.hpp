#pragma once

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/gzip_stream.h>
#include "common.hpp"

namespace loom
{
namespace protobuf
{


// adapted from https://groups.google.com/forum/#!topic/protobuf/UBZJXJxR7QY

template <typename Message>
inline void serialize_delimited (
        google::protobuf::io::CodedOutputStream & stream,
        Message & message)
{
    LOOM_ASSERT1(message.IsInitialized(), "message not initialized");
    uint32_t message_size = message.ByteSize();
    LOOM_ASSERT1(message_size > 0, "zero sized message");
    stream.WriteLittleEndian32(message_size);
    message.SerializeWithCachedSizes(& stream);
}

template <typename Message>
inline bool parse_delimited (
        google::protobuf::io::CodedInputStream & stream,
        Message & message)
{
    uint32_t message_size = 0;
    auto old_limit = stream.PushLimit(sizeof(uint32_t));
    stream.ReadLittleEndian32(& message_size);
    stream.PopLimit(old_limit);

    if (message_size == 0) {
        return false;
    } else {
        auto old_limit = stream.PushLimit(message_size);
        message.ParseFromCodedStream(& stream);
        stream.PopLimit(old_limit);
        return true;
    }
}

// see https://developers.google.com/protocol-buffers/docs/reference/cpp

inline bool endswith (const char * filename, const char * ext)
{
    return strlen(filename) >= strlen(ext) and
        strcmp(filename + strlen(filename) - strlen(ext), ext) == 0;
}

class InFileStream
{
public:

    InFileStream (const char * filename)
    {
        fid_ = open(filename, O_RDONLY | O_NOATIME);
        LOOM_ASSERT(fid_ != -1, "failed to open values file");
        raw_ = new google::protobuf::io::FileInputStream(fid_);
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
        close(fid_);
    }

    template<class Message>
    bool try_read (Message & message)
    {
        return parse_delimited(* coded_, message);
    }

private:

    int fid_;
    google::protobuf::io::FileInputStream * raw_;
    google::protobuf::io::GzipInputStream * gzip_;
    google::protobuf::io::CodedInputStream * coded_;
};

class OutFileStream
{
public:

    OutFileStream (const char * filename)
    {
        fid_ = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        LOOM_ASSERT(fid_ != -1, "failed to open values file");
        raw_ = new google::protobuf::io::FileOutputStream(fid_);
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
        close(fid_);
    }

    template<class Message>
    void write (Message & message)
    {
        return serialize_delimited(* coded_, message);
    }

private:

    int fid_;
    google::protobuf::io::FileOutputStream * raw_;
    google::protobuf::io::GzipOutputStream * gzip_;
    google::protobuf::io::CodedOutputStream * coded_;
};

} // namespace protobuf
} // namespace loom
