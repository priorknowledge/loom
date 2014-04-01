#pragma once

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
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
class InFileStream
{
public:

    InFileStream (const char * filename)
    {
        fid_ = open(filename, O_RDONLY | O_NOATIME);
        LOOM_ASSERT(fid_ != -1, "failed to open values file");
        raw_input_ = new google::protobuf::io::FileInputStream(fid_);
        coded_input_ = new google::protobuf::io::CodedInputStream(raw_input_);
    }

    ~InFileStream ()
    {
        delete coded_input_;
        delete raw_input_;
        close(fid_);
    }

    template<class Message>
    bool try_read (Message & message)
    {
        return parse_delimited(* coded_input_, message);
    }

private:

    int fid_;
    google::protobuf::io::ZeroCopyInputStream * raw_input_;
    google::protobuf::io::CodedInputStream * coded_input_;
};

} // namespace protobuf
} // namespace loom
