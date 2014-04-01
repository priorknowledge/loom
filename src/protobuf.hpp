#pragma once

#include <google/protobuf/io/coded_stream.h>
#include "common.hpp"

// adapted from https://groups.google.com/forum/#!topic/protobuf/UBZJXJxR7QY
template <typename Message>
bool serialize_delimited(std::ostream & stream, Message & message)
{
    LOOM_ASSERT(message.IsInitialized(), "message not initialized");

    google::protobuf::io::OstreamOutputStream ostreamWrapper(&stream);
    google::protobuf::io::CodedOutputStream codedOStream(&ostreamWrapper);

    // Write the message size first.
    int message_size = message.ByteSize();
    LOOM_ASSERT(message_size > 0. "zero sized message");
    codedOStream.WriteLittleEndian32(message_size);

    // Write the message.
    message.SerializeWithCachedSizes(&codedOStream);

    return stream.good();
}

template <typename Message>
bool parse_delimited(std::istream & stream, Message& message)
{
    uint32_t message_size = 0;

    // Read the message size.
    {
        google::protobuf::io::IstreamInputStream istreamWrapper(
            &stream,
            sizeof(uint32_t));
        google::protobuf::io::CodedInputStream codedIStream(&istreamWrapper);

        // Don't consume more than sizeof(uint32_t) from the stream.
        google::protobuf::io::CodedInputStream::Limit oldLimit =
            codedIStream.PushLimit(sizeof(uint32_t));
        codedIStream.ReadLittleEndian32(&message_size);
        codedIStream.PopLimit(oldLimit);
        LOOM_ASSERT(message_size > 0, "zero sized message");
        LOOM_ASSERT(
            istreamWrapper.ByteCount() == sizeof(uint32_t));
    }

    // Read the message.
    {
        google::protobuf::io::IstreamInputStream istreamWrapper(&stream,
message_size);
        google::protobuf::io::CodedInputStream
codedIStream(&istreamWrapper);

        // Read the message, but don't consume more than message_size bytes
from the stream.
        google::protobuf::io::CodedInputStream::Limit oldLimit =
codedIStream.PushLimit(message_size);
        message.ParseFromCodedStream(&codedIStream);
        codedIStream.PopLimit(oldLimit);
        assert(istreamWrapper.ByteCount() == message_size);
    }

    return stream.good();
}
