#pragma once

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <vector>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/gzip_stream.h>
#include "common.hpp"

namespace loom
{
namespace protobuf
{

inline bool endswith (const char * filename, const char * suffix)
{
    return strlen(filename) >= strlen(suffix) and
        strcmp(filename + strlen(filename) - strlen(suffix), suffix) == 0;
}

class InFile : noncopyable
{
public:

    InFile (const char * filename) : filename_(filename)
    {
        _open();
    }

    ~InFile ()
    {
        _close();
    }

    bool is_file () const { return fid_ != STDIN_FILENO; }

    template<class Message>
    void read (Message & message)
    {
        bool success = message.ParseFromZeroCopyStream(stream_);
        LOOM_ASSERT(success, "failed to parse message from " << filename_);
    }

    template<class Message>
    bool try_read_stream (Message & message)
    {
        google::protobuf::io::CodedInputStream coded(stream_);
        uint32_t message_size = 0;
        if (LOOM_LIKELY(coded.ReadLittleEndian32(& message_size))) {
            auto old_limit = coded.PushLimit(message_size);
            bool success = message.ParseFromCodedStream(& coded);
            LOOM_ASSERT(success, "failed to parse message from " << filename_);
            coded.PopLimit(old_limit);
            return true;
        } else {
            return false;
        }
    }

    bool try_read_stream (std::vector<char> & raw)
    {
        google::protobuf::io::CodedInputStream coded(stream_);
        uint32_t message_size = 0;
        if (LOOM_LIKELY(coded.ReadLittleEndian32(& message_size))) {
            auto old_limit = coded.PushLimit(message_size);
            raw.resize(message_size);
            bool success = coded.ReadRaw(raw.data(), message_size);
            LOOM_ASSERT(success, "failed to parse message from " << filename_);
            coded.PopLimit(old_limit);
            return true;
        } else {
            return false;
        }
    }

    template<class Message>
    void cyclic_read_stream (Message & message)
    {
        LOOM_ASSERT2(is_file(), "only files support cyclic_read_stream");
        if (LOOM_UNLIKELY(not try_read_stream(message))) {
            _close();
            _open();
            bool success = try_read_stream(message);
            LOOM_ASSERT(success, "stream is empty");
        }
    }

private:

    void _open ()
    {
        if (filename_ == "-" or filename_ == "-.gz") {
            fid_ = STDIN_FILENO;
        } else {
            fid_ = open(filename_.c_str(), O_RDONLY | O_NOATIME);
            LOOM_ASSERT(fid_ != -1, "failed to open input file " << filename_);
        }

        file_ = new google::protobuf::io::FileInputStream(fid_);

        if (endswith(filename_.c_str(), ".gz")) {
            gzip_ = new google::protobuf::io::GzipInputStream(file_);
            stream_ = gzip_;
        } else {
            gzip_ = nullptr;
            stream_ = file_;
        }
    }

    void _close ()
    {
        delete gzip_;
        delete file_;
        if (is_file()) {
            close(fid_);
        }
    }

    const std::string filename_;
    int fid_;
    google::protobuf::io::FileInputStream * file_;
    google::protobuf::io::GzipInputStream * gzip_;
    google::protobuf::io::ZeroCopyInputStream * stream_;
};


class OutFile : noncopyable
{
public:

    OutFile (const char * filename) : filename_(filename)
    {
        if (filename_ == "-" or filename_ == "-.gz") {
            fid_ = STDOUT_FILENO;
        } else {
            fid_ = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0664);
            LOOM_ASSERT(fid_ != -1, "failed to open output file " << filename);
        }

        file_ = new google::protobuf::io::FileOutputStream(fid_);

        if (endswith(filename, ".gz")) {
            gzip_ = new google::protobuf::io::GzipOutputStream(file_);
            stream_ = gzip_;
        } else {
            gzip_ = nullptr;
            stream_ = file_;
        }
    }

    ~OutFile ()
    {
        delete gzip_;
        delete file_;
        if (fid_ != STDOUT_FILENO) {
            close(fid_);
        }
    }

    template<class Message>
    void write (Message & message)
    {
        LOOM_ASSERT1(message.IsInitialized(), "message not initialized");
        bool success = message.SerializeToZeroCopyStream(stream_);
        LOOM_ASSERT(success, "failed to serialize message to " << filename_);
    }

    template<class Message>
    void write_stream (Message & message)
    {
        google::protobuf::io::CodedOutputStream coded(stream_);
        LOOM_ASSERT1(message.IsInitialized(), "message not initialized");
        uint32_t message_size = message.ByteSize();
        coded.WriteLittleEndian32(message_size);
        message.SerializeWithCachedSizes(& coded);
    }

    void write_stream (const std::vector<char> & raw)
    {
        google::protobuf::io::CodedOutputStream coded(stream_);
        coded.WriteLittleEndian32(raw.size());
        coded.WriteRaw(raw.data(), raw.size());
    }

    void flush ()
    {
        if (gzip_) {
            gzip_->Flush();
        }
        file_->Flush();
    }

private:

    const std::string filename_;
    int fid_;
    google::protobuf::io::FileOutputStream * file_;
    google::protobuf::io::GzipOutputStream * gzip_;
    google::protobuf::io::ZeroCopyOutputStream * stream_;
};

} // namespace protobuf

template<class Message>
Message protobuf_load (const char * filename)
{
    Message message;
    protobuf::InFile file(filename);
    file.read(message);
    return message;
}

template<class Message>
Message protobuf_dump (
        const Message & message,
        const char * filename)
{
    protobuf::OutFile file(filename);
    file.write(message);
}

template<class Message>
std::vector<Message> protobuf_stream_load (const char * filename)
{
    std::vector<Message> messages(1);
    protobuf::InFile stream(filename);
    while (stream.try_read_stream(messages.back())) {
        messages.resize(messages.size() + 1);
    }
    messages.pop_back();
    return messages;
}

template<class Message>
void protobuf_stream_dump (
        const std::vector<Message> & messages,
        const char * filename)
{
    protobuf::OutFile stream(filename);
    for (const auto & message : messages) {
        stream.write_stream(message);
    }
}

} // namespace loom
