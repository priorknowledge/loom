// Copyright (c) 2014, Salesforce.com, Inc.  All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
// - Redistributions of source code must retain the above copyright
//   notice, this list of conditions and the following disclaimer.
// - Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
// - Neither the name of Salesforce.com nor the names of its contributors
//   may be used to endorse or promote products derived from this
//   software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE
// COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
// OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
// TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
// USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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

    void create (const char * filename)
    {
        LOOM_ASSERT(not file_, "logger is already open");
        file_ = new protobuf::OutFile(filename);
    }

    void append (const char * filename)
    {
        LOOM_ASSERT(not file_, "logger is already open");
        file_ = new protobuf::OutFile(filename, protobuf::OutFile::APPEND);
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
