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

#include <iostream>
#include <cstdlib>
#include <cstring>

class Args
{
public:

    Args (int argc, char ** argv, const char * help_message) :
        argc_(argc - 1),
        argv_(argv + 1),
        help_message_(help_message)
    {}

    const char * pop ()
    {
        if (argc_) {
            --argc_;
            return *argv_++;
        } else {
            std::cerr << help_message_ << std::endl;
            exit(1);
        }
        return nullptr;  // pacify gcc
    }

    const char * pop_optional_file ()
    {
        const char * filename = pop();
        if (strcmp(filename, "--none") == 0) {
            return nullptr;
        } else {
            return filename;
        }
    }

    double pop_default (double default_value)
    {
        if (argc_) {
            --argc_;
            return atof(*argv_++);
        } else {
            return default_value;
        }
    }

    int32_t pop_default (int32_t default_value)
    {
        if (argc_) {
            --argc_;
            return atoi(*argv_++);
        } else {
            return default_value;
        }
    }

    int64_t pop_default (int64_t default_value)
    {
        if (argc_) {
            --argc_;
            return atol(*argv_++);
        } else {
            return default_value;
        }
    }

    void done ()
    {
        if (argc_ > 0) {
            std::cerr << help_message_ << std::endl;
            exit(1);
        }
    }

private:
    int argc_;
    char ** argv_;
    const char * help_message_;
};
