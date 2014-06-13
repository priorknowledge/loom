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
#include <loom/assignments.hpp>

namespace loom
{

class StreamInterval : noncopyable
{
public:

    StreamInterval (const char * rows_in) :
        unassigned_(rows_in),
        assigned_(rows_in)
    {
    }

    void load (const protobuf::Checkpoint::StreamInterval & rows)
    {
        #pragma omp parallel sections
        {
            #pragma omp section
            unassigned_.set_position(rows.unassigned_pos());

            #pragma omp section
            assigned_.set_position(rows.assigned_pos());
        }
    }

    void dump (protobuf::Checkpoint::StreamInterval & rows)
    {
        rows.set_unassigned_pos(unassigned_.position());
        rows.set_assigned_pos(assigned_.position());
    }

    void init_from_assignments (const Assignments & assignments)
    {
        LOOM_ASSERT(assignments.row_count(), "nothing to initialize");
        LOOM_ASSERT(assigned_.is_file(), "only files support StreamInterval");

        #pragma omp parallel sections
        {
            #pragma omp section
            seek_first_unassigned_row(assignments);

            #pragma omp section
            seek_first_assigned_row(assignments);
        }
    }

    template<class Message>
    void read_unassigned (Message & message)
    {
        unassigned_.cyclic_read_stream(message);
    }

    template<class Message>
    void read_assigned (Message & message)
    {
        assigned_.cyclic_read_stream(message);
    }

private:

    void seek_first_unassigned_row (const Assignments & assignments)
    {
        const auto last_assigned_rowid = assignments.rowids().back();
        protobuf::Row row;

        while (true) {
            bool success = unassigned_.try_read_stream(row);
            LOOM_ASSERT(success, "row.id not found: " << last_assigned_rowid);
            if (row.id() == last_assigned_rowid) {
                break;
            }
        }
    }

    void seek_first_assigned_row (const Assignments & assignments)
    {
        const auto first_assigned_rowid = assignments.rowids().front();
        protobuf::InFile peeker(unassigned_.filename());
        protobuf::Row row;
        std::vector<char> unused;

        while (true) {
            bool success = peeker.try_read_stream(row);
            LOOM_ASSERT(success, "row.id not found: " << first_assigned_rowid);
            if (row.id() == first_assigned_rowid) {
                break;
            }
            assigned_.try_read_stream(unused);
        }
    }

    protobuf::InFile unassigned_;
    protobuf::InFile assigned_;
};

} // namespace loom
