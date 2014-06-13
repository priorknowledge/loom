# Copyright (c) 2014, Salesforce.com, Inc.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# - Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# - Neither the name of Salesforce.com nor the names of its contributors
#   may be used to endorse or promote products derived from this
#   software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
# TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
# USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from libcpp cimport bool
from libcpp.vector cimport vector
from libc.stdint cimport uint32_t, uint64_t


cdef extern from "loom/schema.pb.h":
    cppclass Value_cc "protobuf::loom::ProductModel::Value":
        Value_cc "Value" () nogil except +
        void Clear () nogil except +
        int ByteSize () nogil except +
        int observed_size() nogil except +
        bool observed(int index) nogil except +
        void add_observed(bool value) nogil except +
        int booleans_size () nogil except +
        bool booleans (int index) nogil except +
        void add_booleans (bool value) nogil except +
        int counts_size () nogil except +
        uint32_t counts (int index) nogil except +
        void add_counts (uint32_t value) nogil except +
        int reals_size () nogil except +
        float reals (int index) nogil except +
        void add_reals (float value) nogil except +

    cppclass Row_cc "protobuf::loom::Row":
        Row_cc "Row" () nogil except +
        void Clear () nogil except +
        int ByteSize () nogil except +
        uint64_t id () except +
        void set_id (uint64_t value) except +
        Value_cc * data "mutable_data" () except +

    cppclass Assignment_cc "protobuf::loom::Assignment":
        Assignment_cc "Assignment" () nogil except +
        void Clear () nogil except +
        int ByteSize () nogil except +
        uint64_t rowid () except +
        void set_rowid (uint64_t value) except +
        int groupids_size () nogil except +
        uint32_t groupids (int index) nogil except +
        void add_groupids (uint32_t value) nogil except +


cdef extern from "loom/protobuf_stream.hpp" namespace "loom::protobuf":
    cppclass InFile:
        InFile (char * filename) nogil except +
        bool try_read_stream[Message] (Message & message) nogil except +

    cppclass OutFile:
        OutFile (char * filename) nogil except +
        void write_stream[Message] (Message & message) nogil except +


cdef class Row:
    cdef Row_cc * ptr

    def __cinit__(self):
        self.ptr = new Row_cc()

    def __dealloc__(self):
        del self.ptr

    def Clear(self):
        self.ptr.Clear()

    def ByteSize(self):
        return self.ptr.ByteSize()

    property id:
        def __set__(self, uint64_t id_):
            self.ptr.set_id(id_)

        def __get__(self):
            return self.ptr.id()

    def observed_size(self):
        return self.ptr.data().observed_size()

    def observed(self, int index):
        return self.ptr.data().observed(index)

    def add_observed(self, bool value):
        self.ptr.data().add_observed(value)

    def booleans_size(self):
        return self.ptr.data().booleans_size()

    def booleans(self, int index):
        return self.ptr.data().booleans(index)

    def add_booleans(self, bool value):
        self.ptr.data().add_booleans(value)

    def counts_size(self):
        return self.ptr.data().counts_size()

    def counts(self, int index):
        return self.ptr.data().counts(index)

    def add_counts(self, uint32_t value):
        self.ptr.data().add_counts(value)

    def reals_size(self):
        return self.ptr.data().reals_size()

    def reals(self, int index):
        return self.ptr.data().reals(index)

    def add_reals(self, float value):
        self.ptr.data().add_reals(value)

    def dump(self):
        observed = [self.observed(i) for i in xrange(self.observed_size())]
        booleans = [self.booleans(i) for i in xrange(self.booleans_size())]
        counts = [self.counts(i) for i in xrange(self.counts_size())]
        reals = [self.reals(i) for i in xrange(self.reals_size())]
        return {
            'id': self.id,
            'data': {
                'observed': observed,
                'booleans': booleans,
                'counts': counts,
                'reals': reals,
            }
        }


cdef class Assignment:
    cdef Assignment_cc * ptr

    def __cinit__(self):
        self.ptr = new Assignment_cc()

    def __dealloc__(self):
        del self.ptr

    def Clear(self):
        self.ptr.Clear()

    def ByteSize(self):
        return self.ptr.ByteSize()

    property rowid:
        def __set__(self, uint64_t rowid):
            self.ptr.set_rowid(rowid)

        def __get__(self):
            return self.ptr.rowid()

    def groupids_size(self):
        return self.ptr.groupids_size()

    def groupids(self, int index):
        return self.ptr.groupids(index)

    def add_groupids(self, uint32_t value):
        self.ptr.add_groupids(value)

    def dump(self):
        cdef int i
        cdef list groupids = [
            self.ptr.groupids(i)
            for i in xrange(self.ptr.groupids_size())
        ]
        return {'rowid': self.ptr.rowid(), 'groupids': groupids}


def row_stream_dump(stream, char * filename):
    cdef OutFile * f = new OutFile(filename)
    cdef Row message
    for message in stream:
        f.write_stream(message.ptr[0])
    del f


def row_stream_load(filename):
    cdef InFile * f = new InFile(filename)
    cdef Row message = Row()
    while f.try_read_stream(message.ptr[0]):
        yield message
    del f


def assignment_stream_dump(stream, char * filename):
    cdef OutFile * f = new OutFile(filename)
    cdef Assignment message
    for message in stream:
        f.write_stream(message.ptr[0])
    del f


def assignment_stream_load(filename):
    cdef InFile * f = new InFile(filename)
    cdef Assignment message = Assignment()
    while f.try_read_stream(message.ptr[0]):
        yield message
    del f
