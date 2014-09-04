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

import os
from libcpp cimport bool
from libcpp.vector cimport vector
from libc.stdint cimport uint32_t, uint64_t


cdef extern from "loom/schema.pb.h":
    ctypedef enum Sparsity "protobuf::loom::ProductValue::Observed::Sparsity":
        SPARSITY_NONE "protobuf::loom::ProductValue::Observed::NONE"
        SPARSITY_SPARSE "protobuf::loom::ProductValue::Observed::SPARSE"
        SPARSITY_DENSE "protobuf::loom::ProductValue::Observed::DENSE"
        SPARSITY_ALL "protobuf::loom::ProductValue::Observed::ALL"

    cppclass Observed_cc "protobuf::loom::ProductValue::Observed":
        Observed_cc "protobuf::loom::ProductValue::Observed" () nogil except +
        void Clear () nogil except +
        Sparsity sparsity () nogil except +
        void set_sparsity (Sparsity) nogil except +
        int dense_size () nogil except +
        bool dense (int index) nogil except +
        void add_dense (bool value) nogil except +
        int sparse_size () nogil except +
        uint32_t sparse (int index) nogil except +
        void add_sparse (uint32_t value) nogil except +

    cppclass Value_cc "protobuf::loom::ProductValue":
        Value_cc "Value" () nogil except +
        void Clear () nogil except +
        int ByteSize () nogil except +
        Observed_cc * observed "mutable_observed" () nogil except +
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
        Value_cc * pos "mutable_diff()->mutable_pos" () except +
        Value_cc * neg "mutable_diff()->mutable_neg" () except +

    cppclass Assignment_cc "protobuf::loom::Assignment":
        Assignment_cc "Assignment" () nogil except +
        void Clear () nogil except +
        int ByteSize () nogil except +
        uint64_t rowid () except +
        void set_rowid (uint64_t value) except +
        int groupids_size () nogil except +
        uint32_t groupids (int index) nogil except +
        void add_groupids (uint32_t value) nogil except +

    cppclass Request_cc "protobuf::loom::Query::Request":
        Request_cc "Request" () nogil except +

    cppclass Response_cc "protobuf::loom::Query::Response":
        Response_cc "Response" () nogil except +


cdef dict SPARSITY_ERRORS = {
    SPARSITY_NONE: "invalid sparsity type: NONE",
    SPARSITY_SPARSE: "invalid sparsity type: SPARSE",
    SPARSITY_DENSE: "invalid sparsity type: DENSE",
    SPARSITY_ALL: "invalid sparsity type: ALL",
}


cdef extern from "loom/protobuf_stream.hpp" namespace "loom::protobuf":
    cppclass InFile:
        InFile (int fid) nogil except +
        InFile (char * filename) nogil except +
        bool try_read_stream[Message] (Message & message) nogil except +

    cppclass OutFile:
        OutFile (int fid) nogil except +
        OutFile (char * filename) nogil except +
        void write_stream[Message] (Message & message) nogil except +
        void flush () nogil except +


cdef class Row:
    cdef Row_cc * ptr

    def __cinit__(self):
        self.ptr = new Row_cc()
        self.ptr.pos().observed().set_sparsity(SPARSITY_DENSE)
        self.ptr.neg().observed().set_sparsity(SPARSITY_NONE)

    def __dealloc__(self):
        del self.ptr

    def Clear(self):
        self.ptr.Clear()
        self.ptr.pos().observed().set_sparsity(SPARSITY_DENSE)
        self.ptr.neg().observed().set_sparsity(SPARSITY_NONE)

    def ByteSize(self):
        return self.ptr.ByteSize()

    property id:
        def __set__(self, uint64_t id_):
            self.ptr.set_id(id_)

        def __get__(self):
            return self.ptr.id()

    def observed_size(self):
        assert self.ptr.pos().observed().sparsity() == SPARSITY_DENSE,\
            SPARSITY_ERRORS[self.ptr.pos().observed().sparsity()]
        return self.ptr.pos().observed().dense_size()

    def observed(self, int index):
        assert self.ptr.pos().observed().sparsity() == SPARSITY_DENSE,\
            SPARSITY_ERRORS[self.ptr.pos().observed().sparsity()]
        return self.ptr.pos().observed().dense(index)

    def add_observed(self, bool value):
        assert self.ptr.pos().observed().sparsity() == SPARSITY_DENSE,\
            SPARSITY_ERRORS[self.ptr.pos().observed().sparsity()]
        self.ptr.pos().observed().add_dense(value)

    def iter_observed(self):
        assert self.ptr.pos().observed().sparsity() == SPARSITY_DENSE,\
            SPARSITY_ERRORS[self.ptr.pos().observed().sparsity()]
        cdef int i
        for i in xrange(self.ptr.pos().observed().dense_size()):
            yield self.ptr.pos().observed().dense(i)
        raise StopIteration()

    def booleans_size(self):
        return self.ptr.pos().booleans_size()

    def booleans(self, int index):
        return self.ptr.pos().booleans(index)

    def add_booleans(self, bool value):
        self.ptr.pos().add_booleans(value)

    def iter_booleans(self):
        cdef int i
        for i in xrange(self.booleans_size()):
            yield self.booleans(i)
        raise StopIteration()

    def counts_size(self):
        return self.ptr.pos().counts_size()

    def counts(self, int index):
        return self.ptr.pos().counts(index)

    def add_counts(self, uint32_t value):
        self.ptr.pos().add_counts(value)

    def iter_counts(self):
        cdef int i
        for i in xrange(self.counts_size()):
            yield self.counts(i)
        raise StopIteration()

    def reals_size(self):
        return self.ptr.pos().reals_size()

    def reals(self, int index):
        return self.ptr.pos().reals(index)

    def add_reals(self, float value):
        self.ptr.pos().add_reals(value)

    def iter_reals(self):
        cdef int i
        for i in xrange(self.reals_size()):
            yield self.reals(i)
        raise StopIteration()

    def dump(self):
        return {
            'id': self.id,
            'data': {
                'observed': list(self.iter_observed()),
                'booleans': list(self.iter_booleans()),
                'counts': list(self.iter_counts()),
                'reals': list(self.iter_reals()),
            }
        }

    def iter_data(self):
        return {
            'observed': self.iter_observed(),
            'booleans': self.iter_booleans(),
            'counts': self.iter_counts(),
            'reals': self.iter_reals(),
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


cdef class Request:
    cdef Request_cc * ptr

    def __cinit__(self):
        self.ptr = new Request_cc()

    def __dealloc__(self):
        del self.ptr


cdef class Response:
    cdef Response_cc * ptr

    def __cinit__(self):
        self.ptr = new Response_cc()

    def __dealloc__(self):
        del self.ptr


def make_dir_for(filename):
    dirname = os.path.dirname(filename)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname)


def row_stream_dump(stream, char * filename):
    make_dir_for(filename)
    cdef OutFile * f = new OutFile(filename)
    cdef Row message
    for message in stream:
        f.write_stream(message.ptr[0])
    del f


def row_stream_load(char * filename):
    cdef InFile * f = new InFile(filename)
    cdef Row message = Row()
    while f.try_read_stream(message.ptr[0]):
        yield message
    del f


def assignment_stream_dump(stream, char * filename):
    make_dir_for(filename)
    cdef OutFile * f = new OutFile(filename)
    cdef Assignment message
    for message in stream:
        f.write_stream(message.ptr[0])
    del f


def assignment_stream_load(char * filename):
    cdef InFile * f = new InFile(filename)
    cdef Assignment message = Assignment()
    while f.try_read_stream(message.ptr[0]):
        yield message
    del f


cdef class RequestStream:
    cdef OutFile * ptr

    def __cinit__(self, int fid):
        self.ptr = new OutFile(fid)

    def __dealloc__(self):
        del self.ptr

    def write(self, Request message):
        self.ptr.write_stream(message.ptr[0])
        self.ptr.flush()


cdef class ResponseStream:
    cdef InFile * ptr

    def __cinit__(self, int fid):
        self.ptr = new InFile(fid)

    def __dealloc__(self):
        del self.ptr

    def read(self, Response message):
        if not self.ptr.try_read_stream(message.ptr[0]):
            raise IOError('ResponseStream ended early')
