from libcpp cimport bool
from libc.stdint cimport uint32_t, uint64_t


cdef extern from "distributions/io/protobuf_stream.hpp":
    cppclass InFile_cc "distributions::protobuf::InFile":
        InFile_cc (char * filename) nogil except +
        uint32_t try_read_stream[Message] (Message & message) nogil except +

    cppclass OutFile_cc "distributions::protobuf::OutFile":
        OutFile_cc (char * filename) nogil except +
        void write_stream[Message] (Message & message) nogil except +


cdef extern from "loom/schema.pb.h":
    cppclass SparseValue_cc "protobuf::loom::ProductModel::SparseValue":
        SparseValue_cc "SparseValue" () nogil except +
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

    cppclass SparseRow_cc "protobuf::loom::SparseRow":
        SparseRow_cc "SparseRow" () nogil except +
        void Clear () nogil except +
        int ByteSize () nogil except +
        uint64_t id () except +
        void set_id (uint64_t value) except +
        SparseValue_cc * data "mutable_data" () except +


cdef class SparseRow:
    cdef SparseRow_cc * ptr

    def __cinit__(self):
        self.ptr = new SparseRow_cc()

    def __dealloc__(self):
        del self.ptr

    def Clear(self):
        self.ptr.Clear()

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

    def add_booleans(self, bool value):
        self.ptr.data().add_booleans(value)

    def counts_size(self):
        return self.ptr.data().counts_size()

    def add_counts(self, uint32_t value):
        self.ptr.data().add_counts(value)

    def add_reals(self, float value):
        self.ptr.data().add_reals(value)


cdef class InFile:
    cdef InFile_cc * ptr

    def __cinit__(self, char * filename):
        self.ptr = new InFile_cc(filename)

    def __dealloc__(self):
        del self.ptr

    def try_read_stream(self, SparseRow message):
        return self.ptr.try_read_stream(message.ptr[0])


cdef class OutFile:
    cdef OutFile_cc * ptr

    def __cinit__(self, char * filename):
        self.ptr = new OutFile_cc(filename)

    def __dealloc__(self):
        del self.ptr

    def write_stream(self, SparseRow message):
        self.ptr.write_stream(message.ptr[0])
