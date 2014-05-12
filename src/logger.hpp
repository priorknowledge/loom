#pragma once

#include <fstream>
#include <sstream>
#include "common.hpp"

namespace loom
{

class Logger
{
public:

    void open (const char * filename) { file_.open(filename); }
    operator bool () const { return file_.is_open(); }

    class Dict;
    void log (Dict && args);

private:

    std::ofstream file_;
};

class Logger::Dict
{
public:

    Dict () : started_(false) {}

    template<class T>
    Dict (const char * key, const T & value)
    {
        message_ << "\"" << key << "\": " << value;
        started_ = true;
    }

    template<class T>
    Dict & operator() (const char * key, const T & value)
    {
        if (started_) {
            message_ << ", \"" << key << "\": " << value;
        } else {
            message_ << "\"" << key << "\": " << value;
            started_ = true;
        }
        return * this;
    }

    friend std::ostream & operator<< (std::ostream & os, const Dict & dict)
    {
        return os << '{' << dict.message_ << '}';
    }

private:

    std::ostringstream message_;
    bool started_;
};

extern Logger global_logger;

} // namespace loom
