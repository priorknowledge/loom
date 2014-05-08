#pragma once

#include <iostream>
#include <cstdlib>

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
