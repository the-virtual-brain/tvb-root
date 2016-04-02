// Copyright (c) 2013 Steinwurf ApS
// All Rights Reserved
//
// Distributed under the "BSD License". See the accompanying LICENSE.rst file.

#pragma once

#include <boost/program_options.hpp>

class commandline_arguments
{
    public:

        /// Constructor
        commandline_arguments();

        // void
        template<class T> void add_option(const char* option,
                                          const char* description)
        {
            m_options.add_options()(option, boost::program_options::value<T>(),
                                    description);
        }

        boost::program_options::variables_map parse(int argc, char **argv);

    private:

        /// The options
        boost::program_options::options_description m_options;
};

extern boost::program_options::variables_map variable_map;
