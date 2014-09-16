#!/bin/bash
# Runs the web part of the software and the computation back-end server, both on your local machine.
python app.py start DEVELOPMENT_PROFILE

# Runs the desktop client interface of TVB, on the local machine.
# export VERSIONER_PYTHON_PREFER_32_BIT=yes
# python app.py start DESKTOP_PROFILE

# Starts a IDLE console with the correct environment set
# python app.py LIBRARY_PROFILE
# python app.py CONSOLE_PROFILE

