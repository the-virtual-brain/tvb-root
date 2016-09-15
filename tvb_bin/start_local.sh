#!/bin/bash
# Runs the web part of the software and the computation back-end server, both on your local machine.
python tvb_bin/app.py start WEB_PROFILE

# Runs the desktop client interface of TVB, on the local machine.
# export VERSIONER_PYTHON_PREFER_32_BIT=yes
# python tvb_bin/app.py start DESKTOP_PROFILE

# Starts a IDLE console with the correct scripting environment set
# python tvb_bin/app.py start LIBRARY_PROFILE
# python tvb_bin/app.py start COMMAND_PROFILE

