#!/bin/bash
#This file, when executed, will:
#  - stop CherryPy server;
#  - remove your database (SqLite DB);
#  - remove your Http session files;
#  - remove possible MatPlotLib script remaining files.
echo off
sh stop.sh
python app.py clean

rm -f session*
rm -f script*.m
rm -f log*
rm -f nohup.out
