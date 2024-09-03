#!/bin/bash

bear -- emcc -I../ -lembind -o tvb_kernels.js tvb_kernels.cpp ../tvbk_conn.c
