# run `bear -- make -B -j` to get compile_commands.json

CC = gcc
CFLAGS = -msse4.2 -O3 -funroll-loops -fopenmp-simd
SRC = tvbk_conn.c
OBJ = $(patsubst %.c,%.o,$(SRC))
SO = libtvbk.so
ctg = tvb_kernels/_ctg_tvbk.py

all: $(SO) $(ctg)

$(SO) : $(OBJ)
	$(CC) -shared $< -o $@
	
$(ctg): tvbk.h
	ctypesgen -l $(SO) $< > $@

tvbk.wasm: em.cpp tvbk.h $(SRC)
	emcc -lembind -o tvbk.js em.cpp $(SRC)
