#
# Makefile
# Tiago Lobato Gimenes (tlgimenes@gmail.com), 
# 2015-02-02 10:38
#

CC=nvcc					# Compiler
C_FLAGS=-std=c++11 -g #-DNDEBUG # C-Compiler flags
NV_FLAGS=-G				# Nvidia compiler flags

SRC=$(wildcard *.cpp)	# Source files in this folder
HPP=-Ikd-tree/include -Iknn/include -Iutils/include  # Headers definitions
OBJ=kd-tree/src/*.o knn/src/*.o utils/src/*.o        # Object files

OUTPUT=cudaknn				# Name of the output file

all:
	cd kd-tree; make
	cd knn; make
	cd utils; make
	$(CC) $(C_FLAGS) $(NV_FLAGS) $(HPP) $(OBJ) $(SRC) -o $(OUTPUT)

clean:
	cd kd-tree; make clean
	cd knn; make clean
	cd utils; make clean
	rm $(OUTPUT)

# vim:ft=make
#

