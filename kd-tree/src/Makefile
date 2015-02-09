#
# Makefile
# Tiago Lobato Gimenes (tlgimenes@gmail.com), 
# 2015-02-02 10:46
#

CC=nvcc					# Compiler
C_FLAGS=-g -std=c++11 -O0# C-Compiler flags
NV_FLAGS=-G				# Nvidia compiler flags

SRC=$(wildcard *.cpp)	# Source files in this folder
OBJ=$(SRC:.cpp=.o)		# Object files

HPP=-I../include -I../../utils/include

all: $(OBJ)

%.o: %.cpp
	$(CC) -c $(C_FLAGS) $(NV_FLAGS) $(HPP) $< -o $@

clean:
	rm *.o

# vim:ft=make
#

