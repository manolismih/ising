CC = gcc
CPP = g++
FLAGS = -std=c99 -Wall -O3
INCLUDES = -I ./inc

########################################################################

clean:
	find -name *.o -or -name *.run | xargs rm -f
	
test_v0: v0
	./v0.run
	diff test/conf-11.bin test/conf-11.ans

v0: V0_sequential.o main.o
	$(CC) $(FLAGS) $(INCLUDES) V0_sequential.o main.o -o v0.run

%.o: src/%.c
	$(CC) $(FLAGS) $(INCLUDES) -c $<
