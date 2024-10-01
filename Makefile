test: test.o local.o
	gcc -o test test.o local.o -lm
	rm local.o test.o

test.o: supsmu/test.c supsmu/local.h
	gcc -c supsmu/test.c -lm

local.o: supsmu/local.c supsmu/local.h
	gcc -c supsmu/local.c -lm

clean:
	rm local.o test.o

