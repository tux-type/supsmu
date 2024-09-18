test: test.o linear.o
	gcc -o test test.o linear.o -lm

test.o: supsmu/test.c supsmu/linear.h
	gcc -c supsmu/test.c -lm

linear.o: supsmu/linear.c supsmu/linear.h
	gcc -c supsmu/linear.c -lm
