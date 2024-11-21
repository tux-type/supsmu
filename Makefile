test: test.o local.o supsmu.o
	gcc -o test test.o local.o supsmu.o -lm -Wall -Wextra -O3
	rm local.o supsmu.o test.o

test.o: supsmu/test.c supsmu/local.h supsmu/supsmu.h
	gcc -c supsmu/test.c -lm -Wall -Wextra -O3

local.o: supsmu/local.c supsmu/local.h
	gcc -c supsmu/local.c -lm -Wall -Wextra -O3

supsmu.o: supsmu/supsmu.c supsmu/supsmu.h
	gcc -c supsmu/supsmu.c -lm -Wall -Wextra -O3

clean:
	rm local.o test.o supsmu.o

