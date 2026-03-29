CC = gcc
CFLAGS = -O3 -fPIC -shared

all: liblambda.so

liblambda.so: lambda_native.c
	$(CC) $(CFLAGS) -o liblambda.so lambda_native.c

run: all
	python lambda_decorator.py

clean:
	rm -f liblambda.so

.PHONY: all run clean
