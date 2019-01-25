CC = gcc
DEBUG = -g -O3
#DEBUG = -g -O0
#DEBUG = -O3
CFLAGS = -Wall $(DEBUG)

all: SLGEWOS

SLGEWOS : SLGEWOS.c
	$(CC) $(CFLAGS) SLGEWOS.c -o SLGEWOS

clean:
	\rm *.o *.a *~ *.so
