CC = gcc
DEBUG = -g -O3
CFLAGS = -Wall $(DEBUG)

all: SLGEWOS

SLGEWOS : 
	cd ./src/SL/GE/WOS && $(MAKE) SLGEWOS

clean:
	\rm *.o *.a *~ *.so
