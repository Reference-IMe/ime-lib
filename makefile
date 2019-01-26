PROJECT_DIR = $(CURDIR)
BIN_DIR = $(PROJECT_DIR)/bin
SRC_DIR = $(PROJECT_DIR)/src
NO_WARN_UNUSED = -Wno-unused-but-set-variable -Wno-unused-result -Wno-unused-variable

export

all: GJE-compare-seq

GJE-compare-seq : 
	cd $(SRC_DIR)/testers && $(MAKE) GJE-compare-seq

clean:
	\rm *.o *.a *~ *.so
