PROJECT_DIR = $(CURDIR)
BIN_DIR = $(PROJECT_DIR)/bin
SRC_DIR = $(PROJECT_DIR)/src
NO_WARN_UNUSED = -Wno-unused-but-set-variable -Wno-unused-variable

export

all: folders testers

folders:
	mkdir -p ./bin
	
testers: 
	cd $(SRC_DIR)/testers && $(MAKE) -f cresco5.mk testers

clean:
	rm -rf ./bin/*
