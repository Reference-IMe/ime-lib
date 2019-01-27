PROJECT_DIR = $(CURDIR)
BIN_DIR = $(PROJECT_DIR)/bin
SRC_DIR = $(PROJECT_DIR)/src
NO_WARN_UNUSED = -Wno-unused-but-set-variable -Wno-unused-result -Wno-unused-variable

export

all: folders SLGEWOS-compare

folders:
	mkdir -p ./bin
	
SLGEWOS-compare: 
	cd $(SRC_DIR)/testers && $(MAKE) SLGEWOS-compare

clean:
	rm ./bin/*
