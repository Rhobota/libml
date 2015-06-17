
INCLUDE_DIR := include
SRC_DIR     := source
OBJ_DIR     := objects
TESTS_DIR   := tests

STATIC_LIB_NAME := libml.a

CC := $(TARGET)g++
AR := $(TARGET)ar
CC_FLAGS_LOCAL := $(CC_FLAGS) \
	-g -O3 -fvisibility=hidden -Wall -Wextra -Werror -pedantic \
	-Wswitch-default -Wcast-qual -Wcast-align -Wconversion \
	-Wno-unused-parameter -Wno-long-long -Wno-sign-conversion \
	-fopenmp \
	-D_FILE_OFFSET_BITS=64 \
	-I ../librho/include \
	-I $(INCLUDE_DIR)  # consider: -Wold-style-cast -Wshadow

ifeq ($(shell uname),Linux)
	# Linux stuff:
	CC_FLAGS_LOCAL += -rdynamic -Wdouble-promotion
else
ifeq ($(shell uname),Darwin)
	# OSX stuff:
	CC_FLAGS_LOCAL += -mmacosx-version-min=10.6
else
	# Mingw and Cygwin stuff:
endif
endif

CC_CUDA := /usr/local/cuda-6.5/bin/nvcc
CC_CUDA_FLAGS_LOCAL := $(CC_FLAGS) \
	-g -lineinfo -O3 -Xcompiler -fvisibility=hidden -Xcompiler -Wall -Xcompiler -Wextra -Xcompiler -Werror \
	-Xcompiler -Wswitch-default -Xcompiler -Wcast-align \
	-Xcompiler -Wno-unused-parameter -Xcompiler -Wno-long-long -Xcompiler -Wno-sign-conversion \
	-Xcompiler -fopenmp \
	-D_FILE_OFFSET_BITS=64 \
	-I ../librho/include \
	-I $(INCLUDE_DIR)  # helpful: -Xptxas="-v"

CPP_SRC_FILES = $(shell find $(SRC_DIR) -name '*.cpp' -type f)
CPP_OBJ_FILES = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(CPP_SRC_FILES))

CU_SRC_FILES = $(shell find $(SRC_DIR) -name '*.cu' -type f)
CU_OBJ_FILES = $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(CU_SRC_FILES))

all : $(CPP_OBJ_FILES) $(CU_OBJ_FILES)

test : all
	@$(TESTS_DIR)/RunTests.bash

clean :
	@rm -rf $(OBJ_DIR)
	@echo "Clean successful."

$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cpp
	@echo "Compiling $< ..."
	@mkdir -p $(@D)
	$(CC) $(CC_FLAGS_LOCAL) -c -o $@ $<
	$(AR) crsv $(OBJ_DIR)/$(STATIC_LIB_NAME) $@
	@echo

$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cu
	@echo "Compiling $< ..."
	@mkdir -p $(@D)
	$(CC_CUDA) $(CC_CUDA_FLAGS_LOCAL) -c -o $@ $<
	$(AR) crsv $(OBJ_DIR)/$(STATIC_LIB_NAME) $@
	@echo

.PHONY : all test clean
