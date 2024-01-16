SIMULATION_DIR := $(PWD)
SPDLOG_DIR     := $(PWD)/external/spdlog
JSON_DIR       := $(PWD)/external/json

INC_PATH :=
INC_PATH += -I$(SIMULATION_DIR)/
INC_PATH += -I$(SPDLOG_DIR)/include
INC_PATH += -I$(JSON_DIR)/include

LIB_PATH :=
LIBS :=

CXX    := g++
CFLAGS :=
CFLAGS += -std=c++20 -fpic -fopenmp -pthread

# `filter X, A B` return those of A, B that are equal to X
ifeq ($(VERSION), $(filter $(VERSION), "DEBUG" ""))
CFLAGS += -O0 -ggdb -Wall -Wextra -Wpedantic -Wno-unused-parameter -Werror
endif

ifeq ($(VERSION), RELEASE)
CFLAGS += -ftree-vectorize -O3
endif


EXECUTABLE := simulation.out

RESDIR := bin
OBJDIR := bin-int

# precompiled header
PCH := src/pch.h

# other from src/**
SRCS := src/main.cpp

# interfaces
SRCS +=																	 \
	src/interfaces/simulation.cpp          \
	src/interfaces/simulation_factory.cpp  \

SRCS +=                                  \
  src/utils/log.cpp                      \
  src/utils/binary_file.cpp              \
  src/utils/time_manager.cpp             \
  src/utils/configuration.cpp            \

OBJS := $(SRCS:%.cpp=$(OBJDIR)/%.o)
DEPS := $(SRCS:%.cpp=$(OBJDIR)/%.d)


all: $(OBJDIR)/$(PCH).gch $(RESDIR)/$(EXECUTABLE)

-include $(DEPS)

# creates a directory for the target if it doesn't exist
MKDIR=@mkdir -p $(@D)

$(OBJDIR)/$(PCH).gch: $(PCH)
	@echo -e "\033[0;33m\nCompiling header src/pch.h.\033[0m"
	$(MKDIR)
	$(CXX) $(CFLAGS) $(INC_PATH) $< -o $@

$(RESDIR)/$(EXECUTABLE): $(OBJS)
	@echo -e "\033[0;33m\nCreating the resulting binary.\033[0m"
	$(MKDIR)
	$(CXX) $(CFLAGS) $(LIB_PATH) $^ -Wl,-rpath=$(SPDLOG_DIR)/lib $(LIBS) -o $@

$(OBJDIR)/%.o: %.cpp message_compiling
	$(MKDIR)
	$(CXX) $(CFLAGS) $(INC_PATH) -c $< -o $@

$(OBJDIR)/%.d: %.cpp
	$(MKDIR)
	@$(CXX) $(CFLAGS) $(INC_PATH) $< -MM -MT $(@:$(OBJDIR)/%.d=$(OBJDIR)/%.o) >$@

.PHONY: clean
clean:
	@rm $(OBJDIR)/$(PCH).gch $(DEPS) $(OBJS) $(RESDIR)/$(EXECUTABLE)

# to prevent multiple messages
.INTERMEDIATE: message_compiling
message_compiling:
	@echo -e "\033[0;33m\nCompiling other files from src/**.\033[0m"
