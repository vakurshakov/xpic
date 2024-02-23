CXX    := mpicxx
CFLAGS := -std=c++20 -fpic -fopenmp -pthread

# `filter X, A B` return those of A, B that are equal to X
ifeq ($(VERSION), $(filter $(VERSION), "DEBUG" ""))
CFLAGS += -O0 -ggdb -Wall -Wextra -Wpedantic -Wno-unused-parameter -Wno-unused-variable -Werror
PETSC_ARCH := linux-mpi-debug
endif

ifeq ($(VERSION), RELEASE)
CFLAGS += -ftree-vectorize -O3
PETSC_ARCH := linux-mpi-opt
endif

SIMULATION_DIR := $(PWD)
JSON_DIR       := $(PWD)/external/json
PETSC_DIR      := $(PWD)/external/petsc
SPDLOG_DIR     := $(PWD)/external/spdlog

INC_PATH :=
INC_PATH += -I$(SIMULATION_DIR)/
INC_PATH += -I$(JSON_DIR)/include
INC_PATH += -I$(PETSC_DIR)/include -I$(PETSC_DIR)/$(PETSC_ARCH)/include
INC_PATH += -I$(SPDLOG_DIR)/include

LIB_PATH := -L$(PETSC_DIR)/$(PETSC_ARCH)/lib
LIBS := -Wl,-rpath=$(PETSC_DIR)/$(PETSC_ARCH)/lib -lpetsc -lf2clapack -lf2cblas -lm -lX11

EXECUTABLE := simulation.out

RESDIR := bin
OBJDIR := bin-int

# precompiled header
PCH := src/pch.h

# other from src/**
SRCS :=              \
	src/main.cpp       \
	src/constants.cpp  \

SRCS +=                                \
  src/interfaces/simulation.cpp        \

SRCS +=                                \
  src/utils/log.cpp                    \
  src/utils/configuration.cpp          \
  src/utils/mpi_binary_file.cpp        \
  src/utils/sync_binary_file.cpp       \

SRCS +=                                     \
  src/implementations/basic/simulation.cpp  \

SRCS +=													                                 \
	src/implementations/basic/diagnostics/field_view.cpp           \
	src/implementations/basic/diagnostics/fields_energy.cpp        \
	src/implementations/basic/diagnostics/diagnostics_builder.cpp  \

OBJS := $(SRCS:%.cpp=$(OBJDIR)/%.o)
DEPS := $(SRCS:%.cpp=$(OBJDIR)/%.d)


all: $(OBJDIR)/$(PCH).gch $(RESDIR)/$(EXECUTABLE)

-include $(DEPS)

# creates a directory for the target if it doesn't exist
MKDIR=@mkdir -p $(@D)

# using g++ for a precompiled header looks bad
$(OBJDIR)/$(PCH).gch: $(PCH)
	@echo -e "\033[0;33m\nCompiling header src/pch.h.\033[0m"
	$(MKDIR)
	g++ $(CFLAGS) $(INC_PATH) $< -o $@

$(RESDIR)/$(EXECUTABLE): $(OBJS)
	@echo -e "\033[0;33m\nCreating the resulting binary.\033[0m"
	$(MKDIR)
	$(CXX) $(CFLAGS) $^ $(LIB_PATH) $(LIBS) -o $@

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
