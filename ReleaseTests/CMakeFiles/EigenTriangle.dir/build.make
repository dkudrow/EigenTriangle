# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Produce verbose output by default.
VERBOSE = 1

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/dkudrow/Documents/CombBLAS

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/dkudrow/Documents/CombBLAS

# Include any dependencies generated for this target.
include ReleaseTests/CMakeFiles/EigenTriangle.dir/depend.make

# Include the progress variables for this target.
include ReleaseTests/CMakeFiles/EigenTriangle.dir/progress.make

# Include the compile flags for this target's objects.
include ReleaseTests/CMakeFiles/EigenTriangle.dir/flags.make

ReleaseTests/CMakeFiles/EigenTriangle.dir/EigenTriangle.o: ReleaseTests/CMakeFiles/EigenTriangle.dir/flags.make
ReleaseTests/CMakeFiles/EigenTriangle.dir/EigenTriangle.o: ReleaseTests/EigenTriangle.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/dkudrow/Documents/CombBLAS/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object ReleaseTests/CMakeFiles/EigenTriangle.dir/EigenTriangle.o"
	cd /home/dkudrow/Documents/CombBLAS/ReleaseTests && mpicxx   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/EigenTriangle.dir/EigenTriangle.o -c /home/dkudrow/Documents/CombBLAS/ReleaseTests/EigenTriangle.cpp

ReleaseTests/CMakeFiles/EigenTriangle.dir/EigenTriangle.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/EigenTriangle.dir/EigenTriangle.i"
	cd /home/dkudrow/Documents/CombBLAS/ReleaseTests && mpicxx  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/dkudrow/Documents/CombBLAS/ReleaseTests/EigenTriangle.cpp > CMakeFiles/EigenTriangle.dir/EigenTriangle.i

ReleaseTests/CMakeFiles/EigenTriangle.dir/EigenTriangle.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/EigenTriangle.dir/EigenTriangle.s"
	cd /home/dkudrow/Documents/CombBLAS/ReleaseTests && mpicxx  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/dkudrow/Documents/CombBLAS/ReleaseTests/EigenTriangle.cpp -o CMakeFiles/EigenTriangle.dir/EigenTriangle.s

ReleaseTests/CMakeFiles/EigenTriangle.dir/EigenTriangle.o.requires:
.PHONY : ReleaseTests/CMakeFiles/EigenTriangle.dir/EigenTriangle.o.requires

ReleaseTests/CMakeFiles/EigenTriangle.dir/EigenTriangle.o.provides: ReleaseTests/CMakeFiles/EigenTriangle.dir/EigenTriangle.o.requires
	$(MAKE) -f ReleaseTests/CMakeFiles/EigenTriangle.dir/build.make ReleaseTests/CMakeFiles/EigenTriangle.dir/EigenTriangle.o.provides.build
.PHONY : ReleaseTests/CMakeFiles/EigenTriangle.dir/EigenTriangle.o.provides

ReleaseTests/CMakeFiles/EigenTriangle.dir/EigenTriangle.o.provides.build: ReleaseTests/CMakeFiles/EigenTriangle.dir/EigenTriangle.o

# Object files for target EigenTriangle
EigenTriangle_OBJECTS = \
"CMakeFiles/EigenTriangle.dir/EigenTriangle.o"

# External object files for target EigenTriangle
EigenTriangle_EXTERNAL_OBJECTS =

ReleaseTests/EigenTriangle: ReleaseTests/CMakeFiles/EigenTriangle.dir/EigenTriangle.o
ReleaseTests/EigenTriangle: ReleaseTests/CMakeFiles/EigenTriangle.dir/build.make
ReleaseTests/EigenTriangle: libCommGridlib.a
ReleaseTests/EigenTriangle: libMPITypelib.a
ReleaseTests/EigenTriangle: libMemoryPoollib.a
ReleaseTests/EigenTriangle: ReleaseTests/CMakeFiles/EigenTriangle.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable EigenTriangle"
	cd /home/dkudrow/Documents/CombBLAS/ReleaseTests && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/EigenTriangle.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
ReleaseTests/CMakeFiles/EigenTriangle.dir/build: ReleaseTests/EigenTriangle
.PHONY : ReleaseTests/CMakeFiles/EigenTriangle.dir/build

ReleaseTests/CMakeFiles/EigenTriangle.dir/requires: ReleaseTests/CMakeFiles/EigenTriangle.dir/EigenTriangle.o.requires
.PHONY : ReleaseTests/CMakeFiles/EigenTriangle.dir/requires

ReleaseTests/CMakeFiles/EigenTriangle.dir/clean:
	cd /home/dkudrow/Documents/CombBLAS/ReleaseTests && $(CMAKE_COMMAND) -P CMakeFiles/EigenTriangle.dir/cmake_clean.cmake
.PHONY : ReleaseTests/CMakeFiles/EigenTriangle.dir/clean

ReleaseTests/CMakeFiles/EigenTriangle.dir/depend:
	cd /home/dkudrow/Documents/CombBLAS && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/dkudrow/Documents/CombBLAS /home/dkudrow/Documents/CombBLAS/ReleaseTests /home/dkudrow/Documents/CombBLAS /home/dkudrow/Documents/CombBLAS/ReleaseTests /home/dkudrow/Documents/CombBLAS/ReleaseTests/CMakeFiles/EigenTriangle.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : ReleaseTests/CMakeFiles/EigenTriangle.dir/depend

