# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.3

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


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
CMAKE_COMMAND = /opt/local/bin/cmake

# The command to remove a file.
RM = /opt/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/philip/Documents/opencv/aruco_test

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/philip/Documents/opencv/aruco_test/build

# Include any dependencies generated for this target.
include CMakeFiles/aruco_simple.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/aruco_simple.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/aruco_simple.dir/flags.make

CMakeFiles/aruco_simple.dir/aruco_simple.cpp.o: CMakeFiles/aruco_simple.dir/flags.make
CMakeFiles/aruco_simple.dir/aruco_simple.cpp.o: ../aruco_simple.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/philip/Documents/opencv/aruco_test/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/aruco_simple.dir/aruco_simple.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/aruco_simple.dir/aruco_simple.cpp.o -c /Users/philip/Documents/opencv/aruco_test/aruco_simple.cpp

CMakeFiles/aruco_simple.dir/aruco_simple.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/aruco_simple.dir/aruco_simple.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /Users/philip/Documents/opencv/aruco_test/aruco_simple.cpp > CMakeFiles/aruco_simple.dir/aruco_simple.cpp.i

CMakeFiles/aruco_simple.dir/aruco_simple.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/aruco_simple.dir/aruco_simple.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /Users/philip/Documents/opencv/aruco_test/aruco_simple.cpp -o CMakeFiles/aruco_simple.dir/aruco_simple.cpp.s

CMakeFiles/aruco_simple.dir/aruco_simple.cpp.o.requires:

.PHONY : CMakeFiles/aruco_simple.dir/aruco_simple.cpp.o.requires

CMakeFiles/aruco_simple.dir/aruco_simple.cpp.o.provides: CMakeFiles/aruco_simple.dir/aruco_simple.cpp.o.requires
	$(MAKE) -f CMakeFiles/aruco_simple.dir/build.make CMakeFiles/aruco_simple.dir/aruco_simple.cpp.o.provides.build
.PHONY : CMakeFiles/aruco_simple.dir/aruco_simple.cpp.o.provides

CMakeFiles/aruco_simple.dir/aruco_simple.cpp.o.provides.build: CMakeFiles/aruco_simple.dir/aruco_simple.cpp.o


# Object files for target aruco_simple
aruco_simple_OBJECTS = \
"CMakeFiles/aruco_simple.dir/aruco_simple.cpp.o"

# External object files for target aruco_simple
aruco_simple_EXTERNAL_OBJECTS =

aruco_simple: CMakeFiles/aruco_simple.dir/aruco_simple.cpp.o
aruco_simple: CMakeFiles/aruco_simple.dir/build.make
aruco_simple: CMakeFiles/aruco_simple.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/philip/Documents/opencv/aruco_test/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable aruco_simple"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/aruco_simple.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/aruco_simple.dir/build: aruco_simple

.PHONY : CMakeFiles/aruco_simple.dir/build

CMakeFiles/aruco_simple.dir/requires: CMakeFiles/aruco_simple.dir/aruco_simple.cpp.o.requires

.PHONY : CMakeFiles/aruco_simple.dir/requires

CMakeFiles/aruco_simple.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/aruco_simple.dir/cmake_clean.cmake
.PHONY : CMakeFiles/aruco_simple.dir/clean

CMakeFiles/aruco_simple.dir/depend:
	cd /Users/philip/Documents/opencv/aruco_test/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/philip/Documents/opencv/aruco_test /Users/philip/Documents/opencv/aruco_test /Users/philip/Documents/opencv/aruco_test/build /Users/philip/Documents/opencv/aruco_test/build /Users/philip/Documents/opencv/aruco_test/build/CMakeFiles/aruco_simple.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/aruco_simple.dir/depend

