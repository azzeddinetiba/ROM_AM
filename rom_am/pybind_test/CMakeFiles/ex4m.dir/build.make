# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/tiba/share_directory/ROM_AM/rom_am/pybind_test

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/tiba/share_directory/ROM_AM/rom_am/pybind_test

# Include any dependencies generated for this target.
include CMakeFiles/ex4m.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/ex4m.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/ex4m.dir/flags.make

CMakeFiles/ex4m.dir/example4.cpp.o: CMakeFiles/ex4m.dir/flags.make
CMakeFiles/ex4m.dir/example4.cpp.o: example4.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/tiba/share_directory/ROM_AM/rom_am/pybind_test/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/ex4m.dir/example4.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/ex4m.dir/example4.cpp.o -c /home/tiba/share_directory/ROM_AM/rom_am/pybind_test/example4.cpp

CMakeFiles/ex4m.dir/example4.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ex4m.dir/example4.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/tiba/share_directory/ROM_AM/rom_am/pybind_test/example4.cpp > CMakeFiles/ex4m.dir/example4.cpp.i

CMakeFiles/ex4m.dir/example4.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ex4m.dir/example4.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/tiba/share_directory/ROM_AM/rom_am/pybind_test/example4.cpp -o CMakeFiles/ex4m.dir/example4.cpp.s

# Object files for target ex4m
ex4m_OBJECTS = \
"CMakeFiles/ex4m.dir/example4.cpp.o"

# External object files for target ex4m
ex4m_EXTERNAL_OBJECTS =

ex4m.cpython-38-x86_64-linux-gnu.so: CMakeFiles/ex4m.dir/example4.cpp.o
ex4m.cpython-38-x86_64-linux-gnu.so: CMakeFiles/ex4m.dir/build.make
ex4m.cpython-38-x86_64-linux-gnu.so: CMakeFiles/ex4m.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/tiba/share_directory/ROM_AM/rom_am/pybind_test/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared module ex4m.cpython-38-x86_64-linux-gnu.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ex4m.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/ex4m.dir/build: ex4m.cpython-38-x86_64-linux-gnu.so

.PHONY : CMakeFiles/ex4m.dir/build

CMakeFiles/ex4m.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/ex4m.dir/cmake_clean.cmake
.PHONY : CMakeFiles/ex4m.dir/clean

CMakeFiles/ex4m.dir/depend:
	cd /home/tiba/share_directory/ROM_AM/rom_am/pybind_test && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/tiba/share_directory/ROM_AM/rom_am/pybind_test /home/tiba/share_directory/ROM_AM/rom_am/pybind_test /home/tiba/share_directory/ROM_AM/rom_am/pybind_test /home/tiba/share_directory/ROM_AM/rom_am/pybind_test /home/tiba/share_directory/ROM_AM/rom_am/pybind_test/CMakeFiles/ex4m.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/ex4m.dir/depend

