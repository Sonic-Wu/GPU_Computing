# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.15

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
CMAKE_COMMAND = /snap/clion/107/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /snap/clion/107/bin/cmake/linux/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/xinyu/Dropbox/GPU_computing/source/homework0

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/xinyu/Dropbox/GPU_computing/source/homework0/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/hello.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/hello.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/hello.dir/flags.make

CMakeFiles/hello.dir/hello_generated_main.cu.o: CMakeFiles/hello.dir/hello_generated_main.cu.o.depend
CMakeFiles/hello.dir/hello_generated_main.cu.o: CMakeFiles/hello.dir/hello_generated_main.cu.o.Debug.cmake
CMakeFiles/hello.dir/hello_generated_main.cu.o: ../main.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/xinyu/Dropbox/GPU_computing/source/homework0/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building NVCC (Device) object CMakeFiles/hello.dir/hello_generated_main.cu.o"
	cd /home/xinyu/Dropbox/GPU_computing/source/homework0/cmake-build-debug/CMakeFiles/hello.dir && /snap/clion/107/bin/cmake/linux/bin/cmake -E make_directory /home/xinyu/Dropbox/GPU_computing/source/homework0/cmake-build-debug/CMakeFiles/hello.dir//.
	cd /home/xinyu/Dropbox/GPU_computing/source/homework0/cmake-build-debug/CMakeFiles/hello.dir && /snap/clion/107/bin/cmake/linux/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING=Debug -D generated_file:STRING=/home/xinyu/Dropbox/GPU_computing/source/homework0/cmake-build-debug/CMakeFiles/hello.dir//./hello_generated_main.cu.o -D generated_cubin_file:STRING=/home/xinyu/Dropbox/GPU_computing/source/homework0/cmake-build-debug/CMakeFiles/hello.dir//./hello_generated_main.cu.o.cubin.txt -P /home/xinyu/Dropbox/GPU_computing/source/homework0/cmake-build-debug/CMakeFiles/hello.dir//hello_generated_main.cu.o.Debug.cmake

# Object files for target hello
hello_OBJECTS =

# External object files for target hello
hello_EXTERNAL_OBJECTS = \
"/home/xinyu/Dropbox/GPU_computing/source/homework0/cmake-build-debug/CMakeFiles/hello.dir/hello_generated_main.cu.o"

hello: CMakeFiles/hello.dir/hello_generated_main.cu.o
hello: CMakeFiles/hello.dir/build.make
hello: /usr/local/cuda/lib64/libcudart_static.a
hello: /usr/lib/x86_64-linux-gnu/librt.so
hello: CMakeFiles/hello.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/xinyu/Dropbox/GPU_computing/source/homework0/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable hello"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/hello.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/hello.dir/build: hello

.PHONY : CMakeFiles/hello.dir/build

CMakeFiles/hello.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/hello.dir/cmake_clean.cmake
.PHONY : CMakeFiles/hello.dir/clean

CMakeFiles/hello.dir/depend: CMakeFiles/hello.dir/hello_generated_main.cu.o
	cd /home/xinyu/Dropbox/GPU_computing/source/homework0/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/xinyu/Dropbox/GPU_computing/source/homework0 /home/xinyu/Dropbox/GPU_computing/source/homework0 /home/xinyu/Dropbox/GPU_computing/source/homework0/cmake-build-debug /home/xinyu/Dropbox/GPU_computing/source/homework0/cmake-build-debug /home/xinyu/Dropbox/GPU_computing/source/homework0/cmake-build-debug/CMakeFiles/hello.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/hello.dir/depend

