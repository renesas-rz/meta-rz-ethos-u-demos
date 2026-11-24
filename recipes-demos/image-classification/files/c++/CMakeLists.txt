cmake_minimum_required(VERSION 3.12)
set(CMAKE_CXX_STANDARD 17)
project(image-classification C CXX)

SET(EXE_NAME "image-classification")

file(GLOB SRC
	"main.cpp"
	"camera.cpp"
	"camera.h"
	"opencl_utils.cpp"
	"opencl_utils.h"
	"app_info.h"
	"demo.cpp"
	"demo.h"
	"graphstats.cpp"
	"graphstats.h"
	"memlog.cpp"
	"memlog.h"
	"timing.cpp"
	"timing.h"
	"processing.cpp"
	"processing.h"
	"wayland_utils.h"
	"wayland_utils.cpp"
	"${CMAKE_SYSROOT}/usr/include/wayland/xdg-shell-client-protocol.c"
)

add_executable(${EXE_NAME} ${SRC})

# Version and device name shown by the program (see app_info.h). The Yocto recipe passes -DAPP_VERSION=${PV}.
if(DEFINED APP_VERSION)
	target_compile_definitions(${EXE_NAME} PRIVATE APP_VERSION="${APP_VERSION}")
endif()

if(DEFINED BOARD_NAME)
	target_compile_definitions(${EXE_NAME} PRIVATE BOARD_NAME="${BOARD_NAME}")
endif()

find_package(Threads REQUIRED)

# OpenCV
find_package(PkgConfig REQUIRED)
pkg_check_modules(OPENCV REQUIRED opencv4)
include_directories(${OPENCV_INCLUDE_DIRS})
link_directories(${OPENCV_LIBRARY_DIRS})

# Link against TensorFlow Lite, OpenCV and wayland-client libraries
target_link_libraries(${EXE_NAME} tensorflow-lite ${OPENCV_LIBRARIES} wayland-client Threads::Threads)

install(TARGETS ${EXE_NAME} DESTINATION "bin")

# Documentation: this README. A missing file is reported, but does not stop the build.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/README_USB_CAMERA.md")
	install(FILES README_USB_CAMERA.md DESTINATION "share/doc/image-classification")
else()
	message(WARNING "README_USB_CAMERA.md not found next to CMakeLists.txt: it is not installed")
endif()

# Manual page (section 1). A missing file is reported, but does not stop the build.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/image-classification.1")
	install(FILES image-classification.1 DESTINATION "share/man/man1")
else()
	message(WARNING "image-classification.1 not found next to CMakeLists.txt: the manual page is not installed")
endif()
