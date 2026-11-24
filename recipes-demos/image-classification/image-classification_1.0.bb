SUMMARY = "Ethos-U Image Classification Sample Application"
DESCRIPTION = "Image Classification application that runs inference on an image file \
using MobileNet V2. The application uses TensorFlow Lite with the Ethos-U delegate to run \
inference on the Arm(R) Ethos(TM)-U55. It supports single picture, slideshow \
and live USB camera modes, with the result shown on the console or on a \
Wayland display."

LICENSE = "Apache-2.0"
LIC_FILES_CHKSUM = "file://${COMMON_LICENSE_DIR}/Apache-2.0;md5=89aea4e17d99a7cacdbeed46a0096b10"

DEPENDS = " \
	ethos-u-linux-driver-stack \
	opencv \
	wayland \
	xdg-shell-client-protocol \
"

SRC_URI = "file://c++/"

S = "${WORKDIR}/c++"

inherit cmake pkgconfig

EXTRA_OECMAKE += "-DAPP_VERSION=${PV}"

COMPATIBLE_MACHINE = "smarc-rzg3e"
PACKAGE_ARCH = "${MACHINE_ARCH}"

RDEPENDS:${PN} = " \
	ethos-u-linux-driver-stack \
	libopencv-core \
	libopencv-imgcodecs \
	libopencv-imgproc \
	libopencv-videoio \
	ml-sample-images \
	mobilenet-v2 \
"

# Needed at run time by the C++ application for the camera pipeline
# (v4l2src ! videoconvert ! appsink, with a MJPEG fallback that uses
# jpegdec), but not found by the automatic library dependencies because
# the GStreamer plugins are loaded by OpenCV at run time.
RDEPENDS:${PN} += " \
	gstreamer1.0-plugins-base-app \
	gstreamer1.0-plugins-base-videoconvertscale \
	gstreamer1.0-plugins-good-jpeg \
	gstreamer1.0-plugins-good-video4linux2 \
"
