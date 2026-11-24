/*
 * Copyright (C) 2026 Renesas Electronics Corp.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

#include <fcntl.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <opencv4/opencv2/core/ocl.hpp>

#include "camera.h"

#define CAMERA_WARMUP_READS 5

/* True if the node is a USB video capture node (skips metadata, codec and non-USB nodes) */
static bool is_usb_capture_device(const string &path) {
	struct v4l2_capability cap;
	uint32_t caps;
	bool is_capture;
	bool is_usb;

	int fd = open(path.c_str(), O_RDONLY | O_NONBLOCK);

	if (fd < 0)
		return false;

	memset(&cap, 0, sizeof(cap));

	if (ioctl(fd, VIDIOC_QUERYCAP, &cap) < 0) {
		close(fd);

		return false;
	}

	close(fd);

	caps = (cap.capabilities & V4L2_CAP_DEVICE_CAPS) ? cap.device_caps : cap.capabilities;
	is_capture = (caps & (V4L2_CAP_VIDEO_CAPTURE | V4L2_CAP_VIDEO_CAPTURE_MPLANE)) != 0;
	is_usb = strncmp(reinterpret_cast<const char *>(cap.bus_info), "usb-", 4) == 0;

	return is_capture && is_usb;
}

/*
 * Disables "exposure_dynamic_framerate" (V4L2_CID_EXPOSURE_AUTO_PRIORITY).
 * With it enabled, many UVC cameras lower the frame rate in dim light (e.g. 30 -> 15 fps).
 * Best effort: does nothing if the camera has no such control, and the camera keeps working if it cannot be changed.
 */
static void disable_dynamic_framerate(const string &device) {
	struct v4l2_queryctrl query;
	struct v4l2_control ctrl;

	int fd = open(device.c_str(), O_RDWR | O_NONBLOCK);

	if (fd < 0) {
		printf("Could not open %s to set exposure_dynamic_framerate: %s\n", device.c_str(), strerror(errno));

		return;
	}

	memset(&query, 0, sizeof(query));
	query.id = V4L2_CID_EXPOSURE_AUTO_PRIORITY;

	if (ioctl(fd, VIDIOC_QUERYCTRL, &query) < 0 || (query.flags & V4L2_CTRL_FLAG_DISABLED)) {
		printf("Camera has no exposure_dynamic_framerate control, nothing to change\n");
		close(fd);

		return;
	}

	memset(&ctrl, 0, sizeof(ctrl));
	ctrl.id = V4L2_CID_EXPOSURE_AUTO_PRIORITY;

	if (ioctl(fd, VIDIOC_G_CTRL, &ctrl) == 0 && ctrl.value == 0) {
		printf("Camera: exposure_dynamic_framerate is already disabled\n");
		close(fd);

		return;
	}

	ctrl.value = 0;

	if (ioctl(fd, VIDIOC_S_CTRL, &ctrl) < 0)
		printf("Camera: exposure_dynamic_framerate not changed (%s), the frame rate may drop in low light\n", strerror(errno));
	else
		printf("Camera: exposure_dynamic_framerate disabled to keep the frame rate constant\n");

	close(fd);
}

string find_usb_camera() {
	for (int i = 0; i < CAMERA_MAX_VIDEO_NODES; i++) {
		struct stat st;
		string path = "/dev/video" + to_string(i);

		if (stat(path.c_str(), &st) != 0)
			continue;

		if (is_usb_capture_device(path))
			return path;
	}

	return "";
}

bool is_valid_camera_device(const string &device, string &reason) {
	struct stat st;

	if (device.empty()) {
		reason = "the name is empty";

		return false;
	}

	if (device.size() > 128) {
		reason = "the name is too long";

		return false;
	}

	for (char c : device) {
		bool allowed = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
		               c == '/' || c == '_' || c == '.' || c == '-' || c == ':';

		if (!allowed) {
			reason = "only letters, digits and / _ . : - are allowed (no spaces or other characters, also not in front of or after the name)";

			return false;
		}
	}

	if (device.compare(0, 5, "/dev/") != 0 || device.find("..") != string::npos) {
		reason = "it must be a device name below /dev/, for example /dev/video0";

		return false;
	}

	if (stat(device.c_str(), &st) != 0) {
		reason = "the device does not exist";

		return false;
	}

	if (!S_ISCHR(st.st_mode)) {
		reason = "it is not a device";

		return false;
	}

	return true;
}

bool UsbCamera::try_open(const string &source, int api) {
	capture_.release();

	if (!capture_.open(source, api))
		return false;

	if (api == cv::CAP_V4L2) {
		capture_.set(cv::CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH);
		capture_.set(cv::CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT);
	}

	/* Opening can succeed while the format negotiation fails, so only accept the source once a frame arrives */
	cv::Mat frame;

	for (int i = 0; i < CAMERA_WARMUP_READS; i++) {
		if (capture_.read(frame) && !frame.empty())
			return true;
	}

	capture_.release();

	return false;
}

bool UsbCamera::open_source(const string &device) {
	const string size_caps = ",width=" + to_string(CAMERA_WIDTH) + ",height=" + to_string(CAMERA_HEIGHT);
	/* The device was checked in open(); the quotes are an extra protection */
	const string source = "v4l2src device=\"" + device + "\"";
	const string sink = " ! videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1 sync=false";

	/* Pipelines in order of preference: raw at fixed size, MJPEG at fixed size, camera default format */
	const vector<string> pipelines = {
		source + " ! video/x-raw" + size_caps + sink,
		source + " ! image/jpeg" + size_caps + " ! jpegdec" + sink,
		source + sink
	};

	for (const string &pipeline : pipelines) {
		if (try_open(pipeline, cv::CAP_GSTREAMER)) {
			printf("Camera opened with GStreamer pipeline: %s\n", pipeline.c_str());

			return true;
		}
	}

	if (try_open(device, cv::CAP_V4L2)) {
		printf("Camera opened with OpenCV V4L2 backend: %s\n", device.c_str());

		return true;
	}

	return false;
}

bool UsbCamera::open(const string &device, bool keep_dynamic_framerate) {
	string reason;

	release();

	if (!is_valid_camera_device(device, reason)) {
		printf("Invalid camera device '%s': %s\n", device.c_str(), reason.c_str());

		return false;
	}

	if (!keep_dynamic_framerate)
		disable_dynamic_framerate(device);

	if (!open_source(device))
		return false;

	thread_ = thread(&UsbCamera::capture_loop, this);

	return true;
}

void UsbCamera::capture_loop() {
	/* Only the main thread uses the GPU (-g). This keeps OpenCV from probing OpenCL in the capture thread. */
	cv::ocl::setUseOpenCL(false);

	cv::Mat frame;
	int failures = 0;
	unsigned int window_frames = 0;
	auto window_start = chrono::steady_clock::now();

	while (!stop_) {
		if (!capture_.read(frame) || frame.empty()) {
			if (++failures >= CAMERA_MAX_READ_FAILURES)
				break;

			continue;
		}

		failures = 0;

		auto captured = chrono::steady_clock::now();

		/* Measure the real frame rate of the camera */
		window_frames++;

		chrono::duration<double> window_time = captured - window_start;

		if (window_time.count() >= 1.0) {
			capture_fps_ = window_frames / window_time.count();
			window_frames = 0;
			window_start = captured;
		}

		{
			lock_guard<mutex> lock(mutex_);
			latest_time_ = captured;

			/* Hand the new frame over and reuse the old buffer for the next capture */
			cv::swap(latest_, frame);
			frame_counter_++;
		}

		cond_.notify_one();
	}

	{
		lock_guard<mutex> lock(mutex_);
		failed_ = true;
	}

	cond_.notify_all();
}

bool UsbCamera::read(cv::Mat &frame, chrono::steady_clock::time_point *captured) {
	unique_lock<mutex> lock(mutex_);

	cond_.wait_for(lock, chrono::milliseconds(CAMERA_READ_TIMEOUT_MS), [this] {
		return frame_counter_ != consumed_counter_ || failed_;
	});

	if (frame_counter_ == consumed_counter_)
		return false;

	latest_.copyTo(frame);
	consumed_counter_ = frame_counter_;

	if (captured)
		*captured = latest_time_;

	return true;
}

void UsbCamera::release() {
	stop_ = true;

	if (thread_.joinable())
		thread_.join();

	capture_.release();

	lock_guard<mutex> lock(mutex_);

	latest_.release();
	capture_fps_ = 0.0;
	frame_counter_ = 0;
	consumed_counter_ = 0;
	failed_ = false;
	stop_ = false;
}

double UsbCamera::capture_fps() const {
	return capture_fps_;
}

UsbCamera::~UsbCamera() {
	release();
}
