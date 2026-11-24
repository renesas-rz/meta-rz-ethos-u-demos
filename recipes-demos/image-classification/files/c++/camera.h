/*
 * Copyright (C) 2026 Renesas Electronics Corp.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#ifndef CAMERA_H
#define CAMERA_H

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>

#include <opencv4/opencv2/core/core.hpp>
#include <opencv4/opencv2/videoio.hpp>

#define CAMERA_WIDTH 640
#define CAMERA_HEIGHT 480
/* Frame rate the camera is expected to deliver at this size (C922 with dynamic frame rate disabled: 30 fps) */
#define CAMERA_FRAME_RATE 30
#define CAMERA_MAX_VIDEO_NODES 64
/* UsbCamera::read() waits at most this long for a new frame */
#define CAMERA_READ_TIMEOUT_MS 1000
/*
 * Failed reads in a row after which the camera counts as lost. Used at two levels:
 * the capture thread counts failed reads from the camera, and the main loop counts failed calls of UsbCamera::read(),
 * each of which waits up to CAMERA_READ_TIMEOUT_MS, so a camera that stops delivering frames is reported after
 * about CAMERA_MAX_READ_FAILURES x CAMERA_READ_TIMEOUT_MS (5 seconds).
 */
#define CAMERA_MAX_READ_FAILURES 5

using namespace std;

/*****************************************
* Functions
******************************************/

/*
 * Scans /dev/video* for a USB video capture device.
 * Returns the device path, or an empty string if none is found.
 */
string find_usb_camera();

/*
 * Checks a camera device given by the user, because it is put into a GStreamer pipeline text.
 * Accepted: a path that starts with /dev/, consists only of letters, digits and / _ . : -
 * (no spaces, quotes or other characters, nothing in front of or after it, no ".."),
 * and that exists as a character device. Otherwise false, and reason says why.
 */
bool is_valid_camera_device(const string &device, string &reason);

/*
 * A capture thread reads the camera continuously and keeps only the newest frame,
 * so a slow consumer always gets a fresh frame instead of an old queued one.
 */
class UsbCamera {
public:
	UsbCamera() = default;
	~UsbCamera();
	UsbCamera(const UsbCamera&) = delete;
	UsbCamera& operator=(const UsbCamera&) = delete;

	/*
	 * Tries several GStreamer pipelines, then the OpenCV V4L2 backend, then starts the capture thread. Returns false if all fail.
	 * Unless keep_dynamic_framerate is set, the camera's dynamic frame rate (if it has one) is disabled to keep the frame rate constant.
	 */
	bool open(const string &device, bool keep_dynamic_framerate = false);

	/*
	 * Copies the newest frame (BGR) that has not been returned yet. Waits up to CAMERA_READ_TIMEOUT_MS for one.
	 * Returns false on timeout or if the camera stopped delivering frames.
	 * If captured is given, it receives the time the capture thread got the frame from the camera pipeline.
	 */
	bool read(cv::Mat &frame, chrono::steady_clock::time_point *captured = nullptr);

	/* Stops the capture thread and closes the camera */
	void release();

	/* Frame rate the camera really delivers, measured by the capture thread over the last second. 0 until measured. */
	double capture_fps() const;

private:
	bool open_source(const string &device);
	bool try_open(const string &source, int api);
	void capture_loop();

	cv::VideoCapture capture_;
	thread thread_;
	mutex mutex_;
	condition_variable cond_;
	cv::Mat latest_;
	chrono::steady_clock::time_point latest_time_;
	uint64_t frame_counter_ = 0;
	uint64_t consumed_counter_ = 0;
	bool failed_ = false;
	atomic<bool> stop_{false};
	atomic<double> capture_fps_{0.0};
};

#endif
