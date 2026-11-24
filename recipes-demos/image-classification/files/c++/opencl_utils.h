/*
 * Copyright (C) 2026 Renesas Electronics Corp.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#ifndef OPENCL_UTILS_H
#define OPENCL_UTILS_H

#include <opencv4/opencv2/core/core.hpp>

/*
 * Without enable (no -g), OpenCL is only switched off for OpenCV in the calling thread: the OpenCL library is not loaded,
 * no GPU context is created, and nothing is printed. Call it before the first OpenCV call of the thread.
 * With enable, OpenCL is probed and, if it works, switched on for the GPU pre-processing below, and one line says which
 * device is used (or why it cannot be used).
 * Returns true only if enable is set and OpenCL can be used.
 */
bool setup_opencl(bool enable);

/* Crop, resize and colour conversion on the GPU (OpenCL, through cv::UMat) */
class GpuPreprocessor {
public:
	/*
	 * Crops the square from the BGR frame, resizes it to size with INTER_LINEAR, converts it to RGB
	 * and copies the result to rgb_out (CPU memory). Returns false if OpenCL failed; rgb_out is then unchanged.
	 */
	bool process(const cv::Mat &frame, const cv::Rect &crop, const cv::Size &size, cv::Mat &rgb_out);

private:
	cv::UMat frame_;
	cv::UMat small_;
	cv::UMat rgb_;
};

#endif
