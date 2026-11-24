/*
 * Copyright (C) 2026 Renesas Electronics Corp.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <cstdio>
#include <exception>
#include <string>

#include <opencv4/opencv2/core/ocl.hpp>
#include <opencv4/opencv2/imgproc/imgproc.hpp>

#include "opencl_utils.h"

using namespace std;

bool setup_opencl(bool enable) {
	try {
		if (!enable) {
			/*
			 * Does not initialise anything: the OpenCL library is not loaded and no GPU context is created.
			 * It also keeps OpenCV from probing the GPU by itself on the first resize or colour conversion of this thread.
			 */
			cv::ocl::setUseOpenCL(false);

			return false;
		}

		if (!cv::ocl::haveOpenCL()) {
			printf("OpenCL: not available in this OpenCV build or driver, the CPU is used\n");

			return false;
		}

		cv::ocl::Device device = cv::ocl::Device::getDefault();
		string name = device.name();
		string version = device.version();

		cv::ocl::setUseOpenCL(true);

		if (!cv::ocl::useOpenCL()) {
			printf("OpenCL: could not be enabled, the CPU is used\n");

			return false;
		}

		printf("OpenCL: using %s (%s) for resize and colour conversion. The first frame is slow while the kernels are compiled\n", name.c_str(), version.c_str());

		return true;
	} catch (const exception &e) {
		printf("OpenCL: not usable (%s), the CPU is used\n", e.what());

		return false;
	}
}

bool GpuPreprocessor::process(const cv::Mat &frame, const cv::Rect &crop, const cv::Size &size, cv::Mat &rgb_out) {
	try {
		/* Upload the whole (continuous) frame and crop on the GPU, which avoids a strided copy on the CPU */
		frame.copyTo(frame_);

		cv::resize(frame_(crop), small_, size, 0, 0, cv::INTER_LINEAR);
		cv::cvtColor(small_, rgb_, cv::COLOR_BGR2RGB);

		/* Download; this also waits until the GPU has finished */
		rgb_.copyTo(rgb_out);

		return true;
	} catch (const exception &e) {
		printf("OpenCL: failed (%s), switching back to the CPU\n", e.what());

		return false;
	}
}
