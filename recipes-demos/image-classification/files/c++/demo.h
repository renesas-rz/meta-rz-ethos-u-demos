/*
 * Copyright (C) 2026 Renesas Electronics Corp.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#ifndef DEMO_H
#define DEMO_H

#include <deque>
#include <string>
#include <vector>

#include <opencv4/opencv2/core/core.hpp>

#include "graphstats.h"
#include "processing.h"

/* Weight of the newest frame when the class scores are averaged over time (about the last 5 frames) */
#define DEMO_SMOOTHING_ALPHA 0.35f
/* A best match below this confidence (in percent) is shown greyed out and marked "uncertain" */
#define DEMO_UNCERTAIN_PERCENT 20.0f
/* A best match at or above this confidence (in percent) is shown in green, between the two limits in amber */
#define DEMO_CONFIDENT_PERCENT 51.0f
/* With the camera, green is kept until the confidence falls this far below the limit, so the colour does not flicker */
#define DEMO_CONFIDENT_MARGIN 5.0f
/* Inference times that are kept: the histogram, the caption and the range of the graph use all of them (about 10 seconds) */
#define DEMO_HISTORY_FRAMES 300
/* The line of the graph shows the newest frames only (about 3 seconds) */
#define DEMO_LINE_FRAMES 100
/* The histogram next to the line is drawn if the space for the graph is at least this wide (displays about 950 pixels wide and more) */
#define DEMO_HISTOGRAM_MIN_WIDTH 330
/* How fast the vertical range of the graph narrows after a slow frame has left the window (part of the gap per refresh; 0.02 = half after 3.5 s) */
#define DEMO_SCALE_DECAY 0.02f
/* With partial redraw the text and graph are redrawn only every this many frames (about 10 times per second at 30 FPS) */
#define DEMO_TEXT_PERIOD 3

using namespace std;

/* What differs between the camera, the slideshow and a single picture */
struct DemoInfo {
	/* Right side of the header */
	string title = "Image Classification";
	/* The picture holds R,G,B (loaded from file) instead of B,G,R (camera) */
	bool frame_is_rgb = false;
	/* Outline the centre square the model looks at (camera only; pictures are used whole) */
	bool show_crop_box = false;
	/* Average the class scores over the last frames (camera only; pictures are independent) */
	bool smooth_scores = false;
	/* Keep the static parts and redraw the picture every frame but the text only every DEMO_TEXT_PERIOD frames (camera only) */
	bool partial_redraw = false;

	float inference_time = 0.0f;
	/* False for a frame whose inference time must not enter the graph, the histogram and the caption (the warm-up frame) */
	bool has_inference_sample = true;
	float average_inference_time = 0.0f;
	int inference_samples = 0;
	string time_caption = "NPU inference (avg)";

	/* Second headline number: FPS for the camera, inferences per second for pictures. 0 or less shows "--" */
	float rate = 0.0f;
	string rate_unit = "FPS";
	string rate_caption = "end-to-end";

	/* Bottom right: graph of the inference time, or (without graph) a text such as "3/12" and the file name */
	bool show_graph = true;
	string side_value;
	string side_caption;
};

/*
 * Eye-catching view: Renesas header, large picture, large best match with a confidence bar,
 * four smaller matches, headline numbers and a graph of the inference time.
 */
class DemoView {
public:
	/*
	 * scores: confidence of every class in percent (before smoothing).
	 * Draws into display_window (CV_8UC4, B,G,R,X byte order), which is allocated on first use.
	 */
	void render(const cv::Mat& frame,
	            const vector<float>& scores,
	            const vector<string>& labels,
	            const DisplayLayout& layout,
	            const DemoInfo& info,
	            cv::Mat& display_window);

private:
	vector<float> smoothed_;
	deque<float> history_;
	GraphScale scale_;
	unsigned int frame_count_ = 0;
	bool confident_ = false;
};

#endif
