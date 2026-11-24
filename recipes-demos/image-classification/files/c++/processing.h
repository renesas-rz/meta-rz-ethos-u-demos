/*
 * Copyright (C) 2026 Renesas Electronics Corp.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#ifndef PROCESSING_H
#define PROCESSING_H

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <regex>

#include <getopt.h>
#include <opencv4/opencv2/core/core.hpp>
#include <opencv4/opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv4/opencv2/imgproc/imgproc.hpp>

#define LABEL_FILE_PATH "/usr/share/image-classification/labels/labels_mobilenet_v2_1.0_224.txt"
#define MODEL_INPUT_SIZE 224
#define TEXT_OFFSET 20

using namespace std;

const cv::Scalar WHITE_TEXT(255, 255, 255);
const cv::Scalar YELLOW_TEXT(0, 255, 255);

enum class ModelDataType {
	INT8,
	UINT8,
	FLOAT32
};

struct DisplayLayout {
	int width = 0;
	int height = 0;
	float font_size = 0.0f;
	float title_font = 0.0f;
	float heading_font = 0.0f;
	int outer_margin = 0;
	int column_gap = 0;
	int top_margin = 0;
	int image_panel_width = 0;
	int available_image_width = 0;
	int available_image_height = 0;
	int text_start_width = 0;
	int available_text_width = 0;
	int normal_line_height = 0;
	int section_gap = 0;
};

/*****************************************
* Functions
******************************************/
vector<string> load_labels_file();
cv::Mat load_image_file(string image_file_path);
/* Resizes and quantizes a copy of the image (RGB). The image of the caller is never changed. */
cv::Mat preprocess_image(cv::Mat image, int width, int height, ModelDataType data_type, float scale, double zero_point);
void post_process_output(int64_t output_size, const void* output_data_ptr, ModelDataType data_type, float scale, double zero_point, const vector<string>& labels, cv::Mat& image, float inference_time, int inference_runs, bool use_display);

cv::Rect get_center_crop(const cv::Mat& frame);
/* Draws only the four corners of a box (like a camera viewfinder). The stroke stays inside the box. */
void draw_corner_box(cv::Mat& image, const cv::Rect& box, const cv::Scalar& colour, int thickness);
/* Confidence of every class in percent (dequantized). Throws if the output size does not match the number of labels. */
vector<float> get_scores(int64_t output_size, const void* output_data_ptr, ModelDataType data_type, float scale, double zero_point, size_t label_count);
/* The 5 best classes as (percent, class index), best first */
/*
 * A class is listed from this confidence (in percent) on. The output of the model is an int8 softmax with steps of 0.39 percent, so everything
 * below 0.2 percent is stored as exactly 0, and the order among those zeros means nothing: listed anyway they would be arbitrary classes.
 * The steps around the limit are 0.78 percent (not listed) and 1.17 percent (listed).
 */
#define TOP_RESULTS_MINIMUM_PERCENT 1.0f

/*
 * The smallest confidence (in percent) at which a class is listed, as the option --min-percent sets it; TOP_RESULTS_MINIMUM_PERCENT until it is set.
 * 0 lists every class above 0 percent (a class at 0 is never listed).
 */
void set_min_percent(float percent);
float get_min_percent();

/* Reads the value of --min-percent: a number from 0 to 100 that is completely a number. False (value unchanged) if the text is anything else. */
bool parse_min_percent(const char* text, float& value);

/*
 * The best classes as (confidence, class number), the best first: up to 5, and only those with at least the minimum (get_min_percent(), or the
 * value that is given), so there may be fewer, or none. Equal confidences come with the higher class number first, as in the original code: with the
 * minimum 0 the list is exactly the one that the original code printed (the five best, without those at 0).
 */
vector<pair<float, int>> top_results_from_scores(const vector<float>& scores);
vector<pair<float, int>> top_results_from_scores(const vector<float>& scores, float minimum_percent);
/* Keeps only the first 3 comma separated items of long labels */
string shorten_label(const string& label);
vector<pair<float, int>> get_top_results(int64_t output_size, const void* output_data_ptr, ModelDataType data_type, float scale, double zero_point, size_t label_count);
bool get_display_layout(DisplayLayout& layout);
void render_static_window(const cv::Mat& image_rgb, const vector<pair<float, int>>& results, const vector<string>& labels, const DisplayLayout& layout, float inference_time, int inference_runs, const string& image_info, cv::Mat& display_window);
void render_live_window(const cv::Mat& frame_bgr, const vector<pair<float, int>>& results, const vector<string>& labels, const DisplayLayout& layout, float average_inference_time, int inference_samples, float fps, cv::Mat& display_window);
void print_live_results(const vector<pair<float, int>>& results, const vector<string>& labels, float average_inference_time, int inference_samples, float fps);

#endif
