/*
 * Copyright (C) 2026 Renesas Electronics Corp.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <cerrno>
#include <cmath>
#include <functional>

#include "processing.h"
#include "wayland_utils.h"

vector<string> load_labels_file() {
	string labels_file_path = LABEL_FILE_PATH;
	vector<string> list = {};
	vector<string> empty = {};
	string line = "";

	ifstream labels_file(labels_file_path);

	if (!labels_file.is_open())
		return empty;

	/* Read labels from file and store into vector */
	while (getline(labels_file, line)) {
		list.push_back(line);

		if (labels_file.fail())
			return empty;
	}

	labels_file.close();

	return list;
}

cv::Mat load_image_file(string image_file_path) {
	cv::Mat image;

	/* Read image and convert to RGB */
	image = cv::imread(image_file_path);

	if (image.empty())
		return image;

	cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

	return image;
}

cv::Mat preprocess_image(cv::Mat image, int width, int height, ModelDataType data_type, float scale, double zero_point) {
	int interpolation_method = 0;

	/* Resize image only if needed */
	if (image.size() != cv::Size(width, height)) {
		if (image.cols > width || image.rows > height)
			interpolation_method = cv::INTER_AREA;
		else
			interpolation_method = cv::INTER_LINEAR;

		cv::resize(image, image, cv::Size(width, height), 0, 0, interpolation_method);
	}

	/* Float models do not require normalization */
	if (data_type == ModelDataType::FLOAT32) {
		cv::Mat converted;

		image.convertTo(converted, CV_32F, 2.0 / 255.0, -1.0);

		return converted;
	}

	bool is_int8 = (data_type == ModelDataType::INT8);
	double min_value = is_int8 ? -128.0 : 0.0;
	double max_value = is_int8 ? 127.0 : 255.0;

	/* Calculate normalization values*/
	double real_min = (min_value - zero_point) * (double)scale;
	double norm_low = (real_min < 0) ? -1.0 : 0.0;
	double norm_range = 1.0 - norm_low;

	/* Normalize the pixels and quantize using models scale and zero_point */
	cv::Mat lut(1, 256, is_int8 ? CV_8SC1 : CV_8UC1);

	for (int pixel = 0; pixel < 256; pixel++) {
		double value = nearbyint((((pixel / 255.0) * norm_range + norm_low) / (double)scale) + zero_point);

		/* Clip to prevent values from not fitting boundary */
		value = min(max(value, min_value), max_value);

		if (is_int8)
			lut.at<int8_t>(pixel) = static_cast<int8_t>(value);
		else
			lut.at<uint8_t>(pixel) = static_cast<uint8_t>(value);
	}

	/*
	 * The result gets its own buffer. "image" is a copy of the caller's header and shares its pixels as long as no
	 * resize happened. For a UINT8 model the LUT keeps the type of the source, so writing the result into "image"
	 * would change the picture of the caller (which shows it after this call). INT8 and FLOAT32 change the type and
	 * would get a new buffer anyway; this makes all three the same and independent of that detail.
	 */
	cv::Mat quantized;

	cv::LUT(image, lut, quantized);

	return quantized;
}

void draw_text(cv::Mat& image_to_draw_on, string text_to_write, int text_start_width, int text_start_height, float font_size, const cv::Scalar& text_colour) {
	cv::putText(image_to_draw_on, text_to_write, cv::Point(text_start_width, text_start_height), cv::FONT_HERSHEY_DUPLEX, font_size, text_colour, 1);
}


/*
 * Trim text so that it fits within the available horizontal space.
 */
static string fit_text_to_width(const string& text,
								int maximum_width,
								double font_scale) {
	if (text.empty() || maximum_width <= 0)
		return "";

	int baseline = 0;

	cv::Size size =
		cv::getTextSize(text,
						cv::FONT_HERSHEY_DUPLEX,
						font_scale,
						1,
						&baseline);

	if (size.width <= maximum_width)
		return text;

	const string ellipsis = "...";
	string fitted = text;

	while (!fitted.empty()) {

		fitted.pop_back();

		string candidate = fitted + ellipsis;

		size =
			cv::getTextSize(candidate,
							cv::FONT_HERSHEY_DUPLEX,
							font_scale,
							1,
							&baseline);

		if (size.width <= maximum_width)
			return candidate;
	}

	return ellipsis;
}


cv::Rect get_center_crop(const cv::Mat& frame) {
	int side = min(frame.cols, frame.rows);

	return cv::Rect((frame.cols - side) / 2, (frame.rows - side) / 2, side, side);
}

void draw_corner_box(cv::Mat& image, const cv::Rect& box, const cv::Scalar& colour, int thickness) {
	/* Each corner is about 12% of the shorter side long, but at least 14 pixels */
	const int length = std::max(14, static_cast<int>(std::min(box.width, box.height) * 0.12));
	const int half = thickness / 2;
	const int left = box.x + half;
	const int top = box.y + half;
	const int right = box.x + box.width - 1 - half;
	const int bottom = box.y + box.height - 1 - half;

	cv::line(image, cv::Point(left, top), cv::Point(left + length, top), colour, thickness);
	cv::line(image, cv::Point(left, top), cv::Point(left, top + length), colour, thickness);

	cv::line(image, cv::Point(right, top), cv::Point(right - length, top), colour, thickness);
	cv::line(image, cv::Point(right, top), cv::Point(right, top + length), colour, thickness);

	cv::line(image, cv::Point(left, bottom), cv::Point(left + length, bottom), colour, thickness);
	cv::line(image, cv::Point(left, bottom), cv::Point(left, bottom - length), colour, thickness);

	cv::line(image, cv::Point(right, bottom), cv::Point(right - length, bottom), colour, thickness);
	cv::line(image, cv::Point(right, bottom), cv::Point(right, bottom - length), colour, thickness);
}

/* Keeps only the first 3 comma separated items of long labels */
string shorten_label(const string& label) {
	static const regex label_regex("([^,]*,[^,]*,[^,]*)");
	smatch match;

	if (regex_search(label, match, label_regex))
		return string(match[1]);

	return label;
}

vector<float> get_scores(int64_t output_size,
						 const void* output_data_ptr,
						 ModelDataType data_type,
						 float scale,
						 double zero_point,
						 size_t label_count) {
	vector<float> scores;

	if (output_size < 5 || (size_t)output_size > label_count) {
		throw runtime_error(
			"Output size (" + to_string(output_size) +
			") does not match number of labels (" + to_string(label_count) + ")");
	}

	scores.reserve(output_size);

	/* Calculate confidence for each class in percentages, dequantizing with the output scale and zero point */
	for (int64_t i = 0; i < output_size; i++) {
		float value;

		if (data_type == ModelDataType::INT8)
			value = (static_cast<const int8_t*>(output_data_ptr)[i] - zero_point) * scale;
		else if (data_type == ModelDataType::UINT8)
			value = (static_cast<const uint8_t*>(output_data_ptr)[i] - zero_point) * scale;
		else
			value = static_cast<const float*>(output_data_ptr)[i];

		scores.push_back(value * 100);
	}

	return scores;
}

static float min_percent = TOP_RESULTS_MINIMUM_PERCENT;

void set_min_percent(float percent) {
	min_percent = percent;
}

float get_min_percent() {
	return min_percent;
}

bool parse_min_percent(const char* text, float& value) {
	char* end = nullptr;

	if (!text || !*text)
		return false;

	errno = 0;

	const double number = strtod(text, &end);

	/* The whole text must be the number, and it must be a percentage ("nan" and "inf" are numbers for strtod, but not here) */
	if (errno != 0 || end == text || *end != '\0' || !isfinite(number) || number < 0.0 || number > 100.0)
		return false;

	value = (float)number;

	return true;
}

vector<pair<float, int>> top_results_from_scores(const vector<float>& scores) {
	return top_results_from_scores(scores, min_percent);
}

vector<pair<float, int>> top_results_from_scores(const vector<float>& scores, float minimum_percent) {
	vector<pair<float, int>> results;

	results.reserve(scores.size());

	/* Only classes with a meaningful confidence are listed (see TOP_RESULTS_MINIMUM_PERCENT) */
	for (size_t i = 0; i < scores.size(); i++) {
		if (scores[i] > 0.0f && scores[i] >= minimum_percent)
			results.push_back(make_pair(scores[i], (int)i));
	}

	/*
	 * Speed up sorting by only sorting top 5 percentages. The order is that of the original code (std::greater on the pair): equal percentages come
	 * with the higher class number first. With the minimum 0 the list is therefore exactly the one that the original code prints.
	 */
	const size_t count = min<size_t>(5, results.size());

	partial_sort(results.begin(), results.begin() + count, results.end(), greater{});
	results.resize(count);

	return results;
}

vector<pair<float, int>> get_top_results(int64_t output_size,
										 const void* output_data_ptr,
										 ModelDataType data_type,
										 float scale,
										 double zero_point,
										 size_t label_count) {
	return top_results_from_scores(get_scores(output_size, output_data_ptr, data_type, scale, zero_point, label_count));
}

bool get_display_layout(DisplayLayout& layout) {
	constexpr int MIN_DISPLAY_WIDTH  = 800;
	constexpr int MAX_DISPLAY_WIDTH  = 1920;
	constexpr int MIN_DISPLAY_HEIGHT = 600;
	constexpr int MAX_DISPLAY_HEIGHT = 1080;

	pair<int, int> display_dims = get_display_dims();

	layout = DisplayLayout();
	layout.width  = display_dims.first;
	layout.height = display_dims.second;

	if (layout.width <= 0 || layout.height <= 0)
		return false;

	if (layout.width  < MIN_DISPLAY_WIDTH  || layout.width  > MAX_DISPLAY_WIDTH ||
		layout.height < MIN_DISPLAY_HEIGHT || layout.height > MAX_DISPLAY_HEIGHT) {

		printf("Unsupported display resolution: %dx%d\n", layout.width, layout.height);

		return false;
	}

	/*
	 * Scale text primarily according to display height.
	 *
	 * 600 px high  -> 0.55
	 * 720 px high  -> ~0.66
	 * 1080 px high -> capped at 0.90
	 */
	const double height_scale = static_cast<double>(layout.height) / 600.0;

	layout.font_size = static_cast<float>(std::clamp(0.55 * height_scale, 0.55, 0.90));

	printf("Display: %dx%d, font: %.2f\n", layout.width, layout.height, layout.font_size);

	/* Adaptive margins */
	layout.outer_margin = std::max(12, layout.width / 80);
	layout.column_gap   = std::max(16, layout.width / 50);
	layout.top_margin   = std::max(24, layout.height / 25);

	/* Approximately 40% of the screen is allocated to the image */
	layout.image_panel_width      = static_cast<int>(layout.width * 0.40);
	layout.available_image_width  = layout.image_panel_width - (2 * layout.outer_margin);
	layout.available_image_height = layout.height - (2 * layout.outer_margin);

	/* Text starts independently of the actual resized image width */
	layout.text_start_width     = layout.image_panel_width + layout.column_gap;
	layout.available_text_width = layout.width - layout.text_start_width - layout.outer_margin;

	/* Adaptive vertical spacing */
	layout.normal_line_height = std::max(26, static_cast<int>(32.0 * (layout.font_size / 0.55f)));
	layout.section_gap        = std::max(16, layout.normal_line_height / 2);
	layout.title_font         = std::min(layout.font_size + 0.10f, 1.00f);
	layout.heading_font       = std::min(layout.font_size + 0.05f, 0.95f);

	return true;
}

/* Allocates the display window once (4 channels, B,G,R,X byte order as the wayland XRGB8888 buffer expects) and clears it */
static void clear_display_window(const DisplayLayout& layout, cv::Mat& display_window) {
	if (display_window.empty())
		display_window.create(layout.height, layout.width, CV_8UC4);

	display_window.setTo(cv::Scalar(0, 0, 0, 0));
}

/*
 * Scales the image into the image panel (keeping the aspect ratio), centres it and copies it into the display window.
 * image_is_rgb: the image holds R,G,B (as loaded from file), otherwise B,G,R (as delivered by the camera).
 * Returns the area the image occupies in the display window.
 */
static cv::Rect place_image(const cv::Mat& image, bool image_is_rgb, const DisplayLayout& layout, cv::Mat& display_window) {
	cv::Mat resized;
	cv::Mat resized_bgra;

	const double width_scale  = static_cast<double>(layout.available_image_width)  / static_cast<double>(image.cols);
	const double height_scale = static_cast<double>(layout.available_image_height) / static_cast<double>(image.rows);
	const double image_scale_factor = std::min(width_scale, height_scale);

	const int resized_width  = std::max(1, static_cast<int>(std::round(image.cols * image_scale_factor)));
	const int resized_height = std::max(1, static_cast<int>(std::round(image.rows * image_scale_factor)));

	const int interpolation_method = image_scale_factor > 1.0 ? cv::INTER_LINEAR : cv::INTER_AREA;

	cv::resize(image, resized, cv::Size(resized_width, resized_height), 0, 0, interpolation_method);
	cv::cvtColor(resized, resized_bgra, image_is_rgb ? cv::COLOR_RGB2BGRA : cv::COLOR_BGR2BGRA);

	/* Centre the image within its panel */
	const int image_x = layout.outer_margin + std::max(0, (layout.available_image_width  - resized_bgra.cols) / 2);
	const int image_y = layout.outer_margin + std::max(0, (layout.available_image_height - resized_bgra.rows) / 2);

	cv::Rect roi(image_x, image_y, resized_bgra.cols, resized_bgra.rows);

	resized_bgra.copyTo(display_window(roi));

	return roi;
}

/*
 * Draws the information text with a running Y position. Each timing line is drawn on its own row.
 * Returns the Y position where the classification results start.
 */
static int draw_info_panel(cv::Mat& display_window, const DisplayLayout& layout, const vector<string>& timing_lines) {
	const int x = layout.text_start_width;
	int y = layout.top_margin;

	draw_text(display_window,
			  fit_text_to_width("Renesas Image Classification Application", layout.available_text_width, layout.title_font),
			  x, y, layout.title_font, YELLOW_TEXT);

	y += layout.normal_line_height + layout.section_gap;

	draw_text(display_window,
			  fit_text_to_width("Model: MobileNet V2 (224x224)", layout.available_text_width, layout.font_size),
			  x, y, layout.font_size, WHITE_TEXT);

	y += layout.normal_line_height;

	draw_text(display_window,
			  fit_text_to_width("Accelerator: Ethos-U55 NPU", layout.available_text_width, layout.font_size),
			  x, y, layout.font_size, WHITE_TEXT);

	y += layout.normal_line_height + layout.section_gap;

	draw_text(display_window, "Model Information", x, y, layout.heading_font, YELLOW_TEXT);

	y += layout.normal_line_height;

	draw_text(display_window, "NPU Operators: 95", x, y, layout.font_size, WHITE_TEXT);

	y += layout.normal_line_height;

	draw_text(display_window, "CPU Operators: 0", x, y, layout.font_size, WHITE_TEXT);

	y += layout.normal_line_height + layout.section_gap;

	for (const string& line : timing_lines) {
		draw_text(display_window,
				  fit_text_to_width(line, layout.available_text_width, layout.font_size),
				  x, y, layout.font_size, WHITE_TEXT);

		y += layout.normal_line_height;
	}

	y += layout.section_gap;

	draw_text(display_window, "Identified Classifications:", x, y, layout.heading_font, YELLOW_TEXT);

	y += layout.normal_line_height;

	return y;
}

/* Draws the top 5 results on the display window, or prints them to the terminal when display_window is null */
static void draw_top_results(cv::Mat* display_window,
							 const vector<pair<float, int>>& results,
							 const vector<string>& labels,
							 const DisplayLayout& layout,
							 int classification_start_y) {
	/* If no class reaches the minimum (--min-percent), alert the user via display or terminal */
	if (results.empty()) {
		if (display_window)
			draw_text(*display_window, "No classifications identified!", layout.text_start_width, classification_start_y, layout.font_size, WHITE_TEXT);
		else
			printf("\nNo classifications identified!\n");

		return;
	}

	for (size_t i = 0; i < results.size(); i++) {
		string label = shorten_label(labels[results[i].second]);
		float confidence = results[i].first;
		stringstream confidence_text;

		confidence_text << fixed << setprecision(2) << confidence;

		if (display_window) {
			string result_text = fit_text_to_width(confidence_text.str() + "% -> " + label,
												   layout.available_text_width,
												   layout.font_size);

			draw_text(*display_window,
					  result_text,
					  layout.text_start_width,
					  classification_start_y + (i * layout.normal_line_height),
					  layout.font_size,
					  WHITE_TEXT);
		} else {
			printf("\n%.2f%% -> %s\n", confidence, label.c_str());
		}
	}
}

void render_static_window(const cv::Mat& image_rgb,
						  const vector<pair<float, int>>& results,
						  const vector<string>& labels,
						  const DisplayLayout& layout,
						  float inference_time,
						  int inference_runs,
						  const string& image_info,
						  cv::Mat& display_window) {
	stringstream inference_time_text;
	string inference_runs_text;

	inference_time_text << fixed << setprecision(2) << inference_time;

	if (inference_runs == 1)
		inference_runs_text = " run";
	else
		inference_runs_text = " runs";

	clear_display_window(layout, display_window);

	/* The image from file is stored as R,G,B */
	place_image(image_rgb, true, layout, display_window);

	vector<string> timing_lines = {"Average Inference Time: " + inference_time_text.str() + "ms over " + to_string(inference_runs) + inference_runs_text};

	if (!image_info.empty())
		timing_lines.push_back(image_info);

	int classification_start_y = draw_info_panel(display_window, layout, timing_lines);

	draw_top_results(&display_window, results, labels, layout, classification_start_y);
}

void post_process_output(int64_t output_size,
						 const void* output_data_ptr,
						 ModelDataType data_type,
						 float scale,
						 double zero_point,
						 const vector<string>& labels,
						 cv::Mat& image,
						 float inference_time,
						 int inference_runs,
						 bool use_display) {
	DisplayLayout layout;
	cv::Mat display_window;

	vector<pair<float, int>> results = get_top_results(output_size, output_data_ptr, data_type, scale, zero_point, labels.size());

	/* Switch to not using display if no usable display is detected */
	if (use_display)
		use_display = get_display_layout(layout);

	if (use_display) {
		printf("\nDisplaying top Image Classification results....\n");

		render_static_window(image, results, labels, layout, inference_time, inference_runs, "", display_window);

		/* Show completed display window on HDMI display */
		display_results(layout.width, layout.height, display_window);
	} else {
		printf("\nPrinting top Image Classification results....\n");

		draw_top_results(nullptr, results, labels, layout, 0);
	}
}

/* "Average Inference Time: 9.28ms over 10 frames", or "--" while there are no samples yet */
static string average_time_text(float average_inference_time, int inference_samples) {
	stringstream text;

	text << "Average Inference Time: ";

	if (inference_samples > 0)
		text << fixed << setprecision(2) << average_inference_time << "ms over " << inference_samples << (inference_samples == 1 ? " frame" : " frames");
	else
		text << "--";

	return text.str();
}

void render_live_window(const cv::Mat& frame_bgr,
						const vector<pair<float, int>>& results,
						const vector<string>& labels,
						const DisplayLayout& layout,
						float average_inference_time,
						int inference_samples,
						float fps,
						cv::Mat& display_window) {
	stringstream fps_text;

	clear_display_window(layout, display_window);

	/* Camera frames are stored as B,G,R */
	cv::Rect placed = place_image(frame_bgr, false, layout, display_window);

	/* Outline the region the model actually sees (centre square), inset so the border is not clipped */
	cv::Rect crop = get_center_crop(frame_bgr);
	double scale_x = static_cast<double>(placed.width)  / frame_bgr.cols;
	double scale_y = static_cast<double>(placed.height) / frame_bgr.rows;
	cv::Rect outline(placed.x + cvRound(crop.x * scale_x) + 1,
					 placed.y + cvRound(crop.y * scale_y) + 1,
					 cvRound(crop.width  * scale_x) - 2,
					 cvRound(crop.height * scale_y) - 2);

	draw_corner_box(display_window, outline, cv::Scalar(0, 255, 0, 0), std::max(2, display_window.rows / 240));

	if (fps > 0.0f)
		fps_text << fixed << setprecision(1) << fps;
	else
		fps_text << "--";

	int classification_start_y = draw_info_panel(display_window,
												 layout,
												 {average_time_text(average_inference_time, inference_samples), "FPS (End-to-End): " + fps_text.str()});

	draw_top_results(&display_window, results, labels, layout, classification_start_y);
}

void print_live_results(const vector<pair<float, int>>& results, const vector<string>& labels, float average_inference_time, int inference_samples, float fps) {
	printf("\n%s | FPS (End-to-End): %.1f\n", average_time_text(average_inference_time, inference_samples).c_str(), fps);

	draw_top_results(nullptr, results, labels, DisplayLayout(), 0);
}