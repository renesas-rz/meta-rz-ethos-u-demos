/*
 * Copyright (C) 2026 Renesas Electronics Corp.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <algorithm>
#include <cmath>
#include <cstdio>

#include <opencv4/opencv2/imgproc/imgproc.hpp>

#include "app_info.h"
#include "demo.h"

namespace {

/* Colours are in B,G,R order, as the display buffer is */
/* Renesas blue #2A289B */
const cv::Scalar COLOR_RENESAS_BLUE(155, 40, 42, 0);
const cv::Scalar COLOR_HEADER_TEXT(240, 232, 232, 0);
/* Light tint of the Renesas blue (#7896FF) for the chip name; readable on the dark bar */
const cv::Scalar COLOR_CHIP(255, 150, 120, 0);
const cv::Scalar COLOR_DIVIDER(215, 150, 150, 0);
const cv::Scalar COLOR_BACKGROUND(26, 22, 22, 0);
const cv::Scalar COLOR_PANEL(46, 40, 38, 0);
const cv::Scalar COLOR_TRACK(70, 64, 62, 0);
const cv::Scalar COLOR_ACCENT(40, 190, 255, 0);
const cv::Scalar COLOR_GREEN(90, 200, 80, 0);
const cv::Scalar COLOR_GREEN_DIM(62, 120, 46, 0);
const cv::Scalar COLOR_GREEN_BRIGHT(140, 235, 130, 0);
const cv::Scalar COLOR_TEXT(240, 240, 240, 0);
const cv::Scalar COLOR_MUTED(165, 165, 165, 0);
const cv::Scalar COLOR_GREY_BAR(150, 140, 135, 0);

const int FONT = cv::FONT_HERSHEY_DUPLEX;

/* Height of a capital letter at font scale 1.0 is about 21 pixels */
double scale_for_height(double pixels) {
	return pixels / 21.0;
}

int text_width(const string& text, double scale, int thickness) {
	int baseline = 0;

	return cv::getTextSize(text, FONT, scale, thickness, &baseline).width;
}

/* Shrinks the font until the text fits, and as a last resort shortens it with "..." */
string fit_text(const string& text, double& scale, int max_width, double min_scale, int thickness) {
	while (scale > min_scale && text_width(text, scale, thickness) > max_width)
		scale -= 0.05;

	if (text_width(text, scale, thickness) <= max_width)
		return text;

	string cut = text;

	while (!cut.empty() && text_width(cut + "...", scale, thickness) > max_width)
		cut.pop_back();

	return cut + "...";
}

/* Alternative wordings of a caption, longest first: the text, without a "(...)" part, before the first comma */
vector<string> caption_variants(const string& text) {
	vector<string> variants = {text};
	const size_t paren = text.find(" (");
	const size_t comma = text.find(',');

	if (paren != string::npos)
		variants.push_back(text.substr(0, paren));

	if (comma != string::npos)
		variants.push_back(text.substr(0, comma));

	return variants;
}

/* The first wording that fits at the given font scale; if none fits, the last one is shrunk or shortened with "..." */
string fit_caption(const vector<string>& variants, double& scale, int max_width) {
	const double start_scale = scale;

	for (const string& variant : variants) {
		scale = start_scale;

		if (text_width(variant, scale, 1) <= max_width)
			return variant;
	}

	scale = start_scale;

	return fit_text(variants.back(), scale, max_width, 0.4, 1);
}

void put_text(cv::Mat& image, const string& text, int x, int baseline_y, double scale, const cv::Scalar& colour, int thickness) {
	cv::putText(image, text, cv::Point(x, baseline_y), FONT, scale, colour, thickness, cv::LINE_AA);
}

string first_item(const string& label) {
	size_t comma = label.find(',');

	return comma == string::npos ? label : label.substr(0, comma);
}

string format_number(const char* format, double value) {
	char buffer[32];

	snprintf(buffer, sizeof(buffer), format, value);

	return string(buffer);
}

void fill_rect(cv::Mat& image, int x, int y, int width, int height, const cv::Scalar& colour) {
	if (width > 0 && height > 0)
		cv::rectangle(image, cv::Rect(x, y, width, height), colour, cv::FILLED);
}

/* Confidence bar: the full length is 100% */
void draw_bar(cv::Mat& image, int x, int y, int width, int height, float percent, const cv::Scalar& fill) {
	float clamped = min(100.0f, max(0.0f, percent));

	fill_rect(image, x, y, width, height, COLOR_TRACK);
	fill_rect(image, x, y, static_cast<int>(width * clamped / 100.0f), height, fill);
}

/* Picture, as large as the area allows, optionally with an outline of the centre square the model looks at */
void draw_picture(cv::Mat& window, const cv::Mat& frame, bool frame_is_rgb, bool show_crop_box, const cv::Rect& area) {
	cv::Mat resized;

	const double scale = min(static_cast<double>(area.width) / frame.cols, static_cast<double>(area.height) / frame.rows);
	const int width = max(1, static_cast<int>(round(frame.cols * scale)));
	const int height = max(1, static_cast<int>(round(frame.rows * scale)));
	const int x = area.x + (area.width - width) / 2;
	const int y = area.y + (area.height - height) / 2;

	cv::resize(frame, resized, cv::Size(width, height), 0, 0, scale > 1.0 ? cv::INTER_LINEAR : cv::INTER_AREA);

	/* Convert straight into the window (same size and type, so nothing is reallocated) */
	cv::Mat target = window(cv::Rect(x, y, width, height));

	cv::cvtColor(resized, target, frame_is_rgb ? cv::COLOR_RGB2BGRA : cv::COLOR_BGR2BGRA);

	if (!show_crop_box)
		return;

	const cv::Rect crop = get_center_crop(frame);
	const double scale_x = static_cast<double>(width) / frame.cols;
	const double scale_y = static_cast<double>(height) / frame.rows;

	draw_corner_box(window,
					cv::Rect(x + cvRound(crop.x * scale_x) + 1, y + cvRound(crop.y * scale_y) + 1,
							 cvRound(crop.width * scale_x) - 2, cvRound(crop.height * scale_y) - 2),
					COLOR_ACCENT, max(2, window.rows / 240));
}

/*
 * Graph of the inference time: a line of the newest frames (newest at the right) and, if there is room,
 * a histogram of all kept frames next to it, drawn in the same vertical scale, so no extra axis is needed.
 */
void draw_timing_graph(cv::Mat& window, const deque<float>& history, const vector<float>& sorted, const cv::Rect& area,
					   float low, float high, bool with_histogram) {
	auto y_of = [&](float value) {
		float clipped = min(max(value, low), high);

		return area.y + area.height - static_cast<int>((clipped - low) / (high - low) * area.height);
	};

	const int line_width = with_histogram ? static_cast<int>(area.width * 0.62) : area.width;
	const size_t count = min<size_t>(history.size(), DEMO_LINE_FRAMES);

	if (count >= 2) {
		vector<cv::Point> points;
		const size_t first = history.size() - count;
		const size_t offset = DEMO_LINE_FRAMES - count;

		for (size_t i = 0; i < count; i++) {
			int x = area.x + static_cast<int>(line_width * static_cast<double>(offset + i) / (DEMO_LINE_FRAMES - 1));

			points.push_back(cv::Point(x, y_of(history[first + i])));
		}

		cv::polylines(window, points, false, COLOR_GREEN, 2, cv::LINE_AA);
	}

	if (!with_histogram || sorted.empty())
		return;

	const int bins = 10;
	const int histogram_x = area.x + line_width + max(6, area.width / 20);
	const int histogram_width = area.x + area.width - histogram_x;
	const vector<int> counts = histogram(sorted, low, high, bins);
	const int tallest = *max_element(counts.begin(), counts.end());
	const int median_bin = histogram_bin(percentile(sorted, 50.0f), low, high, bins);

	for (int bin = 0; bin < bins; bin++) {
		if (counts[bin] == 0 || tallest == 0)
			continue;

		const int y_top = y_of(low + (bin + 1) * (high - low) / bins);
		const int y_bottom = y_of(low + bin * (high - low) / bins);
		const int length = max(1, histogram_width * counts[bin] / tallest);

		fill_rect(window, histogram_x, y_top + 1, length, max(1, y_bottom - y_top - 1), bin == median_bin ? COLOR_GREEN_BRIGHT : COLOR_GREEN_DIM);
	}

	cv::line(window, cv::Point(histogram_x - 1, area.y), cv::Point(histogram_x - 1, area.y + area.height), COLOR_TRACK, 1);
}

} // namespace

void DemoView::render(const cv::Mat& frame,
					  const vector<float>& scores,
					  const vector<string>& labels,
					  const DisplayLayout& layout,
					  const DemoInfo& info,
					  cv::Mat& display_window) {
	const int width = layout.width;
	const int height = layout.height;

	const bool allocate = display_window.empty() || display_window.cols != width || display_window.rows != height;

	if (allocate)
		display_window.create(height, width, CV_8UC4);

	/* Camera: keep the static parts and redraw the text only every few frames. Pictures: draw everything every time */
	const bool full_redraw = allocate || !info.partial_redraw;
	const bool refresh_text = full_redraw || (frame_count_ % DEMO_TEXT_PERIOD) == 0;

	frame_count_++;

	/* Average the class scores over the last few frames so the best match does not flicker between similar classes */
	if (!info.smooth_scores || smoothed_.size() != scores.size()) {
		smoothed_ = scores;
	} else {
		for (size_t i = 0; i < scores.size(); i++)
			smoothed_[i] += DEMO_SMOOTHING_ALPHA * (scores[i] - smoothed_[i]);
	}

	if (info.show_graph && info.has_inference_sample) {
		history_.push_back(info.inference_time);

		if (history_.size() > DEMO_HISTORY_FRAMES)
			history_.pop_front();
	}

	/* Geometry */
	const int margin = max(10, width / 80);
	const int header_height = max(40, static_cast<int>(height * 0.075));
	const int bar_height = max(72, static_cast<int>(height * 0.14));
	const int bar_top = height - bar_height;
	const int content_top = header_height + margin;
	const int content_height = bar_top - content_top - margin;
	const int split = static_cast<int>(width * 0.60);
	const cv::Rect picture_area(margin, content_top, split - margin, content_height);
	const cv::Rect panel(split + margin, content_top, width - split - 2 * margin, content_height);
	const double ph = panel.height;

	const int block_width = static_cast<int>(width * 0.62) / 3;
	const double value_scale = scale_for_height(bar_height * 0.38);
	const double unit_scale = value_scale * 0.45;
	const double caption_scale = scale_for_height(bar_height * 0.14);
	const int value_baseline = bar_top + static_cast<int>(bar_height * 0.62);
	const int caption_baseline = bar_top + static_cast<int>(bar_height * 0.88);
	const cv::Rect side_area(static_cast<int>(width * 0.64), bar_top + static_cast<int>(bar_height * 0.18),
							 width - static_cast<int>(width * 0.64) - margin, static_cast<int>(bar_height * 0.55));

	/* Static parts: background, header with the wordmark, bottom bar with the chip name */
	if (full_redraw) {
		display_window.setTo(COLOR_BACKGROUND);

		fill_rect(display_window, 0, 0, width, header_height, COLOR_RENESAS_BLUE);

		const double wordmark_scale = scale_for_height(header_height * 0.50);
		const int wordmark_width = text_width("Renesas", wordmark_scale, 2);

		put_text(display_window, "Renesas", margin, static_cast<int>(header_height * 0.75), wordmark_scale, COLOR_HEADER_TEXT, 2);

		/* Divider and the device name next to the wordmark */
		const int divider_x = margin + wordmark_width + margin;
		const double board_scale = scale_for_height(header_height * 0.40);
		const int board_width = text_width(BOARD_NAME, board_scale, 1);

		cv::line(display_window, cv::Point(divider_x, static_cast<int>(header_height * 0.28)),
				 cv::Point(divider_x, static_cast<int>(header_height * 0.76)), COLOR_DIVIDER, max(1, header_height / 40));
		put_text(display_window, BOARD_NAME, divider_x + margin, static_cast<int>(header_height * 0.75), board_scale, COLOR_HEADER_TEXT, 1);

		/* The title uses the space that is left on the right */
		double title_scale = scale_for_height(header_height * 0.30);
		const string title_text = fit_text(info.title, title_scale, width - (divider_x + margin + board_width) - 3 * margin, 0.4, 1);

		put_text(display_window, title_text, width - margin - text_width(title_text, title_scale, 1),
				 static_cast<int>(header_height * 0.66), title_scale, COLOR_HEADER_TEXT, 1);

		fill_rect(display_window, 0, bar_top, width, bar_height, COLOR_PANEL);
		fill_rect(display_window, 0, bar_top, width, 3, COLOR_ACCENT);

		double chip_scale = value_scale * 0.62;
		const string chip_text = fit_text("Ethos-U55", chip_scale, block_width - margin, 0.5, 2);

		put_text(display_window, chip_text, margin + 2 * block_width, value_baseline, chip_scale, COLOR_CHIP, 2);

		/* The caption must stay inside its own block, otherwise it runs into the graph caption on the right */
		double model_caption_scale = caption_scale;
		const string model_caption = fit_caption({"NPU - MobileNet V2", "MobileNet V2"}, model_caption_scale, block_width - margin);

		put_text(display_window, model_caption, margin + 2 * block_width, caption_baseline, model_caption_scale, COLOR_MUTED, 1);
	}

	draw_picture(display_window, frame, info.frame_is_rgb, info.show_crop_box, picture_area);

	if (!refresh_text)
		return;

	/* Clear only the parts that change */
	if (!full_redraw) {
		fill_rect(display_window, panel.x, panel.y, panel.width, panel.height, COLOR_BACKGROUND);
		fill_rect(display_window, 0, bar_top + 3, 2 * block_width + margin - 2, bar_height - 3, COLOR_PANEL);
		fill_rect(display_window, side_area.x - margin / 2, bar_top + 3, width - side_area.x + margin / 2, bar_height - 3, COLOR_PANEL);
	}

	const vector<pair<float, int>> top = top_results_from_scores(smoothed_);

	/* No class reaches the minimum (--min-percent): there is no best match, and nothing to keep green */
	if (top.empty()) {
		const float minimum = get_min_percent();
		const vector<string> lines = {minimum > 0.0f ? "No class reaches " + format_number("%g", minimum) + "%." : "No class is above 0%.", "The picture is not clear for the model.", "(The model gives less to every class.)"};
		double label_scale = min(2.2, scale_for_height(ph * 0.11));
		const int label_thickness = max(2, static_cast<int>(label_scale * 1.5));
		double line_scale = scale_for_height(ph * 0.036);

		confident_ = false;

		put_text(display_window, "BEST MATCH", panel.x, panel.y + static_cast<int>(ph * 0.06), scale_for_height(ph * 0.03), COLOR_MUTED, 1);
		put_text(display_window, "--", panel.x, panel.y + static_cast<int>(ph * 0.21), label_scale, COLOR_MUTED, label_thickness);

		/* One size for all lines: the one that fits the longest */
		for (const string& line : lines) {
			double fitted = scale_for_height(ph * 0.036);

			fit_text(line, fitted, panel.width, 0.4, 1);
			line_scale = min(line_scale, fitted);
		}

		for (size_t i = 0; i < lines.size(); i++)
			put_text(display_window, lines[i], panel.x, panel.y + static_cast<int>(ph * (0.42 + 0.07 * i)), line_scale, COLOR_MUTED, 1);
	} else {
		/* Best match */
		const bool uncertain = top[0].first < DEMO_UNCERTAIN_PERCENT;

		/* Camera: keep green until the confidence falls clearly below the limit. Pictures are independent, so no memory */
		if (info.smooth_scores)
			confident_ = top[0].first >= (confident_ ? DEMO_CONFIDENT_PERCENT - DEMO_CONFIDENT_MARGIN : DEMO_CONFIDENT_PERCENT);
		else
			confident_ = top[0].first >= DEMO_CONFIDENT_PERCENT;

		/* Green: confident, amber: plausible, grey: uncertain */
		const cv::Scalar match_colour = uncertain ? COLOR_MUTED : (confident_ ? COLOR_GREEN : COLOR_ACCENT);
		const cv::Scalar bar_colour = uncertain ? COLOR_GREY_BAR : (confident_ ? COLOR_GREEN : COLOR_ACCENT);
		const string full_label = labels[top[0].second];
		const string big_label = first_item(full_label);

		put_text(display_window, "BEST MATCH", panel.x, panel.y + static_cast<int>(ph * 0.06), scale_for_height(ph * 0.03), COLOR_MUTED, 1);

		double label_scale = min(2.2, scale_for_height(ph * 0.11));
		const int label_thickness = max(2, static_cast<int>(label_scale * 1.5));
		const string label_text = fit_text(big_label, label_scale, panel.width, 0.8, label_thickness);

		const int label_baseline = panel.y + static_cast<int>(ph * 0.21);

		put_text(display_window, label_text, panel.x, label_baseline, label_scale, match_colour, label_thickness);

		const string short_full = shorten_label(full_label);

		if (short_full != big_label) {
			double sub_scale = scale_for_height(ph * 0.035);
			const string sub_text = fit_text(short_full, sub_scale, panel.width, 0.4, 1);

			/*
			 * Place the grey line below the part of the big label that reaches under the baseline (g, y, p, ...).
			 * The depth is measured by OpenCV, so it is right for every display size.
			 */
			int label_descent = 0;

			cv::getTextSize(label_text, FONT, label_scale, label_thickness, &label_descent);

			const int sub_cap_height = static_cast<int>(21.0 * sub_scale);
			const int gap = max(8, static_cast<int>(ph * 0.018));
			const int sub_baseline = max(panel.y + static_cast<int>(ph * 0.29), label_baseline + label_descent + gap + sub_cap_height);

			put_text(display_window, sub_text, panel.x, sub_baseline, sub_scale, COLOR_MUTED, 1);
		}

		const string percent_text = format_number("%.0f%%", top[0].first);
		const double percent_scale = min(2.4, scale_for_height(ph * 0.08));
		const int percent_baseline = panel.y + static_cast<int>(ph * 0.43);

		put_text(display_window, percent_text, panel.x, percent_baseline, percent_scale, COLOR_TEXT, 2);

		if (uncertain) {
			put_text(display_window, "uncertain", panel.x + text_width(percent_text, percent_scale, 2) + 14, percent_baseline,
					 scale_for_height(ph * 0.035), COLOR_MUTED, 1);
		}

		draw_bar(display_window, panel.x, panel.y + static_cast<int>(ph * 0.455), panel.width, max(16, static_cast<int>(ph * 0.045)),
				 top[0].first, bar_colour);

		/* The other four matches */
		for (size_t i = 1; i < top.size(); i++) {
			const int row_top = panel.y + static_cast<int>(ph * (0.52 + static_cast<int>(i - 1) * 0.12));
			const string percent = format_number("%.1f%%", top[i].first);
			const double row_scale = scale_for_height(ph * 0.034);
			const int percent_width = text_width(percent, row_scale, 1);
			double fitted_scale = row_scale;
			const string name = fit_text(first_item(labels[top[i].second]), fitted_scale, panel.width - percent_width - 14, 0.4, 1);

			put_text(display_window, name, panel.x, row_top + static_cast<int>(ph * 0.06), fitted_scale, COLOR_TEXT, 1);
			put_text(display_window, percent, panel.x + panel.width - percent_width - 2, row_top + static_cast<int>(ph * 0.06), row_scale, COLOR_MUTED, 1);

			draw_bar(display_window, panel.x, row_top + static_cast<int>(ph * 0.08), panel.width, max(6, static_cast<int>(ph * 0.022)),
					 top[i].first, COLOR_GREY_BAR);
		}
	}

	/* Bottom bar: the headline numbers */
	const string time_text = info.inference_samples > 0 ? format_number("%.1f", info.average_inference_time) : "--";
	const string rate_text = info.rate > 0.0f ? format_number("%.0f", info.rate) : "--";

	double time_caption_scale = caption_scale;
	const string time_caption = fit_caption(caption_variants(info.time_caption), time_caption_scale, block_width - margin);

	put_text(display_window, time_text, margin, value_baseline, value_scale, COLOR_TEXT, 2);
	put_text(display_window, "ms", margin + text_width(time_text, value_scale, 2) + 8, value_baseline, unit_scale, COLOR_MUTED, 1);
	put_text(display_window, time_caption, margin, caption_baseline, time_caption_scale, COLOR_MUTED, 1);

	double rate_caption_scale = caption_scale;
	const string rate_caption = fit_caption(caption_variants(info.rate_caption), rate_caption_scale, block_width - margin);

	put_text(display_window, rate_text, margin + block_width, value_baseline, value_scale, COLOR_TEXT, 2);
	put_text(display_window, info.rate_unit, margin + block_width + text_width(rate_text, value_scale, 2) + 8, value_baseline, unit_scale, COLOR_MUTED, 1);
	put_text(display_window, rate_caption, margin + block_width, caption_baseline, rate_caption_scale, COLOR_MUTED, 1);

	/* Bottom right: graph of the inference time (camera), or the picture number and file name */
	if (info.show_graph) {
		/* Statistics of the kept frames (about 10 seconds) */
		vector<float> sorted(history_.begin(), history_.end());

		sort(sorted.begin(), sorted.end());

		if (!sorted.empty())
			scale_.update(sorted.front(), sorted.back(), DEMO_SCALE_DECAY, 0.3f, 4.0f);

		const bool with_histogram = side_area.width >= DEMO_HISTOGRAM_MIN_WIDTH;

		if (sorted.size() >= 2)
			draw_timing_graph(display_window, history_, sorted, side_area, scale_.low(), scale_.high(), with_histogram);

		/*
		 * Caption, from long to short, so it never runs out of the picture. "most" are the middle 90% of the kept frames
		 * (the fastest and the slowest 5% are left out), "worst" is the slowest frame of them.
		 */
		vector<string> graph_captions = {"NPU time per frame"};

		if (sorted.size() >= 30) {
			const string most = format_number("%.1f", percentile(sorted, 5.0f)) + "-" + format_number("%.1f", percentile(sorted, 95.0f));
			const string worst = format_number("%.1f", sorted.back());

			graph_captions = {"NPU time per frame: most " + most + ", worst " + worst + " ms",
							  "most " + most + ", worst " + worst + " ms",
							  "most frames " + most + " ms",
							  most + " ms"};
		}

		double graph_caption_scale = caption_scale;
		const string graph_caption = fit_caption(graph_captions, graph_caption_scale, side_area.width);

		put_text(display_window, graph_caption, side_area.x, caption_baseline, graph_caption_scale, COLOR_MUTED, 1);
	} else {
		double side_scale = value_scale * 0.8;
		const string side_value = fit_text(info.side_value, side_scale, side_area.width, 0.5, 2);
		double side_caption_scale = caption_scale;
		const string side_caption = fit_text(info.side_caption, side_caption_scale, side_area.width, 0.4, 1);

		put_text(display_window, side_value, side_area.x, value_baseline, side_scale, COLOR_TEXT, 2);
		put_text(display_window, side_caption, side_area.x, caption_baseline, side_caption_scale, COLOR_MUTED, 1);
	}
}
