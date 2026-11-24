/*
 * Copyright (C) 2026 Renesas Electronics Corp.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <unistd.h>

#include "timing.h"

#define BAR_WIDTH 20

#define ANSI_RESET "\033[0m"
#define ANSI_BOLD "\033[1m"
/* Bright variants: the standard dark red, green and yellow are hard to read on a black console */
#define ANSI_IDLE "\033[37m"
#define ANSI_GREEN "\033[92m"
#define ANSI_YELLOW "\033[93m"
#define ANSI_RED "\033[1;91m"

void StepStat::add(double value) {
	if (count == 0 || value < min)
		min = value;

	if (count == 0 || value > max)
		max = value;

	sum += value;
	count++;
}

double StepStat::avg() const {
	return count > 0 ? sum / count : 0.0;
}

void TimingWindow::add(TimingStep step, double milliseconds) {
	stats[step].add(milliseconds);
}

void TimingWindow::reset() {
	for (StepStat &stat : stats)
		stat = StepStat();

	frames = 0;
}

namespace {

struct Style {
	bool colour;
	bool unicode;
};

bool detect_colour() {
	const char *term = getenv("TERM");

	if (!isatty(STDOUT_FILENO) || getenv("NO_COLOR"))
		return false;

	return term && *term && strcmp(term, "dumb") != 0;
}

/* The first locale variable that is set decides, as in the C library */
bool detect_unicode() {
	const char *names[] = { "LC_ALL", "LC_CTYPE", "LANG" };

	for (const char *name : names) {
		const char *value = getenv(name);

		if (value && *value) {
			string text = value;

			transform(text.begin(), text.end(), text.begin(), [](unsigned char c) { return tolower(c); });

			return text.find("utf-8") != string::npos || text.find("utf8") != string::npos;
		}
	}

	return false;
}

const char *colour_for_share(const Style &style, double share) {
	if (!style.colour)
		return "";

	if (share < 0.25)
		return ANSI_GREEN;

	if (share < 0.50)
		return ANSI_YELLOW;

	return ANSI_RED;
}

string make_bar(const Style &style, double share) {
	int filled = (int)lround(share * BAR_WIDTH);
	string bar;

	filled = min(BAR_WIDTH, max(0, filled));

	for (int i = 0; i < BAR_WIDTH; i++) {
		if (i < filled)
			bar += style.unicode ? "\xE2\x96\x88" : "#";
		else
			bar += style.unicode ? "\xE2\x96\x91" : "-";
	}

	return bar;
}

void print_group(const Style &style, const char *name) {
	printf("  %s%s%s\n", style.colour ? ANSI_BOLD : "", name, style.colour ? ANSI_RESET : "");
}

/* idle: the step only waits (for the camera), so it is shown dimmed instead of coloured as load */
void print_row(const Style &style, const char *name, const StepStat &stat, double loop_average, bool idle, const char *note) {
	double share = loop_average > 0.0 ? stat.avg() / loop_average : 0.0;
	const char *colour = idle ? (style.colour ? ANSI_IDLE : "") : colour_for_share(style, share);
	const char *reset = style.colour ? ANSI_RESET : "";

	printf("    %-22s %7.2f %7.2f %7.2f %5.0f%%  %s%s%s%s%s\n",
	       name, stat.avg(), stat.min, stat.max, share * 100.0,
	       colour, make_bar(style, share).c_str(), reset, note ? "  " : "", note ? note : "");
}

} // namespace

void print_timing_report(const TimingWindow &window, const TimingReportOptions &options) {
	const Style style = { detect_colour(), detect_unicode() };
	const StepStat *stats = window.stats;
	const double loop_average = stats[STEP_LOOP].avg();
	const double second_call = options.show_second_call ? stats[STEP_INVOKE_AGAIN].avg() : 0.0;
	/* Processing time per frame: the loop without waiting for the camera and without the diagnostic 2nd call */
	const double busy = loop_average - stats[STEP_WAIT].avg() - second_call;
	const double camera_period = options.camera_fps > 0.0 ? 1000.0 / options.camera_fps : 0.0;
	const double load = camera_period > 0.0 ? busy / camera_period : 0.0;
	const double headroom = camera_period - busy;
	const double npu_share = camera_period > 0.0 ? stats[STEP_INVOKE].avg() / camera_period : 0.0;
	const char *load_colour = style.colour ? (load < 0.70 ? ANSI_GREEN : (load < 0.95 ? ANSI_YELLOW : ANSI_RED)) : "";
	const char *reset = style.colour ? ANSI_RESET : "";
	const string line(72, '-');

	printf("\n %s%s%s\n", style.colour ? ANSI_BOLD : "", options.title.c_str(), reset);
	printf(" %s\n", line.c_str());
	printf("  %-24s %7s %7s %7s %6s\n", "step (ms)", "avg", "min", "max", "share");

	print_group(style, "CAPTURE");
	print_row(style, "wait for frame", stats[STEP_WAIT], loop_average, true, "(idle, waits for the camera)");

	if (options.show_mirror)
		print_row(style, "mirror", stats[STEP_MIRROR], loop_average, false, nullptr);

	print_row(style, "crop + resize + colour", stats[STEP_CONVERT], loop_average, false, nullptr);

	print_group(style, "PRE-PROCESS");
	print_row(style, "quantize + model input", stats[STEP_INPUT], loop_average, false, nullptr);

	print_group(style, "NPU");
	print_row(style, "inference, 1st call", stats[STEP_INVOKE], loop_average, false, nullptr);

	if (options.show_second_call)
		print_row(style, "inference, 2nd call", stats[STEP_INVOKE_AGAIN], loop_average, false, "(diagnostic, same frame again)");

	print_group(style, options.show_display ? "POST + DISPLAY" : "POST");
	print_row(style, "read output + top 5", stats[STEP_OUTPUT], loop_average, false, nullptr);

	if (options.show_display) {
		print_row(style, "render window", stats[STEP_RENDER], loop_average, false, nullptr);
		print_row(style, "send to display", stats[STEP_SHOW], loop_average, false, nullptr);
	}

	printf(" %s\n", line.c_str());
	printf("  %-24s %7.2f   (over %u frames)\n", "whole loop", loop_average, window.frames);

	printf("  processing per frame %.1f ms%s -> up to %.1f FPS\n", busy, options.show_second_call ? " (without the 2nd call)" : "", busy > 0.0 ? 1000.0 / busy : 0.0);
	printf("  %scamera %.0f FPS (%s) = %.1f ms per frame: load %.0f%%, headroom %.1f ms%s\n", load_colour, options.camera_fps, options.camera_fps_measured ? "measured" : "assumed", camera_period, load * 100.0, headroom, reset);
	printf("  NPU inference %.1f ms = %.0f%% of the camera frame time\n", stats[STEP_INVOKE].avg(), npu_share * 100.0);
	printf("  frame age: %.1f ms at inference start | %.1f ms after the display call\n", stats[STEP_AGE_INFERENCE].avg(), stats[STEP_AGE_SHOWN].avg());

	if (options.show_second_call)
		printf("  note: -T runs a 2nd inference per frame, so the whole loop is longer; the lines above leave it out, the age after display includes it\n");
}
