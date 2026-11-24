/*
 * Copyright (C) 2026 Renesas Electronics Corp.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#ifndef TIMING_H
#define TIMING_H

#include <string>

using namespace std;

enum TimingStep {
	STEP_WAIT,
	STEP_MIRROR,
	STEP_CONVERT,
	STEP_INPUT,
	STEP_INVOKE,
	STEP_INVOKE_AGAIN,
	STEP_OUTPUT,
	STEP_RENDER,
	STEP_SHOW,
	STEP_LOOP,
	STEP_AGE_INFERENCE,
	STEP_AGE_SHOWN,
	STEP_COUNT
};

struct StepStat {
	double sum = 0.0;
	double min = 0.0;
	double max = 0.0;
	unsigned int count = 0;

	void add(double value);
	double avg() const;
};

/* Collects the time of every step of the camera loop over a number of frames */
struct TimingWindow {
	StepStat stats[STEP_COUNT];
	unsigned int frames = 0;

	void add(TimingStep step, double milliseconds);
	void reset();
};

struct TimingReportOptions {
	string title;
	bool show_mirror = false;
	bool show_display = false;
	bool show_second_call = false;
	/* Frame rate the camera delivers; the summary compares the processing time with its frame time */
	double camera_fps = 30.0;
	/* True if camera_fps was measured, false if it is an assumed value */
	bool camera_fps_measured = false;
};

/*
 * Prints an aligned table with average, minimum, maximum and share of the loop for every step, with bars and a summary.
 * Colour is used only on a real terminal (and not if NO_COLOR is set); bars use block characters only in a UTF-8 locale.
 */
void print_timing_report(const TimingWindow& window, const TimingReportOptions& options);

#endif
