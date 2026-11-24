/*
 * Copyright (C) 2026 Renesas Electronics Corp.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <algorithm>

#include "graphstats.h"

float percentile(const vector<float>& sorted, float percent) {
	if (sorted.empty())
		return 0.0f;

	float position = (sorted.size() - 1) * min(100.0f, max(0.0f, percent)) / 100.0f;
	size_t index = (size_t)position;
	float fraction = position - index;

	if (index + 1 >= sorted.size())
		return sorted.back();

	return sorted[index] * (1.0f - fraction) + sorted[index + 1] * fraction;
}

int histogram_bin(float value, float low, float high, int bins) {
	if (bins < 1 || high <= low || value <= low)
		return 0;

	if (value >= high)
		return bins - 1;

	return min(bins - 1, (int)((value - low) / (high - low) * bins));
}

vector<int> histogram(const vector<float>& samples, float low, float high, int bins) {
	vector<int> counts(max(bins, 1), 0);

	for (float value : samples)
		counts[histogram_bin(value, low, high, (int)counts.size())]++;

	return counts;
}

void GraphScale::update(float window_low, float window_high, float decay, float padding, float min_span) {
	float target_low = max(0.0f, window_low - padding);
	float target_high = window_high + padding;

	min_span_ = min_span;

	if (!valid_) {
		low_ = target_low;
		high_ = target_high;
		valid_ = true;

		return;
	}

	/* Widen at once, narrow slowly; never narrower than the values that are shown */
	low_ = target_low < low_ ? target_low : low_ + (target_low - low_) * decay;
	high_ = target_high > high_ ? target_high : high_ + (target_high - high_) * decay;
}

float GraphScale::low() const {
	return max(0.0f, low_);
}

float GraphScale::high() const {
	return max(high_, low() + min_span_);
}
