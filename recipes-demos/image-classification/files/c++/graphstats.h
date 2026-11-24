/*
 * Copyright (C) 2026 Renesas Electronics Corp.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#ifndef GRAPHSTATS_H
#define GRAPHSTATS_H

#include <vector>

using namespace std;

/* Value at a percentile (0 to 100) of an ascending sorted list, with linear interpolation; 0 for an empty list */
float percentile(const vector<float>& sorted, float percent);

/*
 * Number of samples per bin, for bins of equal size between low and high.
 * Samples outside the range are counted in the first or the last bin.
 */
vector<int> histogram(const vector<float>& samples, float low, float high, int bins);

/* Index of the bin a value falls into (the same rule as histogram()) */
int histogram_bin(float value, float low, float high, int bins);

/*
 * Vertical range of a graph. A new extreme widens the range at once and the range narrows back only slowly,
 * so no value of the shown window is ever cut off, and the graph does not jump when a single slow frame
 * leaves the window.
 */
class GraphScale {
public:
	/*
	 * window_low / window_high: smallest and largest value of the samples that are shown.
	 * decay: part of the gap that is closed on every call while the range narrows (0.02 = slowly).
	 * padding: space added above and below the values. min_span: smallest allowed range.
	 */
	void update(float window_low, float window_high, float decay, float padding, float min_span);

	float low() const;
	float high() const;

private:
	float low_ = 0.0f;
	float high_ = 0.0f;
	float min_span_ = 1.0f;
	bool valid_ = false;
};

#endif
