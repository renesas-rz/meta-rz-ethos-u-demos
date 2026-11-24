/*
 * Copyright (C) 2026 Renesas Electronics Corp.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#ifndef WAYLAND_UTILS_H
#define WAYLAND_UTILS_H

#include <cstdint>
#include <cstdio>
#include <iostream>
#include <string>

#include <opencv4/opencv2/core/core.hpp>
#include <poll.h>
#include <sys/mman.h>
#include <unistd.h>
#include <wayland-client.h>

#include <wayland/xdg-shell-client-protocol.h>

using namespace std;

/* Output (display) that is used when -o is not given */
#define DEFAULT_OUTPUT_NAME "HDMI-A-1"
#define WL_MAX_OUTPUTS 8
#define WL_OUTPUT_NAME_LEN 64

struct wl_client_info {
	struct wl_compositor *wl_compositor;
	struct wl_shm *wl_shm;
	struct xdg_wm_base *xdg_wm_base;
	bool configured;
	/* All outputs of the compositor with their names (the name is known from wl_output version 4) */
	struct wl_output *outputs[WL_MAX_OUTPUTS];
	char output_names[WL_MAX_OUTPUTS][WL_OUTPUT_NAME_LEN];
	int output_count;
};

struct wl_live_buffer {
	struct wl_buffer *buffer;
	bool busy;
};

/*
 * Persistent fullscreen window for live (per-frame) output.
 * Uses two shared memory buffers so a frame is never written while the compositor reads it.
 */
class WaylandDisplay {
public:
	WaylandDisplay() = default;
	~WaylandDisplay();
	WaylandDisplay(const WaylandDisplay&) = delete;
	WaylandDisplay& operator=(const WaylandDisplay&) = delete;

	/* Connects to the compositor and creates the fullscreen window. Throws on failure. */
	void open(int width, int height);

	/* Shows a CV_8UC4 (B,G,R,X byte order) image of exactly width x height. Returns false if the frame was dropped. */
	bool show(const cv::Mat& image);

	/* Handles compositor events (e.g. pings) for up to timeout_ms, so the window stays responsive while waiting */
	void poll_events(int timeout_ms);

private:
	void pump_events(int timeout_ms);

	int width_ = 0;
	int height_ = 0;
	size_t buffer_size_ = 0;
	size_t map_size_ = 0;
	uint8_t *data_ = nullptr;
	struct wl_client_info info_ = {};
	struct wl_display *display_ = nullptr;
	struct wl_registry *registry_ = nullptr;
	struct wl_surface *surface_ = nullptr;
	struct xdg_surface *xdg_surface_ = nullptr;
	struct xdg_toplevel *xdg_toplevel_ = nullptr;
	struct wl_live_buffer buffers_[2] = {};
};

/*****************************************
* Functions
******************************************/
/*
 * Checks the name of a display output, because it is put into a command line:
 * 1 to 32 characters, only letters, digits and . _ - (for example HDMI-A-1, DSI-1, DVI-D-1).
 */
bool is_valid_output_name(const string& name, string& reason);

/* Selects the output that is used for the display size and for the window. Returns false (and changes nothing) if the name is invalid. */
bool set_output_name(const string& name);
const string& get_output_name();

/* Size of the selected output; (0, 0) if it is not found */
pair<int, int> get_display_dims();
void display_results(int display_width, int display_height, cv::Mat& display_image);

#endif
