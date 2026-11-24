/*
 * Copyright (C) 2026 Renesas Electronics Corp.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <algorithm>
#include <vector>

#include "wayland_utils.h"

/* Output selected with -o */
static string g_output_name = DEFAULT_OUTPUT_NAME;

bool is_valid_output_name(const string& name, string& reason) {
	if (name.empty() || name.size() > 32) {
		reason = "the name must have 1 to 32 characters";

		return false;
	}

	for (char c : name) {
		bool allowed = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '.' || c == '_' || c == '-';

		if (!allowed) {
			reason = "only letters, digits and . _ - are allowed (no spaces or other characters, also not in front of or after the name)";

			return false;
		}
	}

	return true;
}

bool set_output_name(const string& name) {
	string reason;

	if (!is_valid_output_name(name, reason))
		return false;

	g_output_name = name;

	return true;
}

const string& get_output_name() {
	return g_output_name;
}

/* wl_output events: only the name is used (it arrives with wl_output version 4) */
static void output_geometry(void *data, struct wl_output *wl_output, int32_t x, int32_t y, int32_t physical_width, int32_t physical_height,
							int32_t subpixel, const char *make, const char *model, int32_t transform) {}

static void output_mode(void *data, struct wl_output *wl_output, uint32_t flags, int32_t width, int32_t height, int32_t refresh) {}

static void output_done(void *data, struct wl_output *wl_output) {}

static void output_scale(void *data, struct wl_output *wl_output, int32_t factor) {}

static void output_name(void *data, struct wl_output *wl_output, const char *name) {
	snprintf((char*)data, WL_OUTPUT_NAME_LEN, "%s", name);
}

static void output_description(void *data, struct wl_output *wl_output, const char *description) {}

static const struct wl_output_listener output_listener = {
	.geometry = output_geometry,
	.mode = output_mode,
	.done = output_done,
	.scale = output_scale,
	.name = output_name,
	.description = output_description,
};

/* Wayland Registry functions */
static void registry_global(void *data, struct wl_registry *wl_registry, uint32_t name, const char *interface, uint32_t version) {
	struct wl_client_info *wl_cl_info = (struct wl_client_info*)data;

	if (strcmp(interface, wl_compositor_interface.name) == 0)
		wl_cl_info->wl_compositor = (wl_compositor*) wl_registry_bind(wl_registry, name, &wl_compositor_interface, 1);
	else if (strcmp(interface, wl_shm_interface.name) == 0)
		wl_cl_info->wl_shm = (wl_shm*) wl_registry_bind(wl_registry, name, &wl_shm_interface, 1);
	else if (strcmp(interface, wl_output_interface.name) == 0 && wl_cl_info->output_count < WL_MAX_OUTPUTS) {
		/* Version 4 is the first one that tells the name of the output */
		uint32_t output_version = version < 4 ? version : 4;
		int index = wl_cl_info->output_count++;

		wl_cl_info->output_names[index][0] = '\0';
		wl_cl_info->outputs[index] = (struct wl_output*) wl_registry_bind(wl_registry, name, &wl_output_interface, output_version);
		wl_output_add_listener(wl_cl_info->outputs[index], &output_listener, (void*)wl_cl_info->output_names[index]);
	} else if (strcmp(interface, xdg_wm_base_interface.name) == 0)
		wl_cl_info->xdg_wm_base = (xdg_wm_base*) wl_registry_bind(wl_registry, name, &xdg_wm_base_interface, 1);
}

static void registry_global_remove(void *data, struct wl_registry *wl_registry, uint32_t name) {}

static const struct wl_registry_listener registry_listener = {
	.global = registry_global,
	.global_remove = registry_global_remove,
};

/* XDG Shell functions */
static void xdg_wm_base_ping(void *data, struct xdg_wm_base *xdg_wm_base, uint32_t serial) {
	xdg_wm_base_pong(xdg_wm_base, serial);
}

static const struct xdg_wm_base_listener xdg_wm_base_listener = {
	.ping = xdg_wm_base_ping,
};

static void xdg_surface_configure(void *data, struct xdg_surface *xdg_surface, uint32_t serial) {
	struct wl_client_info *wl_cl_info = (wl_client_info*)data;

	xdg_surface_ack_configure(xdg_surface, serial);

	wl_cl_info->configured = true;
}

static const struct xdg_surface_listener xdg_surface_listener = {
	.configure = xdg_surface_configure,
};

/* Names of the outputs that wayland-info lists, for the message when the wanted one is missing */
static string list_output_names() {
	FILE* pipe = popen("wayland-info 2>/dev/null | grep -o \"name: '[^']*'\"", "r");
	vector<string> names;
	char line[128];

	if (!pipe)
		return "";

	while (fgets(line, sizeof(line), pipe)) {
		string text = line;
		size_t first = text.find('\'');
		size_t last = text.rfind('\'');

		if (first != string::npos && last > first + 1) {
			string name = text.substr(first + 1, last - first - 1);

			if (find(names.begin(), names.end(), name) == names.end())
				names.push_back(name);
		}
	}

	pclose(pipe);

	string result;

	for (const string& name : names)
		result += (result.empty() ? "" : ", ") + name;

	return result;
}

pair<int, int> get_display_dims() {
	int width = 0;
	int height = 0;

	/* The name was checked with is_valid_output_name(), so it is safe inside the command */
	const string command = "wayland-info | grep -A3 \"name: '" + g_output_name + "'\" | grep logical_width";

	FILE* pipe = popen(command.c_str(), "r");

	if (!pipe) {
		printf("Can't get display information, outputting results to terminal only!\n");
		return {0, 0};
	}

	if (fscanf(pipe,
			   " %*[^0-9]%d, %*[^0-9]%d",
			   &width, &height) != 2) {

		pclose(pipe);

		string available = list_output_names();

		printf("Display output '%s' not found or its size is unknown (available: %s). "
			   "Select one with -o <name>. Outputting results to terminal only!\n",
			   g_output_name.c_str(),
			   available.empty() ? "none found; is wayland-info installed and Weston running?" : available.c_str());

		return {0, 0};
	}

	pclose(pipe);

	printf("Detected display size: %d x %d (%s)\n", width, height, g_output_name.c_str());

	return {width, height};
}

/* The wl_output with the selected name, or NULL (the compositor then chooses the output) */
static struct wl_output *find_selected_output(struct wl_client_info &info) {
	string available;

	for (int i = 0; i < info.output_count; i++) {
		if (g_output_name == info.output_names[i])
			return info.outputs[i];

		if (info.output_names[i][0])
			available += (available.empty() ? "" : ", ") + string(info.output_names[i]);
	}

	printf("Output '%s' is not known to the compositor (available: %s); the compositor chooses the display for the window\n",
		   g_output_name.c_str(), available.empty() ? "no names, the compositor is older than wl_output version 4" : available.c_str());

	return NULL;
}

/*
 * Releases the globals that were bound from the registry: the outputs, the xdg shell, shm and the compositor.
 * An output of version 3 or newer has a release request that tells the compositor; for older versions only the
 * client side proxy can be destroyed. The xdg shell has to outlive the xdg surfaces, so call this after they are destroyed.
 */
static void release_client_info(struct wl_client_info &info) {
	for (int i = 0; i < info.output_count; i++) {
		if (!info.outputs[i])
			continue;

		if (wl_proxy_get_version((struct wl_proxy*)info.outputs[i]) >= WL_OUTPUT_RELEASE_SINCE_VERSION)
			wl_output_release(info.outputs[i]);
		else
			wl_output_destroy(info.outputs[i]);

		info.outputs[i] = NULL;
	}

	info.output_count = 0;

	if (info.xdg_wm_base) {
		xdg_wm_base_destroy(info.xdg_wm_base);
		info.xdg_wm_base = NULL;
	}

	if (info.wl_shm) {
		wl_shm_destroy(info.wl_shm);
		info.wl_shm = NULL;
	}

	if (info.wl_compositor) {
		wl_compositor_destroy(info.wl_compositor);
		info.wl_compositor = NULL;
	}
}

static int create_shm_file(size_t size) {
	char file_name[] = "/tmp/wayland-shm-XXXXXX";

	int fd = mkstemp(file_name);

	if (fd < 0)
		throw runtime_error("Failed to create temporary wayland shared memory file!");

	/* Unlink as file_name is no longer needed at this point */
	unlink(file_name);

	if (ftruncate(fd, size) < 0)
		throw runtime_error("Failed to truncate file!");

	return fd;
}

void display_results(int display_width, int display_height, cv::Mat& display_image) {
	int stride;
	size_t size;
	bool wait_for_exit = true;

	struct wl_client_info wl_cl_info = { 0 };
	struct wl_display *wl_display = wl_display_connect(NULL);

	if (!wl_display)
		throw runtime_error("Failed to connect to wayland, please check if display is connected...");

	/* Get Wayland server registry global registry and attach listener to it */
	struct wl_registry *wl_registry = wl_display_get_registry(wl_display);
	wl_registry_add_listener(wl_registry, &registry_listener, (void*)&wl_cl_info);

	/* Wait for Wayland server to complete all pending events */
	wl_display_roundtrip(wl_display);

	/* A second round trip delivers the events of the outputs we just bound (their names) */
	wl_display_roundtrip(wl_display);

	if (!wl_cl_info.wl_compositor)
		throw runtime_error("Failed to access wayland compositor!");
	else if (!wl_cl_info.wl_shm)
		throw runtime_error("Failed to access wayland shared memory!");

	if (!wl_cl_info.xdg_wm_base)
		throw runtime_error("Failed to access wayland xdg shell!");

	/* The image must hold the whole window (4 bytes per pixel) */
	if (display_image.type() != CV_8UC4 || display_image.cols != display_width || display_image.rows != display_height || !display_image.isContinuous())
		throw runtime_error("Invalid image for the display!");

	/* Use XDG shell to create application window */
	xdg_wm_base_add_listener(wl_cl_info.xdg_wm_base, &xdg_wm_base_listener, NULL);

	struct wl_surface *wl_surface = wl_compositor_create_surface(wl_cl_info.wl_compositor);

	if (!wl_surface)
		throw runtime_error("Failed to create Wayland surface!");

	/* Calculate Wayland shared memory pool size of image to be displayed */
	stride = display_width * 4;
	size = display_height * stride;

	/* Create temporary shared memory file that will be used for the display */
	int fd = create_shm_file(size);

	/* Copy image data to the Wayland shared memory pool */
	uint8_t* pool_data = (uint8_t*) mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

	if (pool_data == MAP_FAILED) {
		close(fd);

		throw runtime_error("Failed to map wayland shared memory!");
	}

	memcpy(pool_data, display_image.data, size);
	struct wl_shm_pool *pool = wl_shm_create_pool(wl_cl_info.wl_shm, fd, size);
	struct wl_buffer *buffer = wl_shm_pool_create_buffer(pool, 0, display_width, display_height, stride, WL_SHM_FORMAT_XRGB8888);

	/* Destory pool and close temporary file */
	wl_shm_pool_destroy(pool);
	close(fd);

	/* Create XDG shell surface and attach listener */
	struct xdg_surface *xdg_surface = xdg_wm_base_get_xdg_surface(wl_cl_info.xdg_wm_base, wl_surface);
	xdg_surface_add_listener(xdg_surface, &xdg_surface_listener, (void*)&wl_cl_info);

	/* Make XDG shell toplevel and fullscreen */
	struct xdg_toplevel *xdg_toplevel = xdg_surface_get_toplevel(xdg_surface);
	xdg_toplevel_set_fullscreen(xdg_toplevel, find_selected_output(wl_cl_info));

	/* Make sure the display is configured before displaying image */
	wl_surface_commit(wl_surface);

	/* Wait until display is finished configuring */
	while (!wl_cl_info.configured) {
		wl_display_dispatch(wl_display);
	}

	/* Display the image */
	wl_surface_attach(wl_surface, buffer, 0, 0);
	wl_surface_damage(wl_surface, 0, 0, display_width, display_height);
	wl_surface_commit(wl_surface);
	wl_display_dispatch(wl_display);

	printf("Please press the ENTER key to close the application\n");

	/* Keep image on display until user presses the enter key on serial terminal */
	while (wait_for_exit) {
		if (cin.get())
			wait_for_exit = false;
	}

	/* Release everything in the reverse order of its creation */
	xdg_toplevel_destroy(xdg_toplevel);
	xdg_surface_destroy(xdg_surface);
	wl_surface_destroy(wl_surface);
	wl_buffer_destroy(buffer);
	munmap(pool_data, size);
	release_client_info(wl_cl_info);
	wl_registry_destroy(wl_registry);

	wl_display_flush(wl_display);
	wl_display_disconnect(wl_display);
}

/* Live display (persistent window) */
static void live_buffer_release(void *data, struct wl_buffer *wl_buffer) {
	((struct wl_live_buffer*)data)->busy = false;
}

static const struct wl_buffer_listener live_buffer_listener = {
	.release = live_buffer_release,
};

WaylandDisplay::~WaylandDisplay() {
	for (auto &buf : buffers_) {
		if (buf.buffer)
			wl_buffer_destroy(buf.buffer);
	}

	if (xdg_toplevel_)
		xdg_toplevel_destroy(xdg_toplevel_);

	if (xdg_surface_)
		xdg_surface_destroy(xdg_surface_);

	if (surface_)
		wl_surface_destroy(surface_);

	/* After the surfaces: the xdg shell must not go away before the xdg surfaces that were made with it */
	release_client_info(info_);

	if (registry_)
		wl_registry_destroy(registry_);

	if (display_) {
		/* Sends the destroy requests; the connection may already be gone, then this does nothing */
		wl_display_flush(display_);
		wl_display_disconnect(display_);
	}

	if (data_)
		munmap(data_, map_size_);
}

void WaylandDisplay::open(int width, int height) {
	width_ = width;
	height_ = height;

	int stride = width_ * 4;

	buffer_size_ = (size_t)height_ * stride;
	map_size_ = buffer_size_ * 2;

	display_ = wl_display_connect(NULL);

	if (!display_)
		throw runtime_error("Failed to connect to wayland, please check if display is connected...");

	registry_ = wl_display_get_registry(display_);
	wl_registry_add_listener(registry_, &registry_listener, (void*)&info_);
	wl_display_roundtrip(display_);

	/* A second round trip delivers the events of the outputs we just bound (their names) */
	wl_display_roundtrip(display_);

	if (!info_.wl_compositor)
		throw runtime_error("Failed to access wayland compositor!");
	else if (!info_.wl_shm)
		throw runtime_error("Failed to access wayland shared memory!");
	else if (!info_.xdg_wm_base)
		throw runtime_error("Failed to access wayland xdg shell!");

	xdg_wm_base_add_listener(info_.xdg_wm_base, &xdg_wm_base_listener, NULL);

	surface_ = wl_compositor_create_surface(info_.wl_compositor);

	if (!surface_)
		throw runtime_error("Failed to create Wayland surface!");

	/* One shared memory file holding both buffers */
	int fd = create_shm_file(map_size_);

	data_ = (uint8_t*) mmap(NULL, map_size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

	if (data_ == MAP_FAILED) {
		data_ = nullptr;
		close(fd);

		throw runtime_error("Failed to map wayland shared memory!");
	}

	struct wl_shm_pool *pool = wl_shm_create_pool(info_.wl_shm, fd, map_size_);

	for (int i = 0; i < 2; i++) {
		buffers_[i].buffer = wl_shm_pool_create_buffer(pool, i * buffer_size_, width_, height_, stride, WL_SHM_FORMAT_XRGB8888);
		buffers_[i].busy = false;
		wl_buffer_add_listener(buffers_[i].buffer, &live_buffer_listener, &buffers_[i]);
	}

	wl_shm_pool_destroy(pool);
	close(fd);

	xdg_surface_ = xdg_wm_base_get_xdg_surface(info_.xdg_wm_base, surface_);
	xdg_surface_add_listener(xdg_surface_, &xdg_surface_listener, (void*)&info_);

	xdg_toplevel_ = xdg_surface_get_toplevel(xdg_surface_);
	xdg_toplevel_set_fullscreen(xdg_toplevel_, find_selected_output(info_));

	wl_surface_commit(surface_);

	while (!info_.configured) {
		if (wl_display_dispatch(display_) < 0)
			throw runtime_error("Lost connection to the wayland compositor!");
	}
}

/* Handles pending compositor events (buffer releases, pings) without blocking longer than timeout_ms */
void WaylandDisplay::pump_events(int timeout_ms) {
	while (wl_display_prepare_read(display_) != 0)
		wl_display_dispatch_pending(display_);

	wl_display_flush(display_);

	struct pollfd pfd = { wl_display_get_fd(display_), POLLIN, 0 };

	if (poll(&pfd, 1, timeout_ms) > 0 && (pfd.revents & (POLLIN | POLLHUP | POLLERR)))
		wl_display_read_events(display_);
	else
		wl_display_cancel_read(display_);

	wl_display_dispatch_pending(display_);
}

bool WaylandDisplay::show(const cv::Mat& image) {
	int index = -1;

	if (!display_ || image.type() != CV_8UC4 || image.cols != width_ || image.rows != height_ || !image.isContinuous())
		throw runtime_error("Invalid image for live display!");

	pump_events(0);

	/* Prefer a free buffer; if both are still held by the compositor wait one short interval, then drop the frame */
	for (int attempt = 0; attempt < 2 && index < 0; attempt++) {
		for (int i = 0; i < 2; i++) {
			if (!buffers_[i].busy) {
				index = i;
				break;
			}
		}

		if (index < 0)
			pump_events(16);
	}

	if (wl_display_get_error(display_) != 0)
		throw runtime_error("Lost connection to the wayland compositor!");

	if (index < 0)
		return false;

	memcpy(data_ + index * buffer_size_, image.data, buffer_size_);

	wl_surface_attach(surface_, buffers_[index].buffer, 0, 0);
	wl_surface_damage(surface_, 0, 0, width_, height_);
	wl_surface_commit(surface_);
	buffers_[index].busy = true;
	wl_display_flush(display_);

	return true;
}

void WaylandDisplay::poll_events(int timeout_ms) {
	if (!display_)
		return;

	pump_events(timeout_ms);

	if (wl_display_get_error(display_) != 0)
		throw runtime_error("Lost connection to the wayland compositor!");
}
