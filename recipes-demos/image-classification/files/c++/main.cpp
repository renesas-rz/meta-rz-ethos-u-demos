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
#include <cerrno>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <csignal>
#include <deque>
#include <dirent.h>
#include <cstring>
#include <getopt.h>
#include <memory>
#include <poll.h>
#include <unistd.h>
#include <opencv4/opencv2/core/core.hpp>
#include <tensorflow/lite/delegates/external/external_delegate.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/kernels/register_ref.h>
#include <tensorflow/lite/model.h>

#include "app_info.h"
#include "camera.h"
#include "demo.h"
#include "memlog.h"
#include "opencl_utils.h"
#include "processing.h"
#include "timing.h"
#include "wayland_utils.h"

#define DEFAULT_IMAGE_FILE_PATH "/usr/share/image-classification/images/cat.bmp"
#define DEFAULT_INFERENCE_RUNS 10
#define DEFAULT_SLIDE_SECONDS 5
#define MAX_SLIDE_SECONDS 3600
#define DELEGATE_LIBRARY_PATH "/usr/lib/libethosu_op_delegate.so.1"
/* Time the slideshow waits between two checks of the keyboard and of the display */
#define SLIDESHOW_POLL_MS 100
#define MODEL_PATH "/usr/share/models/mobilenet_v2/mobilenet_v2_1.0_224_INT8/mobilenet_v2_1.0_224_INT8_vela.tflite"

using namespace std;
using namespace tflite;

/* The option that has no short letter */
#define OPT_MIN_PERCENT 0x100

static const char short_opts[] = "cuamsTDgMi:d:n:t:o:h";
static const option long_opts[] =
{
	{"console", no_argument, NULL, 'c'},
	{"usb", no_argument, NULL, 'u'},
	{"mirror", no_argument, NULL, 'm'},
	{"timing", no_argument, NULL, 'T'},
	{"demo", no_argument, NULL, 'D'},
	{"opencl", no_argument, NULL, 'g'},
	{"output", required_argument, NULL, 'o'},
	{"memory", no_argument, NULL, 'M'},
	{"slideshow", no_argument, NULL, 's'},
	{"time", required_argument, NULL, 't'},
	{"allow-dynamic-framerate", no_argument, NULL, 'a'},
	{"image", required_argument, NULL, 'i'},
	{"device", required_argument, NULL, 'd'},
	{"number", required_argument, NULL, 'n'},
	{"min-percent", required_argument, NULL, OPT_MIN_PERCENT},
	{"help", no_argument, NULL, 'h'},
	{0, 0, 0, 0}
};

void print_help(char *app_name) {
	printf("This application will run inference on a static image or a live USB camera stream using "
		"MobileNet V2 (Image Classification) on the Arm Ethos-U55 NPU.\n"
		"USAGE: %s, [-c] [-u [-m] [-a] [-T] [-D] [-g] [-d <video device>]] [-s [-t <seconds>]] [-o <output>] [-M] [-i <image file>] [-n <number of inference runs>] [-h]\n\n"
		"OPTIONS:\n"
		"-c, --console: Prints results to console only, no display.\n"
		"-o, --output: Display output (Weston output) for the size and the window, e.g. HDMI-A-1, DSI-1 or DVI-D-1.\n"
		"              List the names with: wayland-info | grep \"name: '\"\n"
		"              Default: HDMI-A-1\n"
		"-u, --usb: Runs continuous inference on a USB camera instead of a static image.\n"
		"              Press ENTER (or Ctrl-C) to quit. Cannot be combined with -i or -s.\n"
		"-m, --mirror: Mirrors the camera image horizontally (use with -u).\n"
		"-D, --demo: Eye-catching display for presentations: Renesas header, large picture, large best match with\n"
		"              confidence bar and headline numbers. Works with the camera (-u), the slideshow (-s) and a single\n"
		"              picture. With -u the scores are averaged over the last few frames and a low best match is shown\n"
		"              as uncertain. Needs the display (not with -c).\n"
		"-g, --opencl: Tries the GPU (OpenCL) for the resize and colour conversion of the camera picture (use with -u).\n"
		"              Compare with -T. Falls back to the CPU if OpenCL is not available\n"
		"-M, --memory: Prints the memory use of the program once a minute (use with -u or -s), to look for memory growth.\n"
		"-T, --timing: Prints the average time of every processing step and the frame age (use with -u).\n"
		"              Runs a second inference on each frame to compare with the first one, which lowers the FPS\n"
		"-a, --allow-dynamic-framerate: Keeps the camera's dynamic frame rate (use with -u).\n"
		"              Default: it is disabled for a constant frame rate, which can make the picture darker in dim light\n"
		"-d, --device: Video device to use with -u, e.g. /dev/video0.\n"
		"              Default: first USB video capture device found\n"
		"-s, --slideshow: Shows all images (.bmp, .jpg, .jpeg, .png) of the image folder one after the other,\n"
		"              starting with the selected image, until ENTER (or Ctrl-C) is pressed.\n"
		"-t, --time: Seconds each image is shown with -s.\n"
		"              Default: 5\n"
		"-i, --image: Path to image file to run inference with. With -s its folder is used for the slideshow.\n"
		"              Default: /usr/share/image-classification/images/cat.bmp\n"
		"-n, --number: Number of times to run inference (inference time will be an average of all runs; per image with -s).\n"
		"              With -u: number of frames the inference time and FPS are averaged over (running average).\n"
		"              Default: 10\n"
		"    --min-percent: A class is listed from this confidence (in percent, 0 to 100) on. The model gives its confidence in steps of 0.39 percent,\n"
		"              so what it gives below about 0.2 percent is stored as 0 and the order of such classes means nothing. 0 lists every class above 0 percent.\n"
		"              If no class reaches the limit the program says so.\n"
		"              Default: %g\n"
		"-h, --help: Prints this help and exit.\n",
		app_name, (double)TOP_RESULTS_MINIMUM_PERCENT);
}

bool set_interpreter(unique_ptr<Interpreter> &interpreter, unique_ptr<FlatBufferModel> &flatbuffer_model, string model_file, TfLiteDelegate &ethosu_delegate) {
	unique_ptr<OpResolver> op_resolver;

	/* Set operator resolver to use XNNPack delegate */
	op_resolver = make_unique<ops::builtin::BuiltinOpResolver>();

	/* Load model */
	flatbuffer_model = FlatBufferModel::VerifyAndBuildFromFile(model_file.c_str());

	if (!flatbuffer_model) {
		fprintf(stderr, "Failed to load the model\n");

		return false;
	}

	if (InterpreterBuilder(*flatbuffer_model, *op_resolver)(&interpreter) != kTfLiteOk) {
		fprintf(stderr, "Failed to create TFLite interpreter\n");

		return false;
	}

	if (interpreter->ModifyGraphWithDelegate(&ethosu_delegate) != kTfLiteOk) {
		fprintf(stderr, "Failed to apply Ethos-U delegate\n");

		return false;
	}

	return true;
}

cv::Mat set_input(unique_ptr<Interpreter> &interpreter, cv::Mat &image) {
	int height;
	int width;
	double zero_point = 0.0;
	float scale = 1.0f;
	ModelDataType data_type;

	int index = interpreter->inputs()[0];

	TfLiteTensor *input_tensor = interpreter->tensor(index);
	height = input_tensor->dims->data[1];
	width = input_tensor->dims->data[2];

	if (input_tensor->type == kTfLiteInt8 || input_tensor->type == kTfLiteUInt8) {
		data_type = (input_tensor->type == kTfLiteInt8) ? ModelDataType::INT8 : ModelDataType::UINT8;
		zero_point = input_tensor->params.zero_point;
		scale = input_tensor->params.scale;
	} else if (input_tensor->type == kTfLiteFloat32) {
		data_type = ModelDataType::FLOAT32;
	} else {
		throw runtime_error("Input Data Type not supported");
	}

	cv::Mat processed = preprocess_image(image, width, height, data_type, scale, zero_point);

	if (processed.total() * processed.elemSize() != input_tensor->bytes) {
		throw runtime_error("Input size mismatch: image " + to_string(processed.total() * processed.elemSize()) + " bytes, tensor " + to_string(input_tensor->bytes) + " bytes");
	}

	memcpy(input_tensor->data.raw, processed.data, input_tensor->bytes);

	return processed;
}

struct OutputInfo {
	ModelDataType data_type;
	float scale;
	double zero_point;
	int64_t size;
	const void *data;
};

OutputInfo read_output(unique_ptr<Interpreter> &interpreter) {
	const TfLiteTensor *output_tensor = interpreter->tensor(interpreter->outputs()[0]);
	OutputInfo info = { ModelDataType::FLOAT32, 1.0f, 0.0, 1, output_tensor->data.raw_const };

	/* Get output quantization parameters */
	if (output_tensor->type == kTfLiteInt8 || output_tensor->type == kTfLiteUInt8) {
		info.data_type = (output_tensor->type == kTfLiteInt8) ? ModelDataType::INT8 : ModelDataType::UINT8;
		info.scale = output_tensor->params.scale;
		info.zero_point = output_tensor->params.zero_point;
	} else if (output_tensor->type == kTfLiteFloat32) {
		info.data_type = ModelDataType::FLOAT32;
	} else {
		throw runtime_error("Output Data Type not supported");
	}

	for (int i = 0; i < output_tensor->dims->size; i++)
		info.size *= output_tensor->dims->data[i];

	return info;
}

void get_output(unique_ptr<Interpreter> &interpreter, const vector<string> &labels, cv::Mat &image, float inference_time, int inference_runs, bool use_display) {
	OutputInfo output = read_output(interpreter);

	post_process_output(output.size, output.data, output.data_type, output.scale, output.zero_point, labels, image, inference_time, inference_runs, use_display);
}

static volatile sig_atomic_t stop_requested = 0;

static void handle_stop_signal(int) {
	stop_requested = 1;
}

/* True if the user pressed ENTER on the terminal */
static bool enter_pressed() {
	struct pollfd pfd = { STDIN_FILENO, POLLIN, 0 };

	if (poll(&pfd, 1, 0) > 0 && (pfd.revents & POLLIN)) {
		char buffer[64];

		if (read(STDIN_FILENO, buffer, sizeof(buffer)) > 0)
			return true;
	}

	return false;
}

/* Continuous inference on frames from a USB camera until ENTER / Ctrl-C */
void run_usb_mode(unique_ptr<Interpreter> &interpreter, const vector<string> &labels, string device, bool mirror, bool keep_dynamic_framerate, unsigned int average_frames, bool timing, bool demo, bool use_gpu, bool memlog, bool use_display) {
	UsbCamera camera;
	DisplayLayout layout;
	unique_ptr<WaylandDisplay> wayland_display;
	unique_ptr<DemoView> demo_view;
	unique_ptr<GpuPreprocessor> gpu_preprocessor;
	unique_ptr<MemoryLog> memory_log;
	cv::Mat display_window;
	cv::Mat frame;
	cv::Mat rgb_frame;
	int read_failures = 0;
	float fps = 0.0f;
	/* Sliding windows over the last average_frames frames */
	deque<float> inference_times;
	deque<chrono::steady_clock::time_point> frame_times;
	double inference_sum = 0.0;
	unsigned int frames_since_print = 0;
	bool warm_up_done = false;

	if (device.empty()) {
		device = find_usb_camera();

		if (device.empty())
			throw runtime_error("No USB camera found. Connect one or select a device with -d <video device>");
	}

	printf("Using camera: %s\n", device.c_str());

	if (!camera.open(device, keep_dynamic_framerate))
		throw runtime_error("Failed to open camera: " + device);

	if (use_gpu)
		gpu_preprocessor = make_unique<GpuPreprocessor>();

	if (use_display) {
		if (get_display_layout(layout)) {
			wayland_display = make_unique<WaylandDisplay>();
			wayland_display->open(layout.width, layout.height);

			if (demo)
				demo_view = make_unique<DemoView>();
		} else {
			printf("Display not available, printing results to console only!\n");
			use_display = false;
		}
	}

	signal(SIGINT, handle_stop_signal);
	signal(SIGTERM, handle_stop_signal);

	/* A job in the background must not be stopped when it checks the keyboard: reading then just fails */
	signal(SIGTTIN, SIG_IGN);

	printf("Running inference on camera stream. Press the ENTER key (or Ctrl-C) to quit\n");

	/* Size of the model input (width x height), so the camera frame can be resized to it here */
	const TfLiteTensor *input_tensor = interpreter->tensor(interpreter->inputs()[0]);
	const cv::Size model_size(input_tensor->dims->data[2], input_tensor->dims->data[1]);
	cv::Mat small_frame;

	using clock = chrono::steady_clock;
	auto elapsed_ms = [](clock::time_point from, clock::time_point to) { return chrono::duration<double, milli>(to - from).count(); };
	TimingWindow timing_window;

	if (memlog)
		memory_log = make_unique<MemoryLog>();

	while (!stop_requested && !enter_pressed()) {
		if (memory_log)
			memory_log->tick();

		auto loop_start = clock::now();
		clock::time_point captured = loop_start;

		if (!camera.read(frame, &captured)) {
			if (++read_failures >= CAMERA_MAX_READ_FAILURES)
				throw runtime_error("Failed to read frames from camera");

			continue;
		}

		read_failures = 0;

		auto got_frame = clock::now();

		/* End-to-end FPS: frames per second over the last average_frames frame intervals */
		frame_times.push_back(got_frame);

		if (frame_times.size() > (size_t)average_frames + 1)
			frame_times.pop_front();

		if (frame_times.size() >= 2) {
			chrono::duration<float> span = frame_times.back() - frame_times.front();

			if (span.count() > 0.0f)
				fps = (frame_times.size() - 1) / span.count();
		}

		/* Mirror once, right after capture, so the display, the box and the model input all see the same image */
		if (mirror)
			cv::flip(frame, frame, 1);

		auto mirrored = clock::now();

		/*
		 * Classify the centre square of the frame, so the image is not stretched when resized to the model input.
		 * The resize is done here with INTER_LINEAR (fast on the CPU), so set_input only has to quantize the small picture.
		 */
		bool preprocessed = false;

		if (gpu_preprocessor) {
			preprocessed = gpu_preprocessor->process(frame, get_center_crop(frame), model_size, rgb_frame);

			/* If OpenCL fails, continue on the CPU */
			if (!preprocessed)
				gpu_preprocessor.reset();
		}

		if (!preprocessed) {
			cv::resize(frame(get_center_crop(frame)), small_frame, model_size, 0, 0, cv::INTER_LINEAR);
			cv::cvtColor(small_frame, rgb_frame, cv::COLOR_BGR2RGB);
		}

		auto converted = clock::now();

		set_input(interpreter, rgb_frame);

		auto input_set = clock::now();

		if (interpreter->Invoke() != kTfLiteOk)
			throw runtime_error("Failed to run inference");

		auto inferred = clock::now();
		chrono::duration<float, milli> inference_time = inferred - input_set;

		/* Diagnostic: the same frame again, to compare a possibly "cold" first call with a "warm" second call */
		auto inferred_again = inferred;

		if (timing) {
			if (interpreter->Invoke() != kTfLiteOk)
				throw runtime_error("Failed to run inference");

			inferred_again = clock::now();
		}

		OutputInfo output = read_output(interpreter);
		vector<float> scores;
		vector<pair<float, int>> results;

		if (demo_view)
			scores = get_scores(output.size, output.data, output.data_type, output.scale, output.zero_point, labels.size());
		else
			results = get_top_results(output.size, output.data, output.data_type, output.scale, output.zero_point, labels.size());

		auto output_read = clock::now();

		/* Running average of the inference time over the last average_frames frames. The first inference is a warm-up and not counted */
		bool first_frame = !warm_up_done;

		if (!warm_up_done) {
			warm_up_done = true;

			/* The first interval of the frame rate would contain the slow warm-up (and with -g the OpenCL kernel compile), so start counting after it */
			frame_times.clear();
		} else {
			inference_times.push_back(inference_time.count());
			inference_sum += inference_time.count();

			if (inference_times.size() > average_frames) {
				inference_sum -= inference_times.front();
				inference_times.pop_front();
			}
		}

		int inference_samples = (int)inference_times.size();
		float average_inference_time = inference_samples > 0 ? (float)(inference_sum / inference_samples) : 0.0f;

		auto rendered = output_read;
		auto shown = output_read;

		if (use_display) {
			if (demo_view) {
				DemoInfo info;

				info.show_crop_box = true;
				info.smooth_scores = true;
				info.partial_redraw = true;
				info.inference_time = inference_time.count();

				/* The warm-up frame is not counted in the running average, so it must not enter the graph either */
				info.has_inference_sample = !first_frame;
				info.average_inference_time = average_inference_time;
				info.inference_samples = inference_samples;
				info.rate = fps;

				demo_view->render(frame, scores, labels, layout, info, display_window);
			} else
				render_live_window(frame, results, labels, layout, average_inference_time, inference_samples, fps, display_window);

			rendered = clock::now();

			wayland_display->show(display_window);
			shown = clock::now();
		} else if (++frames_since_print >= average_frames) {
			print_live_results(results, labels, average_inference_time, inference_samples, fps);
			frames_since_print = 0;
		}

		if (timing && !first_frame) {
			auto loop_end = clock::now();

			timing_window.add(STEP_WAIT, elapsed_ms(loop_start, got_frame));
			timing_window.add(STEP_MIRROR, elapsed_ms(got_frame, mirrored));
			timing_window.add(STEP_CONVERT, elapsed_ms(mirrored, converted));
			timing_window.add(STEP_INPUT, elapsed_ms(converted, input_set));
			timing_window.add(STEP_INVOKE, elapsed_ms(input_set, inferred));
			timing_window.add(STEP_INVOKE_AGAIN, elapsed_ms(inferred, inferred_again));
			timing_window.add(STEP_OUTPUT, elapsed_ms(inferred_again, output_read));
			timing_window.add(STEP_RENDER, elapsed_ms(output_read, rendered));
			timing_window.add(STEP_SHOW, elapsed_ms(rendered, shown));
			timing_window.add(STEP_LOOP, elapsed_ms(loop_start, loop_end));
			timing_window.add(STEP_AGE_INFERENCE, elapsed_ms(captured, input_set));
			timing_window.add(STEP_AGE_SHOWN, elapsed_ms(captured, shown));
			timing_window.frames++;

			if (timing_window.frames >= average_frames) {
				TimingReportOptions report;

				report.title = "Camera timing | " + device + " | " + to_string(frame.cols) + "x" + to_string(frame.rows) +
				               " | window " + to_string(timing_window.frames) + " frames | " +
				               (use_display ? "display " + to_string(layout.width) + "x" + to_string(layout.height) : string("console only")) +
				               (gpu_preprocessor ? " | resize on GPU" : " | resize on CPU");
				report.show_mirror = mirror;
				report.show_display = use_display;
				report.show_second_call = true;
				/* The measured camera rate; the constant is only used until the first measurement is available */
				report.camera_fps_measured = camera.capture_fps() > 0.0;
				report.camera_fps = report.camera_fps_measured ? camera.capture_fps() : CAMERA_FRAME_RATE;

				print_timing_report(timing_window, report);

				timing_window.reset();
			}
		}
	}

	camera.release();
}

/*
 * Runs inference the given number of times and returns the average time in ms (rounded to 2 decimal places).
 * The very first inference after start-up is a dry run to clear buffers, so it is not included in the average.
 */
static float run_inference_runs(unique_ptr<Interpreter> &interpreter, unsigned int runs, bool &dry_run_done) {
	float total_inference_times = 0.0f;
	unsigned int completed_runs = 0;

	while (completed_runs < runs) {
		auto start_time = chrono::steady_clock::now();

		/* Run inference */
		if (interpreter->Invoke() != kTfLiteOk)
			throw runtime_error("Failed to run inference");

		if (!dry_run_done) {
			dry_run_done = true;
			continue;
		}

		completed_runs++;

		auto stop_time = chrono::steady_clock::now();
		chrono::duration<float, milli> inference_time = stop_time - start_time;

		printf("Inference Run #%u, inference time: %fms\n", completed_runs, inference_time.count());

		total_inference_times += inference_time.count();
	}

	/* Calculate average inference time and round it to 2 decimal places */
	float average_inference_time = round((total_inference_times / runs) * 100.0f) / 100.0f;

	printf("Average Inference time: %gms over %u %s\n", average_inference_time, runs, runs == 1 ? "run" : "runs");

	return average_inference_time;
}

static bool has_image_extension(const string &name) {
	size_t dot = name.find_last_of('.');

	if (dot == string::npos)
		return false;

	string extension = name.substr(dot + 1);

	transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) { return tolower(c); });

	return extension == "bmp" || extension == "jpg" || extension == "jpeg" || extension == "png";
}

/* All supported image files of a directory, sorted by name */
static vector<string> list_images(const string &directory) {
	vector<string> images;
	DIR *dir = opendir(directory.c_str());

	if (!dir)
		return images;

	while (struct dirent *entry = readdir(dir)) {
		string name = entry->d_name;

		if (has_image_extension(name))
			images.push_back(directory + "/" + name);
	}

	closedir(dir);
	sort(images.begin(), images.end());

	return images;
}

/* Demo view settings for a still picture (single image or slideshow): no smoothing, no crop box, no graph */
static DemoInfo make_still_info(const string &name, size_t index, size_t count, float average_inference_time, unsigned int runs) {
	DemoInfo info;

	info.frame_is_rgb = true;
	info.average_inference_time = average_inference_time;
	info.inference_time = average_inference_time;
	info.inference_samples = (int)runs;
	info.time_caption = "NPU inference (avg of " + to_string(runs) + (runs == 1 ? " run)" : " runs)");
	info.rate = average_inference_time > 0.0f ? 1000.0f / average_inference_time : 0.0f;
	info.rate_unit = "inf/s";
	info.rate_caption = "NPU only, same picture";
	info.show_graph = false;

	if (count > 1) {
		info.side_value = to_string(index + 1) + "/" + to_string(count);
		info.side_caption = name;
	} else {
		info.side_value = name;
		info.side_caption = "picture";
	}

	return info;
}

/* Runs inference on every image of the start image's folder in turn until ENTER / Ctrl-C */
void run_slideshow(unique_ptr<Interpreter> &interpreter, const vector<string> &labels, const string &start_image, unsigned int inference_runs, unsigned int seconds, bool demo, bool memlog, bool use_display) {
	DisplayLayout layout;
	unique_ptr<WaylandDisplay> wayland_display;
	DemoView demo_view;
	unique_ptr<MemoryLog> memory_log;
	cv::Mat display_window;
	bool dry_run_done = false;
	bool quit = false;
	size_t failed_in_a_row = 0;

	size_t slash = start_image.find_last_of('/');
	string directory = (slash == string::npos) ? "." : (slash == 0 ? "/" : start_image.substr(0, slash));
	string start_name = (slash == string::npos) ? start_image : start_image.substr(slash + 1);

	vector<string> images = list_images(directory);

	if (images.empty())
		throw runtime_error("No images (.bmp, .jpg, .jpeg, .png) found in " + directory);

	/* Start with the selected image */
	size_t index = 0;

	for (size_t i = 0; i < images.size(); i++) {
		if (images[i].substr(images[i].find_last_of('/') + 1) == start_name)
			index = i;
	}

	if (use_display) {
		if (get_display_layout(layout)) {
			wayland_display = make_unique<WaylandDisplay>();
			wayland_display->open(layout.width, layout.height);
		} else {
			printf("Display not available, printing results to console only!\n");
			use_display = false;
		}
	}

	signal(SIGINT, handle_stop_signal);
	signal(SIGTERM, handle_stop_signal);

	/* A job in the background must not be stopped when it checks the keyboard: reading then just fails */
	signal(SIGTTIN, SIG_IGN);

	printf("Slideshow of %zu image(s) from %s, %u s per image. Press the ENTER key (or Ctrl-C) to quit\n", images.size(), directory.c_str(), seconds);

	if (memlog)
		memory_log = make_unique<MemoryLog>();

	while (!stop_requested && !quit) {
		size_t current = index;
		const string &path = images[current];
		string name = path.substr(path.find_last_of('/') + 1);

		index = (index + 1) % images.size();

		cv::Mat rgb_image = load_image_file(path);

		if (rgb_image.empty()) {
			fprintf(stderr, "Skipping image that cannot be loaded: %s\n", path.c_str());

			if (++failed_in_a_row >= images.size())
				throw runtime_error("None of the images could be loaded from " + directory);

			continue;
		}

		failed_in_a_row = 0;

		printf("\nImage %zu/%zu: %s\n", current + 1, images.size(), name.c_str());

		set_input(interpreter, rgb_image);

		float average_inference_time = run_inference_runs(interpreter, inference_runs, dry_run_done);

		if (use_display) {
			OutputInfo output = read_output(interpreter);

			if (demo) {
				vector<float> scores = get_scores(output.size, output.data, output.data_type, output.scale, output.zero_point, labels.size());

				demo_view.render(rgb_image, scores, labels, layout, make_still_info(name, current, images.size(), average_inference_time, inference_runs), display_window);
			} else {
				vector<pair<float, int>> results = get_top_results(output.size, output.data, output.data_type, output.scale, output.zero_point, labels.size());

				render_static_window(rgb_image, results, labels, layout, average_inference_time, inference_runs,
				                     "Image: " + name + " (" + to_string(current + 1) + "/" + to_string(images.size()) + ")", display_window);
			}

			/* Both window buffers can still be in use by the compositor, so retry for a short while */
			for (int attempt = 0; attempt < 5 && !wayland_display->show(display_window); attempt++)
				wayland_display->poll_events(20);
		} else {
			get_output(interpreter, labels, rgb_image, average_inference_time, inference_runs, false);
		}

		/* With a single image there is nothing to switch to, so keep it until the user quits */
		bool wait_forever = (images.size() == 1);
		auto deadline = chrono::steady_clock::now() + chrono::seconds(seconds);

		while (!stop_requested) {
			if (memory_log)
				memory_log->tick();

			if (enter_pressed()) {
				quit = true;
				break;
			}

			if (!wait_forever && chrono::steady_clock::now() >= deadline)
				break;

			/* Keeps the window responsive while waiting */
			if (wayland_display)
				wayland_display->poll_events(SLIDESHOW_POLL_MS);
			else
				usleep(SLIDESHOW_POLL_MS * 1000);
		}
	}
}

int main(int argc, char **argv) {
	bool use_display = true;
	bool usb_mode = false;
	bool mirror = false;
	bool timing = false;
	bool opencl = false;
	bool demo = false;
	bool keep_dynamic_framerate = false;
	bool slideshow = false;
	bool output_given = false;
	bool memlog = false;
	bool time_given = false;
	unsigned int slide_seconds = DEFAULT_SLIDE_SECONDS;
	bool image_given = false;
	string device_path = "";
	string image_file_path = DEFAULT_IMAGE_FILE_PATH;
	unsigned int inference_runs = DEFAULT_INFERENCE_RUNS;
	cv::Mat rgb_image;
	vector<string> labels = {};
	TfLiteDelegate *ethosu_delegate = nullptr;

	try {
		while (true) {
			int opt, index = 0;

			opt = getopt_long(argc, argv, short_opts, long_opts, &index);

			if (opt == -1)
				break;

			switch (opt) {
				case 'c': {
					use_display = false;
					break;
				}
				case 'u': {
					usb_mode = true;
					break;
				}
				case 'm': {
					mirror = true;
					break;
				}
				case 'a': {
					keep_dynamic_framerate = true;
					break;
				}
				case 'T': {
					timing = true;
					break;
				}
				case 'D': {
					demo = true;
					break;
				}
				case 'g': {
					opencl = true;
					break;
				}
				case 'M': {
					memlog = true;
					break;
				}
				case 's': {
					slideshow = true;
					break;
				}
				case 't': {
					char *end = nullptr;

					errno = 0;
					long long val = strtoll(optarg, &end, 10);

					/* Verify the seconds per image */
					if (errno != 0 || end == optarg || *end != '\0' || val < 1 || val > MAX_SLIDE_SECONDS) {
						fprintf(stderr, "Invalid number of seconds: %s. Please select a number from 1 to %d!\n", optarg, MAX_SLIDE_SECONDS);

						return EXIT_FAILURE;
					} else {
						slide_seconds = (unsigned int)val;
						time_given = true;
						break;
					}
				}
				case 'o': {
					string reason;

					/* The name is put into a command line, so it is checked strictly */
					if (!is_valid_output_name(string(optarg), reason)) {
						fprintf(stderr, "Invalid output name '%s': %s\n", optarg, reason.c_str());

						return EXIT_FAILURE;
					}

					set_output_name(string(optarg));
					output_given = true;
					break;
				}
				case 'd': {
					string reason;

					device_path = string(optarg);

					/* The name is put into a GStreamer pipeline text, so it is checked strictly */
					if (!is_valid_camera_device(device_path, reason)) {
						fprintf(stderr, "Invalid camera device '%s': %s\n", device_path.c_str(), reason.c_str());

						return EXIT_FAILURE;
					}

					break;
				}
				case 'i': {
					image_given = true;
					image_file_path = string(optarg);

					/* Verify the image file exists and can be opened */
					ifstream image_file(image_file_path);

					if (!image_file) {
						image_file.close();
						fprintf(stderr, "Cannot open image file: %s\n", image_file_path.c_str());

						return EXIT_FAILURE;
					} else {
						image_file.close();
						break;
					}
				}
				case 'n': {
					char *end = nullptr;

					errno = 0;
					long long val = strtoll(optarg, &end, 10);

					/* Verify the number of inference runs */
					if (errno != 0 || end == optarg || *end != '\0' || val < 1 || val > UINT_MAX) {
						fprintf(stderr, "Invalid number of inference runs: %s. Please select a number greater than 0!\n", optarg);

						return EXIT_FAILURE;
					} else {
						inference_runs = (unsigned int)val;
						break;
					}
				}
				case OPT_MIN_PERCENT: {
					float percent = 0.0f;

					if (!parse_min_percent(optarg, percent)) {
						fprintf(stderr, "Invalid minimum confidence: %s. Please select a number from 0 to 100!\n", optarg);

						return EXIT_FAILURE;
					}

					set_min_percent(percent);
					break;
				}
				case 'h': {
					print_help(argv[0]);

					return EXIT_SUCCESS;
				}
				default: {
					print_help(argv[0]);

					return EXIT_FAILURE;
				}
			}
		}

		/* Verify the combination of options */
		if (usb_mode && image_given) {
			fprintf(stderr, "Option -i cannot be used together with -u!\n");

			return EXIT_FAILURE;
		}

		if (usb_mode && slideshow) {
			fprintf(stderr, "Options -u and -s cannot be used together!\n");

			return EXIT_FAILURE;
		}

		if (output_given && !use_display) {
			fprintf(stderr, "Option -o needs the display and cannot be used together with -c!\n");

			return EXIT_FAILURE;
		}

		if (demo && !use_display) {
			fprintf(stderr, "Option -D needs the display and cannot be used together with -c!\n");

			return EXIT_FAILURE;
		}

		if (memlog && !usb_mode && !slideshow) {
			fprintf(stderr, "Option -M can only be used together with -u or -s!\n");

			return EXIT_FAILURE;
		}

		if (!slideshow && time_given) {
			fprintf(stderr, "Option -t can only be used together with -s!\n");

			return EXIT_FAILURE;
		}

		if (!usb_mode && (!device_path.empty() || mirror || keep_dynamic_framerate || timing || opencl)) {
			fprintf(stderr, "Options -d, -m, -a, -T and -g can only be used together with -u!\n");

			return EXIT_FAILURE;
		}

		/*
		 * Shows which build is running: the version of the recipe and the date and time of the compiler macros.
		 * In a Yocto build SOURCE_DATE_EPOCH is set (reproducible builds) and the compiler then reports that date, the date of
		 * the newest source file, and not the time of the compilation. It changes when the sources change, not when the same
		 * sources are built again.
		 */
		printf("image-classification %s (%s), build %s %s\n", APP_VERSION, BOARD_NAME, __DATE__, __TIME__);

		/* Without -g, OpenCL is not touched at all (in any mode); with -g it is probed here and reported */
		const bool use_gpu = setup_opencl(opencl);

		if (!usb_mode && !slideshow) {
			/* Load RGB image from file */
			rgb_image = load_image_file(image_file_path);

			if (rgb_image.empty())
				throw runtime_error("Unable to load image file: " + image_file_path);
		}

		/* Create Ethos-U delegate with default delegate options */
		TfLiteExternalDelegateOptions delegate_options = TfLiteExternalDelegateOptionsDefault(DELEGATE_LIBRARY_PATH);

		ethosu_delegate = TfLiteExternalDelegateCreate(&delegate_options);

		if (!ethosu_delegate) {
			fprintf(stderr, "Failed to create Ethos-U delegate\n");

			return EXIT_FAILURE;
		}

		unique_ptr<FlatBufferModel> flatbuffer_model;
		unique_ptr<Interpreter> interpreter;

		if (!set_interpreter(interpreter, flatbuffer_model, MODEL_PATH, *ethosu_delegate))
			return EXIT_FAILURE;

		if (interpreter->AllocateTensors() != kTfLiteOk)
			throw runtime_error("Failed to allocate tensors");

		/* Load labels from file */
		labels = load_labels_file();

		if (labels.empty())
			throw runtime_error("Could not read labels file: /usr/share/image-classification/labels/labels_mobilenet_v2_1.0_224.txt");

		if (usb_mode || slideshow) {
			if (usb_mode)
				run_usb_mode(interpreter, labels, device_path, mirror, keep_dynamic_framerate, inference_runs, timing, demo, use_gpu, memlog, use_display);
			else
				run_slideshow(interpreter, labels, image_file_path, inference_runs, slide_seconds, demo, memlog, use_display);

			/* The interpreter must be destroyed before the delegate it uses */
			interpreter.reset();
			flatbuffer_model.reset();
			TfLiteExternalDelegateDelete(ethosu_delegate);

			return EXIT_SUCCESS;
		}

		set_input(interpreter, rgb_image);

		/* Run inference for the specified number of runs  */
		bool dry_run_done = false;
		float average_inference_time = run_inference_runs(interpreter, inference_runs, dry_run_done);

		/* Post process the results */
		DisplayLayout demo_layout;

		if (demo && use_display && get_display_layout(demo_layout)) {
			OutputInfo output = read_output(interpreter);
			vector<float> scores = get_scores(output.size, output.data, output.data_type, output.scale, output.zero_point, labels.size());
			string name = image_file_path.substr(image_file_path.find_last_of('/') + 1);
			DemoView demo_view;
			cv::Mat demo_window;

			demo_view.render(rgb_image, scores, labels, demo_layout, make_still_info(name, 0, 1, average_inference_time, inference_runs), demo_window);

			printf("\nDisplaying top Image Classification results....\n");
			display_results(demo_layout.width, demo_layout.height, demo_window);
		} else {
			/* Without usable display the results are printed to the console */
			get_output(interpreter, labels, rgb_image, average_inference_time, inference_runs, use_display && !demo);
		}
	} catch (const exception& e) {
		fprintf(stderr, "ERROR: %s\n", e.what());

		if (ethosu_delegate)
			TfLiteExternalDelegateDelete(ethosu_delegate);

		return EXIT_FAILURE;
	}

	if (ethosu_delegate)
		TfLiteExternalDelegateDelete(ethosu_delegate);

	return EXIT_SUCCESS;
}
