# image-classification (Renesas RZ/G3E)

Image classification with MobileNet V2 (8-bit, 224x224 input, 1000 ImageNet classes) on the Arm Ethos-U55 NPU. The network is compiled with Vela and run through the TensorFlow Lite external delegate. Up to five classes are shown with their confidence in percent, but only those with at least 1 percent (the option `--min-percent <percent>` changes the limit, `0` lists every class above 0 percent): the output of the model is an int8 softmax with steps of 0.39 percent, so everything below 0.2 percent is stored as exactly 0, and the order among those classes means nothing (they would be arbitrary class numbers). If no class reaches 1 percent, the program says so.

Three modes:

| Mode | What it does |
|---|---|
| single picture (default) | Classifies one picture several times, reports the average inference time, shows the result on the display until ENTER is pressed (or prints it with `-c`). |
| slideshow (`-s`) | Classifies all pictures (`.bmp`, `.jpg`, `.jpeg`, `.png`) of a folder one after the other, in a loop. A picture that cannot be loaded is skipped with a message; a folder with one picture stays on the display. |
| camera (`-u`) | Classifies the live picture of a USB camera (UVC, 640x480) continuously. |

On top of each mode there is an optional presentation view (`-D`).

## Quick start

    image-classification -c                   # default picture, result on the console
    image-classification -u -D                # live camera in the presentation view
    image-classification -s -D -t 10          # slideshow in the presentation view, 10 s per picture
    image-classification -D -i /path/x.jpg    # one picture in the presentation view, closes with ENTER
    image-classification -h                   # all options

Quit the camera mode and the slideshow with ENTER or Ctrl-C. A manual page is included (`man image-classification`, see below).

## Options

| Option | Meaning |
|---|---|
| `-c`, `--console` | Print results to the console only, no display. |
| `-i`, `--image <file>` | Picture to classify. With `-s`: the folder of this file is used and the slideshow starts with this picture. Default: `/usr/share/image-classification/images/cat.bmp` |
| `-n`, `--number <count>` | Inference runs per picture (at least 1); the average time is reported (the first run after start-up is a dry run and not counted). With `-u`: number of frames over which the inference time and the frame rate are averaged, and the number of frames between two timing reports or console results. Default: 10 |
| `--min-percent <percent>` | The smallest confidence, 0 to 100, at which a class is listed. The model gives its confidence in steps of 0.39 percent, so what it gives below about 0.2 percent is stored as 0 and the order of such classes means nothing. `0` lists every class above 0 percent and gives exactly the output of the original program: the five best, without those at 0 percent, equal values with the higher class number first. If no class reaches the limit the program says so. Default: 1 |
| `-s`, `--slideshow` | Slideshow of the folder of the selected picture. |
| `-t`, `--time <seconds>` | Seconds per picture in the slideshow, 1 to 3600. Default: 5 |
| `-u`, `--usb` | Camera mode. |
| `-d`, `--device <device>` | Camera device, for example `/dev/video0`. Default: the first USB video capture device. Only names below `/dev/` with letters, digits and `/ _ . : -`, up to 128 characters, are accepted, and the device must exist. |
| `-m`, `--mirror` | Mirror the camera picture horizontally (the shown and the classified picture). |
| `-a`, `--allow-dynamic-framerate` | Keep the camera's dynamic frame rate. By default the program switches it off (UVC control `exposure_dynamic_framerate`), because many cameras then lower the frame rate in dim light, for example from 30 to 15 fps. The picture can be darker in dim light with the default. |
| `-g`, `--opencl` | Use the GPU (OpenCL) for crop, resize and colour conversion of the camera picture. Falls back to the CPU if OpenCL is not available. |
| `-T`, `--timing` | Print a table with the time of every processing step of the camera loop (see below). Runs a second inference per frame, so the frame rate is lower. For measurements, not for permanent use. |
| `-D`, `--demo` | Presentation view (see below). Needs the display. |
| `-o`, `--output <name>` | Display output (Weston output) used for the size and the window, for example `HDMI-A-1` (default), `DSI-1`, `DVI-D-1`. Names: letters, digits and `. _ -`, up to 32 characters. |
| `-M`, `--memory` | Print the memory use once a minute (camera mode and slideshow). |
| `-h`, `--help` | Help. |

Rules: `-u` cannot be combined with `-i` or `-s`. `-d`, `-m`, `-a`, `-T` and `-g` need `-u`. `-t` needs `-s`. `-M` needs `-u` or `-s`. `-D` and `-o` need the display, so they cannot be used with `-c`. A violation is reported and the program ends with exit status 1. The exit status is 0 after a normal end (including quitting with ENTER or Ctrl-C, and `-h`) and 1 after an error.

## Display

- The program draws full screen through Weston (Wayland). Displays from 800x600 to 1920x1080 pixels are supported; the layout (text size, margins, picture area) adapts. For any other size, or if no display is found, the results are printed to the console.
- The size is read with `wayland-info` from one output, and the window is made fullscreen on that same output. Default output: `HDMI-A-1`; select another with `-o`. List the names with `wayland-info | grep "name: '"`. If the output is not found, the available names are printed and the results go to the console. Placing the window on the named output needs `wl_output` version 4 in the compositor; with an older one the compositor chooses the display and a message says so.
- With a single picture, the program waits for ENTER on its standard input. Without a terminal (for example when it is started automatically) it ends at once; use the camera mode or the slideshow in that case.

## Presentation view (`-D`)

Header in Renesas blue with the device name, the picture, the best match in large letters with a confidence bar, up to four next best matches (from 1 percent on), and a bottom bar with headline numbers. If no class reaches 1 percent the panel shows "--" and says that the picture is not clear for the model; the console and the plain view print "No classifications identified!".

- **Colour of the best match:** green from 51 percent, amber from 20 to 51 percent, grey and marked "uncertain" below 20 percent. With the camera, the scores are averaged over the last few frames (about five) so the name does not flicker between similar classes, and green is kept until the confidence falls below 46 percent.
- **Camera:** the four corners of the centre square that is classified are drawn on the picture. The bottom bar shows the average inference time, the frame rate, and a graph of the inference time: a line of the newest frames (about 3 seconds) and, on displays about 950 pixels wide and more, a histogram of the last 300 frames (about 10 seconds) in the same vertical scale. The range of the graph follows the fastest and the slowest frame of the last 10 seconds and narrows back slowly. The caption gives the range of "most" frames (the middle 90 percent) and, where there is room, the slowest frame ("worst"). Text, bars and graph are refreshed about ten times per second; the picture every frame.
- **Slideshow and single picture:** no smoothing, no corner marks, no graph. The second headline number is the number of inferences per second (NPU only, the same picture repeated), and the bottom right shows the picture number and the file name.
- The limits, the graph settings and the colours are constants at the top of `demo.h` and `demo.cpp` (`DEMO_UNCERTAIN_PERCENT`, `DEMO_CONFIDENT_PERCENT`, `DEMO_HISTORY_FRAMES`, ...); the default of the smallest confidence that is listed is `TOP_RESULTS_MINIMUM_PERCENT` in `processing.h`; the option `--min-percent` overrides it. The device name in the header (`BOARD_NAME`) is in `app_info.h`. The text is drawn with OpenCV's built-in font; the header shows the name "Renesas" as text, not the logo.

## Camera mode

- The camera is opened with GStreamer at 640x480: raw video first, then MJPEG, then the camera's default format; OpenCV's V4L2 backend is the last choice. A separate thread reads the camera and keeps only the newest frame, so a slow display or inference does not build up a queue of old pictures. The measured frame rate of the camera is used in the timing report.
- Only the centre square of the picture is classified, so the picture is not stretched.
- The first inference after start-up is a warm-up (the NPU and, with `-g`, the GPU kernels are prepared). It is not counted in the running average, the timing table, the graph with its caption, or the frame rate.
- If the camera stops delivering frames (for example it is unplugged), the program ends with an error within about 5 seconds. If no camera is found at start, it says so and ends.

### Timing report (`-T`)

Every `-n` frames a table with the average, minimum, maximum and share of the loop of each step: waiting for the frame, crop + resize + colour conversion, quantize + model input, inference (first and second call), reading the output, rendering and sending to the display. The summary gives the processing time per frame, the load against the frame time of the camera, and the age of the frame. Colours are used only on a terminal (not with `NO_COLOR`); the bars use block characters only in a UTF-8 locale.

### GPU (`-g`)

Without `-g`, OpenCL is not touched: the library is not loaded and no GPU context is created. With `-g`, the program probes OpenCL, the start-up line says which device is used (or why it cannot be used), and the crop, resize and colour conversion run on the GPU. This needs an OpenCV built with OpenCL and the Mali OpenCL driver; see `yocto/README_OPENCV_BBAPPEND.md`. It is optional: everything works on the CPU.

## Memory log (`-M`)

One line per minute, with the header repeated every 30 lines:

    Memory         | resident memory (kB)                 | malloc heap (kB)           | process
    Memory  minute |    VmRSS  RssAnon  RssFile  RssShmem |   in use     free  mmapped | threads files
    memory     1.0 |    97956    26776    63972      7208 |    10633     3266     4660 |      13    25

Data lines start with `memory`, so `grep '^memory' log` selects them. How to read a long run: "in use" growing means memory that is allocated and not freed; "free" growing with "in use" flat means heap fragmentation; RssAnon growing with all malloc columns flat means memory that does not come from malloc (drivers, GStreamer buffer pools); growing threads or files mean handles that are not released.

## Version and build stamp

The first line printed shows the version, the device name and a build stamp, for example `image-classification 3.0 (RZ/G3E), build Nov  9 2026 06:47:26`. The version is the version of the Yocto recipe (passed to CMake as `APP_VERSION`; "unknown" if it is not passed). The date is the compiler's `__DATE__` and `__TIME__`: in a Yocto build `SOURCE_DATE_EPOCH` is set for reproducible builds, so it is the date of the newest source file, not the time of the compilation. It changes when the sources change, not when the same sources are built again. `APP_VERSION` and `BOARD_NAME` can be set when configuring CMake (`-DBOARD_NAME="RZ/G3S"`); the defaults are in `app_info.h`.

## Running unattended

The camera mode and the slideshow check the keyboard for ENTER; in the background that read fails quietly. For a long unattended run, give the program no terminal and stop it with a signal:

    ./image-classification -u -D -M < /dev/null > memory.log 2>&1 &
    kill -INT %1

## Requirements on the target

- Ethos-U Linux driver stack (`/dev/accel/accel0`), the delegate library `/usr/lib/libethosu_op_delegate.so.1`, the compiled model and the labels (paths below).
- OpenCV with the GStreamer backend (libraries core, imgcodecs, imgproc, videoio), TensorFlow Lite, `wayland-client`.
- GStreamer elements `v4l2src`, `videoconvert`, `appsink`, and `jpegdec` for the MJPEG fallback.
- Weston with xdg-shell; `wayland-info` (package `wayland-utils`) for the display size.
- Optional: OpenCV with OpenCL and the Mali library for `-g`.

Fixed paths: model `/usr/share/models/mobilenet_v2/mobilenet_v2_1.0_224_INT8/mobilenet_v2_1.0_224_INT8_vela.tflite`, labels `/usr/share/image-classification/labels/labels_mobilenet_v2_1.0_224.txt`, pictures `/usr/share/image-classification/images/`. They are constants in `main.cpp` and `processing.h`; there is no option to change the model yet, and the texts "MobileNet V2" and "Ethos-U55" in the presentation view, and the operator counts in the information panel of the plain view, are fixed.

## Build

The C++ application is built with CMake (`pkg-config` finds OpenCV; TensorFlow Lite and `wayland-client` are linked). In the Yocto (scarthgap) build it is the recipe `image-classification_3.0.bb`, which fetches the folder `c++/` (the contents of this folder). The `yocto/` folder next to this folder has the changed recipe, the OpenCV bbappend for OpenCL, a kas overlay for the man page, and `YOCTO.md`, which describes all steps.

## Source files

| File | Purpose |
|---|---|
| `main.cpp` | Options, start-up, and the three modes: single picture, slideshow and the camera loop. |
| `processing.cpp`, `processing.h` | Labels, picture loading, resize and quantizing for the model, the plain result view and the drawing helpers shared with the presentation view. |
| `demo.cpp`, `demo.h` | Presentation view (`-D`): layout, colours, best match, graph and histogram. |
| `graphstats.cpp`, `graphstats.h` | Percentile, histogram bins and the range of the graph (no OpenCV needed). |
| `camera.cpp`, `camera.h` | USB camera discovery, check of the device name, and the capture thread that hands over the newest frame. |
| `timing.cpp`, `timing.h` | Timing table of the camera loop (`-T`). |
| `memlog.cpp`, `memlog.h` | Memory log (`-M`). |
| `opencl_utils.cpp`, `opencl_utils.h` | GPU crop, resize and colour conversion (`-g`). |
| `wayland_utils.cpp`, `wayland_utils.h` | Display size and output selection (`-o`), the one-shot display for a single picture, and the live display with two buffers. |
| `app_info.h` | Device name and version (defaults, can be set when configuring CMake). |
| `image-classification.1` | Manual page. |
| `CMakeLists.txt` | Build; installs the program, the manual page and this file. |

## Typical figures

Measured on an RZ/G3E SMARC board with a Logitech C922 (640x480, 30 fps), Weston at 800x600 to 1280x720. They vary with the setup.

| Item | Value |
|---|---|
| Inference time, single picture or slideshow | about 8 ms |
| Inference time in the camera mode | about 9 to 10 ms (the camera, the display copy and the drawing compete for the memory) |
| Crop + resize + colour conversion | about 4 ms on the CPU, about 2.2 ms with `-g` |
| Drawing of the presentation view | about 8 to 10 ms per frame on average (about 5 to 6 ms for a frame that only redraws the picture) |
| Processing per frame (camera, without `-T`) | about 22 to 25 ms of the 33.3 ms frame time at 30 fps |
| Memory (`-u -D -g`, 1280x720) | constant for 52 minutes: RssAnon 26.8 MB, malloc heap 13.9 MB |

With the dynamic frame rate left on (`-a`), the C922 dropped from 30 to 15 fps in dim light.

## Messages

| Message | Meaning |
|---|---|
| `No USB camera found. Connect one or select a device with -d <video device>` | No USB video capture device. |
| `Invalid camera device '...': ...` | The name given with `-d` is refused (see the rules above). |
| `Failed to open camera: ...` / `Failed to read frames from camera` | The camera could not be opened, or it stopped delivering frames. |
| `Display output '...' not found or its size is unknown (available: ...)` | The output named with `-o` (default `HDMI-A-1`) does not exist; the available names are listed. |
| `Unsupported display resolution: WxH` | Outside 800x600 to 1920x1080; the results go to the console. |
| `OpenCL: not available in this OpenCV build or driver, the CPU is used` | `-g` has no effect. |
| `[ WARN ... GStreamer warning: Cannot query video position ...]` | Printed by OpenCV when it opens the camera pipeline; harmless. |

## Known limitations

- The model and the labels are fixed (see above).
- During some long runs with `-T` on the test board, resets of the display's USB touch controller were logged in `dmesg` (the touch kept working). The cause has not been determined.

## Documentation

This file is installed to `/usr/share/doc/image-classification/` and the manual page `image-classification.1` to `/usr/share/man/man1/`, both in the package `image-classification-doc`. Read the manual page with `man image-classification` on a target that has a man program, or with `man -l image-classification.1` on a PC. This file is the complete description of the options and the presentation view; the manual page and the output of `-h` are shorter forms of it.

## License

Copyright (C) 2026 Renesas Electronics Corp. Licensed under the Apache License, Version 2.0.
