/*
 * Copyright (C) 2026 Renesas Electronics Corp.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#ifndef APP_INFO_H
#define APP_INFO_H

/*
 * Name of the device the application runs on. Shown in the header of the presentation view and in the start-up line.
 * Can be changed when configuring CMake: -DBOARD_NAME="RZ/G3S"
 */
#ifndef BOARD_NAME
#define BOARD_NAME "RZ/G3E"
#endif

/*
 * Version of the application, shown in the start-up line. The Yocto recipe passes its own version
 * (EXTRA_OECMAKE += "-DAPP_VERSION=${PV}"); without it the text below is used.
 */
#ifndef APP_VERSION
#define APP_VERSION "unknown"
#endif

#endif
