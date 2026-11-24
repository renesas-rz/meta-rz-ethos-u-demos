/*
 * Copyright (C) 2026 Renesas Electronics Corp.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#ifndef MEMLOG_H
#define MEMLOG_H

#include <chrono>

/* Seconds between two lines of the memory log */
#define MEMORY_LOG_INTERVAL_SECONDS 60

/*
 * Prints one line about the memory of the process every interval_seconds (call tick() often, e.g. once per frame):
 *   VmRSS, RssAnon (heap and other private memory), RssFile (libraries, model), RssShmem (shared memory, e.g. the display buffers),
 *   what malloc has in use / free / allocated with mmap, the number of threads and of open file descriptors.
 * Reading it over a long run shows where growth comes from:
 *   "in use" grows              -> memory that is allocated and not freed (a leak, in our code or in a library)
 *   "free" grows, "in use" not  -> the malloc heap gets fragmented
 *   RssAnon grows, malloc flat  -> memory that does not come from malloc (drivers, GStreamer buffer pools)
 *   fds or threads grow         -> handles or threads are not released
 */
class MemoryLog {
public:
	explicit MemoryLog(unsigned int interval_seconds = MEMORY_LOG_INTERVAL_SECONDS);

	void tick();

private:
	void print_line();

	std::chrono::steady_clock::time_point start_;
	std::chrono::steady_clock::time_point last_;
	std::chrono::seconds interval_;
	/* The header is printed before the first line and then every 30 lines */
	unsigned int lines_printed_ = 0;
};

#endif
