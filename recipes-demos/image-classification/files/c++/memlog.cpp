/*
 * Copyright (C) 2026 Renesas Electronics Corp.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>

#include <dirent.h>
#include <malloc.h>

#include "memlog.h"

using namespace std;

/* Value of a "Key:   123 kB" line of /proc/self/status, or -1 */
static long status_value(const char *key) {
	ifstream file("/proc/self/status");
	string line;
	size_t length = strlen(key);

	while (getline(file, line)) {
		if (line.compare(0, length, key) == 0 && line.size() > length && line[length] == ':')
			return atol(line.c_str() + length + 1);
	}

	return -1;
}

static long count_entries(const char *path) {
	DIR *dir = opendir(path);
	long count = 0;

	if (!dir)
		return -1;

	while (struct dirent *entry = readdir(dir)) {
		if (strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0)
			count++;
	}

	closedir(dir);

	/* opendir itself used one descriptor that is listed in /proc/self/fd */
	return count - 1;
}

MemoryLog::MemoryLog(unsigned int interval_seconds)
	: start_(chrono::steady_clock::now()), last_(start_ - chrono::seconds(interval_seconds)), interval_(interval_seconds) {
}

void MemoryLog::tick() {
	auto now = chrono::steady_clock::now();

	if (now - last_ < interval_)
		return;

	last_ = now;
	print_line();
}

void MemoryLog::print_line() {
	double minutes = chrono::duration<double>(chrono::steady_clock::now() - start_).count() / 60.0;
	long in_use;
	long free_bytes;
	long mmapped;

#if defined(__GLIBC__) && (__GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 33))
	struct mallinfo2 info = mallinfo2();

	in_use = (long)(info.uordblks / 1024);
	free_bytes = (long)(info.fordblks / 1024);
	mmapped = (long)(info.hblkhd / 1024);
#else
	struct mallinfo info = mallinfo();

	in_use = info.uordblks / 1024;
	free_bytes = info.fordblks / 1024;
	mmapped = info.hblkhd / 1024;
#endif

	/*
	 * The header and the data lines use the same field widths, so the columns line up.
	 * Data lines start with "memory", header lines with "Memory", so "grep '^memory'" selects the data only.
	 */
	if (lines_printed_ % 30 == 0) {
		printf("\n");
		printf("Memory %7s | %-36s | %-26s | %-13s\n", "", "resident memory (kB)", "malloc heap (kB)", "process");
		printf("Memory %7s | %8s %8s %8s %9s | %8s %8s %8s | %7s %5s\n", "minute",
		       "VmRSS", "RssAnon", "RssFile", "RssShmem", "in use", "free", "mmapped", "threads", "files");
	}

	printf("memory %7.1f | %8ld %8ld %8ld %9ld | %8ld %8ld %8ld | %7ld %5ld\n", minutes,
	       status_value("VmRSS"), status_value("RssAnon"), status_value("RssFile"), status_value("RssShmem"),
	       in_use, free_bytes, mmapped, status_value("Threads"), count_entries("/proc/self/fd"));
	fflush(stdout);

	lines_printed_++;
}
