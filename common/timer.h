#ifndef TIMER_H
#define TIMER_H
#include <sys/time.h>

static double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}

#endif