#ifndef DEFINITIONS_H
#define DEFINITIONS_H
//some definitions





#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <sstream>
# include <ctime>
#include <cassert>
#define K_LIN 0
#define K_RBF 1
#define K_POLY 2
#define ALG_KER 0
#define ALG_LIN 1
#define mymax(a,b) ((a)>(b)?(a):(b))
#define mymin(a,b) ((a)<(b)?(a):(b))
//If it is a windows system define WINDOWS as 1 else as 0
#define WINDOWS 0
typedef float M3LFloat;
typedef long long int cache_size_t;
#if WINDOWS
#include <time.h>
#define isnan _isnan
extern bool time_started;
extern time_t _start, _finish;
inline double getTime() {	
	if(!time_started)
	{
		time(&_start);
		time_started=true;
		return 0;
	}
	
	time(&_finish);
	return difftime(_finish, _start);
}

#else
#include <sys/time.h>
inline double getTime() {
  struct timeval tv;
  struct timezone tz;
  long int sec; 
  long int usec;
  double mytime;
  gettimeofday(&tv, &tz);
  sec= (long int) tv.tv_sec;
  usec= (long int) tv.tv_usec;
  mytime = (double) sec+(double)usec*0.000001;
  return mytime;

}
#endif



#endif