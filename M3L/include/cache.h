#ifndef CACHE_H
#define CACHE_H 1
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <float.h>
#include <string.h>
#include <stdarg.h>
#include "definitions.h"
typedef signed char schar;


class Cache
{
public:
	Cache(int l,cache_size_t size);
	~Cache();

	// request data [0,len)
	// return some position p where [p,len) need to be filled
	// (p >= len if nothing needs to be filled)
	int get_data(const int index, M3LFloat **data, int len);
	
	cache_size_t getsize()
	{
	  return size;
	}
	void swap_index(int i, int j);	
private:
	int l;
	cache_size_t size;
	struct head_t
	{
		head_t *prev, *next;	// a circular list
		M3LFloat *data;
		int len;		// data[0,len) is cached in this entry
	};

	head_t *head;
	head_t lru_head;
	void lru_delete(head_t *h);
	void lru_insert(head_t *h);
};

#endif
