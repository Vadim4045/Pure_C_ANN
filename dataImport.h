#ifndef H_DATA_IMPORT
#define H_DATA_IMPORT

#include <stdio.h>
#include <stdlib.h>

double** 	GetData				(const char*, int*);
void 		FreeDataMemory		(double**, int);

#endif //H_DATA_IMPORT