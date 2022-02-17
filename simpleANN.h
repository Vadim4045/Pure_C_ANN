#ifndef H_ANN
#define H_ANN
#define NELEMS(x)  (int)(sizeof(x) / sizeof((x)[0]))

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include "annLayer.h"

typedef struct simpleANN
{
    int layersCount;
    int* configArr;
    annLayer** innerLayers;
    double alfa;
    int epoch;
} Ann;

Ann* newSimpleANN(int, int*, double, const char*);
int simpleAnnGo(Ann*, double*, double*);
int simpleAnnLearn(Ann*, double**, int, double);
int checkNumber(Ann*, double*);
int freeSimpleANN(Ann*);

#endif //H_ANN