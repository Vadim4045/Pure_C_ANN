#ifndef H_LAYER
#define H_LAYER

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <pthread.h>

typedef struct _annLayer
{
    unsigned int count;
    struct _annLayer* next;
    double* content;
    double* fallacy;
    double** weights; 
} annLayer;

annLayer* newLayer(int, int);
annLayer* layerMakeContinue(annLayer*, annLayer*);
void layerFP(annLayer*, double);
void layerBP(annLayer*, double, double);
int freeLayer(annLayer*);
void randomWeights(annLayer*);

#endif //H_LAYER