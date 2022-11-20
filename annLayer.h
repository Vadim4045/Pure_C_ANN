#ifndef H_LAYER
#define H_LAYER

#define MIN_PER_THREAD 20

#include <stdio.h>
#include <stdlib.h>

typedef struct _annLayer
{
    unsigned int            count;
    double                  alfa;
    double                  total_out;
    struct _annLayer*       next;
    double*                 content;
    double*                 loss;
    double**                prev_delta;
    double**                weights;
} annLayer;

annLayer*           NewLayer                (int count, double alfa, int position);
annLayer*           LayerMakeContinue       (annLayer *layer, annLayer *nextLayer);
void                LayerFP                 (annLayer *layer);
void                Activate                (annLayer *layer);
void                SoftmaxActivate         (annLayer *layer);
int                 Result                  (annLayer *layer);
void                LayerBP                 (annLayer *, double);
int                 FreeLayer               (annLayer *layer);
void                RandomWeights           (annLayer *layer);
void                SetInArr                (annLayer * layer, double *arr);
int                 NormalaiseWeights       (annLayer *layer);
double              SoftDerivative          (annLayer *layer, double arg);
double              Sigma                   (double num, double alfa);
double              SigmaPrime              (double num, double alfa);

#endif //H_LAYER