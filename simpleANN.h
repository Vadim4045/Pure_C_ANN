#ifndef H_ANN
#define H_ANN
#define NELEMS(x)  (int)(sizeof(x) / sizeof((x)[0]))

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <dirent.h>
#include "annLayer.h"
#include <time.h>
// #include <unistd.h>

typedef struct simpleANN
{    
    int             layersCount;
    int             epoch;
    double          error;
    int*            configArr;
    const char*     weights_folder;
    annLayer**      innerLayers;    
} Ann;

Ann *   NewSimpleANN                (int count, int *config, double alfa, const char *weightsFolder);
int     SimpleAnnGo                 (Ann * ann);
int     ImportStoredWeights         (Ann *ann, const char * directory);
char *  GetLastWeightsFileName      (Ann * ann, const char * directory);
void    RandomGenerateWeights       (Ann *ann);
void    ExportStoredWeights         (Ann *ann, int epoch, int dataSetLength, int good);
int     SimpleAnnLearn              (Ann* ann, int res, double mu);
void    AnnBP                       (Ann * ann, double mu);
int     CheckNumber                 (Ann* ann, double* arr);
void    SetInput                    (Ann* ann, double* arr);
int     FreeSimpleANN               (Ann* ann);


#endif //H_ANN