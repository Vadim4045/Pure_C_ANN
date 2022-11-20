#include "annLayer.h"

annLayer* NewLayer(int count, double alfa, int position)
{

    annLayer* newLayer = (annLayer*) calloc(1, sizeof(annLayer));
    if (newLayer==NULL)
    {
        return NULL;
    }

    newLayer->count = count;
    newLayer->alfa=alfa;

    newLayer->content = (double*) calloc(count, sizeof(double));
    if(newLayer->content == NULL){
        FreeLayer(newLayer);
        return NULL;
    }
    
    if(position>0)
    {
        newLayer->loss = (double*) calloc(count, sizeof(double));
        if(newLayer->loss == NULL)
        {
            FreeLayer(newLayer);
            return NULL;
        }
    }
    else newLayer->loss = NULL;

    return newLayer;
}

annLayer* LayerMakeContinue(annLayer *layer, annLayer *nextLayer)
{

    unsigned int i;

    layer->next = nextLayer;

    layer->weights = (double**) calloc(nextLayer->count, sizeof(double*));
    if(layer->weights == NULL){
        FreeLayer(layer);
        return NULL;
    }

    for(i=0;i<nextLayer->count;i++){
        layer->weights[i] = (double*) calloc(layer->count+1, sizeof(double));
        if(layer->weights[i]==NULL){
            FreeLayer(layer);
            return NULL;
        }
    }

    layer->prev_delta = (double **)calloc(nextLayer->count, sizeof(double *));
    if (layer->prev_delta == NULL)
    {
        FreeLayer(layer);
        return NULL;
    }

    for (i = 0; i < nextLayer->count; ++i)
    {
        layer->prev_delta[i] = (double *)calloc(layer->count + 1, sizeof(double));
        if (layer->prev_delta[i] == NULL)
        {
            FreeLayer(layer);
            return NULL;
        }
    }
    
    return layer;
}

void RandomWeights(annLayer* layer)
{
    unsigned int i, j;

    for(i = 0; i < layer->next->count; ++i)
        for(j = 0; j < layer->count + 1; ++j){
            layer->weights[i][j] = (((float)rand()/(float)RAND_MAX) - 0.5) * 0.001;
        }
}

void LayerFP(annLayer* layer)
{
    unsigned int i, j;

    for (i = 0; i < layer->next->count; ++i)
    {
        layer->next->content[i] = layer->weights[i][layer->count];

        for (j = 0; j < layer->count; ++j)
        {
            layer->next->content[i] += layer->content[j] * layer->weights[i][j];
        }
    }

}

void LayerBP(annLayer *layer, double mu)
{
    unsigned int i,j;

    if (layer->loss != NULL)
    {
        for(i = 0; i < layer->count; ++i)
            layer->loss[i] = 0;

        for (i = 0; i < layer->next->count; ++i)
            for(j=0;j<layer->count;j++)
                layer->loss[j] += layer->weights[i][j] * layer->next->loss[i] * SigmaPrime(layer->content[j], layer->alfa);
    }

    for(i = 0; i < layer->next->count; ++i)
    {

        double delta = mu * layer->next->loss[i];

        layer->weights[i][layer->count] += delta + 0.1 * layer->prev_delta[i][layer->count];

        layer->prev_delta[i][layer->count] = delta;

        for (j = 0; j < layer->count; ++j)
        {
            delta = mu * layer->next->loss[i] * layer->content[j];

            layer->weights[i][j] += delta + 0.1 * layer->prev_delta[i][j];

            layer->prev_delta[i][j] = delta;
        }
    }
}

int FreeLayer(annLayer* layer)
{
    unsigned int i;

    if(layer == NULL) return 0;
 
    if(layer->loss != NULL)
        free(layer->loss);
    
    
    if(layer->content != NULL)
        free(layer->content);

    if(layer->weights != NULL)
    {
        for(i=0;i<layer->next->count;i++)
            if(layer->weights[i] != NULL)
                free(layer->weights[i]);


        free(layer->weights);
    }

    if (layer->prev_delta != NULL)
    {
        for (i = 0; i < layer->next->count; i++)
            if (layer->prev_delta[i] != NULL)
                free(layer->prev_delta[i]);

        free(layer->prev_delta);
    }

    free(layer);

    return 1;
}

void SoftmaxActivate(annLayer *layer)
{
    unsigned int i;

    layer->total_out = 0;

    for (i = 0; i < layer->count; ++i)
    {
        layer->total_out += layer->content[i];
    }

    for (i = 0; i < layer->count; ++i)
    {
        layer->content[i] /= layer->total_out;
    }
}

inline double SoftDerivative(annLayer *layer, double arg)
{
    return (layer->total_out - arg) / (layer->total_out * layer->total_out);
}

inline void Activate(annLayer *layer)
{
    unsigned int i;

    for (i = 0; i < layer->count; ++i)
        layer->content[i] = Sigma(layer->content[i], layer->alfa);
}

inline double Sigma(double num, double alfa)
{
    return ((alfa * num > num) ? alfa * num : num);
}

inline double SigmaPrime(double num, double alfa)
{
    return ((num > 0) ? 1 : -alfa);
}

inline int Result(annLayer *layer)
{
    unsigned int i, idx;
    double max = -1;

    for (i = 0; i < layer->count; i++)
    {
        if (layer->content[i] > max)
        {
            max = layer->content[i];
            idx = i;
        }
    }
    return idx;
}

void SetInArr(annLayer *layer, double *arr)
{
    layer->content = arr;
}