#ifndef _MAIN
#define _MAIN

#define BUFF_BIG 100
#define BUFF_SMALL 50


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "config_parser.h"
#include "dataImport.h"
#include "simpleANN.h"

double **ShuffleData(double **data, double **newData, int count)
{
    unsigned int i, rnd;

    for (i = 0; i < count; i++)
    {
        rnd = (rand() * count) / RAND_MAX;

        while (newData[rnd] != NULL)
        {
            if (rnd == count - 1)
                rnd = 0;
            else
                rnd++;
        }

        newData[rnd] = data[i];
        data[i] = NULL;
    }

    return newData;
}

int main(int argc, char *argv[])
{
    Ann* ann;

    double** data, **newData, **tmp;
    time_t start_t, end_t;
    double diff_t;
    config* conf;
    int i = 0, j = 0, res, good;

    if(argc == 2){
        conf = ParseCommand(argc, argv);
    }else{
        printf("Not enough parameters.\n");
        exit(0);
    }
    if (conf == NULL){
        printf("Config not readed.\n");
        exit(0);
    }

    data = GetData(conf->data, &conf->data_len);
    if(data == NULL){
        printf("Dataset reading error");
        exit(0);
    }

    newData = (double**)calloc(conf->data_len, sizeof(double*));
    if(newData == NULL){
        printf("Dataset reading error");
        FreeDataMemory(data, conf->data_len);
        exit(0);
    }

    ann = NewSimpleANN(conf->layers, conf->layers_counts, conf->alfa, conf->weights);
    
    if(conf->mode == 0){
        for(i = 1; i <= conf->epochs; ++i){
 
            tmp = ShuffleData(data, newData, conf->data_len);
            newData=data;
            data=tmp;
            good=0;

            time(&start_t);
            
            for(j = 0; j < conf->data_len; ++j){

                res = CheckNumber(ann, (data[j] + conf->layers_counts[0]));
                
                SetInput(ann, data[j]);

                if (SimpleAnnLearn(ann, res, conf->mu)) good++;
            }

            ExportStoredWeights(ann, i, conf->data_len, good);

            time(&end_t);
            
            diff_t = difftime(end_t, start_t);
            
            printf("AVG error = %f\n", ann->error / conf->data_len);
            
            ann->error = 0;
            
            printf("Step - %d from %d: %d/%d on %f sec.\n", i, conf->epochs, conf->data_len, good, diff_t);
        }
    }
    else if(conf->mode == 1){
        time(&start_t);

            good = 0;

            for(j = 0; j < conf->data_len; ++j){
                res = CheckNumber(ann, data[j] + 784);
                
                SetInput(ann, data[j]);

                if(SimpleAnnGo(ann) == res) good++;
            }

            time(&end_t);
            diff_t = difftime(end_t, start_t);

            printf("Step - %d from %d: %d/%d on %f sec.\n", i, conf->epochs, conf->data_len, good, diff_t);
        
    }
    
    FreeSimpleANN(ann);
    FreeDataMemory(data, conf->data_len);

    exit(0);
}

#endif // _MAIN