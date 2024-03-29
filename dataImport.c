#include "dataImport.h"

double** GetData(const char* fileName, int* len){
    int i, j;
    FILE *file;
	double** res;
	unsigned long fileLen;

	file = fopen(fileName, "rb");
	if (!file)
	{
		fprintf(stderr, "Unable to open file %s", fileName);
		return NULL;
	}

	fseek(file, 0, SEEK_END);
	fileLen=ftell(file)/((unsigned long)*len*sizeof(double));
	fseek(file, 0, SEEK_SET);

	res=(double **)calloc(fileLen, sizeof(double*));
	if (!res)
	{
		fprintf(stderr, "Memory error!");
        fclose(file);
		return NULL;
	}

    for(i = 0; i < fileLen; ++i){
        res[i] = (double*)calloc(*len, sizeof(double));
        if(!res[i]){
            FreeDataMemory(res, fileLen);
            return NULL;
        }

        for(j = 0; j < *len; ++j){
            fread(&res[i][j], sizeof(double), 1, file);
        }
    }
	
	fclose(file);

    *len = fileLen;

    return res;
}

void FreeDataMemory(double** data, int count){
    int i;
    if(!data){
        return;
    }

    for(i = 0; i < count; ++i){
        if(data[i]){
            free(data[i]);
        }
    }

    free(data);
}