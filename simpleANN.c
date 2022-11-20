#include "simpleANN.h"

Ann* NewSimpleANN(int count, int* config, double alfa, const char* weightsFolder)
{
    unsigned int i;

    
    Ann* ann = (Ann*) calloc(1, sizeof(Ann));
    if(ann==NULL)
    {
        return NULL;
    }

    ann->error = 0;
    
    ann->configArr = config;
    ann->layersCount = count;
    ann->weights_folder = weightsFolder;

    ann->innerLayers = (annLayer**) calloc(count, sizeof(annLayer*));
    if(ann->innerLayers==NULL)
    {
        FreeSimpleANN(ann);
        return NULL;
    }

    for(i = 0; i < count; ++i)
    {
        ann->innerLayers[i] = NewLayer(config[i], alfa, i);

        if(ann->innerLayers[i] != NULL && i>0)
        {
            ann->innerLayers[i - 1] = LayerMakeContinue(ann->innerLayers[i - 1], ann->innerLayers[i]);
            if (ann->innerLayers[i - 1] == NULL)
            {
                FreeSimpleANN(ann);
                return NULL;
            }
        }
    }

    if (!ImportStoredWeights(ann, weightsFolder))
        RandomGenerateWeights(ann);

    return ann;
}
    
int SimpleAnnGo(Ann* ann){
    unsigned int i;

    LayerFP(ann->innerLayers[0]);
    
    for(i = 1; i < ann->layersCount - 1; ++i)
    {
        Activate(ann->innerLayers[i]);

        LayerFP(ann->innerLayers[i]);
    }

    SoftmaxActivate(ann->innerLayers[ann->layersCount - 1]);

    return Result(ann->innerLayers[ann->layersCount - 1]);
}

int SimpleAnnLearn(Ann* ann, int res, double mu)
{
    int answer;
    unsigned int j;
    unsigned int good = 0;
    double error = 0;
    double delta;

    answer = SimpleAnnGo(ann);
    if (answer == res)
        ++good;

    
    for (j = 0; j < ann->configArr[ann->layersCount - 1]; ++j)
    {

        if(j == res)
        {
            delta = 1 - ann->innerLayers[ann->layersCount - 1]->content[j];
        }
        else
        {
            delta = -ann->innerLayers[ann->layersCount - 1]->content[j];
        }

        error += delta * delta;
        
        ann->innerLayers[ann->layersCount - 1]->loss[j] = delta;
    }
    
    for (j = 0; j < ann->configArr[ann->layersCount - 1]; ++j){
        ann->innerLayers[ann->layersCount - 1]->loss[j] 
            *= SoftDerivative(ann->innerLayers[ann->layersCount - 1] 
            , ann->innerLayers[ann->layersCount - 1]->content[j]);
    }

    error /= ann->innerLayers[ann->layersCount - 1]->count;
    
    ann->error += error;
    
    AnnBP(ann, mu*error);

    return good;
}

void AnnBP(Ann* ann, double mu){
    int i;
    double _mu = mu;
    
    for(i = ann->layersCount - 2; i >= 0; --i)
    {
        LayerBP(ann->innerLayers[i], _mu);
        _mu *= 2;
    }
}

int CheckNumber(Ann* ann, double* data)
{
    unsigned int i;

    for(i = 0; i < ann->configArr[ann->layersCount-1]; ++i)
        if(data[i] == 1.0)
            return i;

    return -1;
}

int FreeSimpleANN(Ann* ann){
    unsigned int i;

    if(ann==NULL){
        return 0;
    }

    for(i=0;i<ann->layersCount;i++){
        if(ann->innerLayers[i] != NULL){
            FreeLayer(ann->innerLayers[i]);
        }
    }

    free(ann->innerLayers);
    free(ann);

    return 1;
}

int ImportStoredWeights(Ann* ann, const char* directory){
    unsigned int i, j, k;
    FILE *file;
    char* weightsFileName;

    if(directory==NULL){
        directory="./";
    }

    weightsFileName = GetLastWeightsFileName(ann, directory);
 
    if(weightsFileName == NULL)
    {
        return 0;
    }
        
    file = fopen(weightsFileName,"rb");

    if(file == NULL){
        free(weightsFileName);
        return 0;
    }

    for(i=0;i<ann->layersCount-1;i++)
    {
        for(j=0;j<ann->innerLayers[i]->next->count;j++)
        {
            for (k = 0; k < ann->innerLayers[i]->count + 1; k++)
            {
                fread(&ann->innerLayers[i]->weights[j][k], sizeof(double), 1, file);
            }
        }
    }

    fclose(file);
    
    printf("Loaded stored weights from %s\n", weightsFileName);

    free(weightsFileName);

    return 1;
}

char* GetLastWeightsFileName(Ann* ann, const char* directory){
    int i, epoch=0;
    DIR *d;
    struct dirent *dir;
    char confStr[32], tmp[7], tmp2[50], maxFile[50];
    char* weightsFileName;

    weightsFileName = (char*) calloc(50, sizeof(char));
    if(weightsFileName==NULL) return NULL;

    strcpy(weightsFileName, directory);

    strcpy(confStr, "_");
    for(i = 0; i < ann->layersCount; ++i){
        snprintf(tmp, sizeof(tmp), "%d_", ann->configArr[i]);
        strcat(confStr, tmp);
    }

    d = opendir(directory);
    if (d) {
        maxFile[0] = '\0';

        while ((dir = readdir(d)) != NULL) {

            if(strstr(dir->d_name, confStr) != NULL && strstr(dir->d_name, ".bin") != NULL){
                strcpy(tmp2,dir->d_name);
                strcpy(tmp,strtok(tmp2, "_"));

                if(atoi(tmp)>epoch){
                    epoch = atoi(tmp);
                    strcpy(maxFile, dir->d_name);
                }
            }
        }
        closedir(d);
    }

    if(maxFile[0] != '\0') {
        strcat(weightsFileName, maxFile);

        ann->epoch = epoch;

        return weightsFileName;
    }


    free(weightsFileName);

    return NULL;
}

void RandomGenerateWeights(Ann* ann){
    int i;

    srand(time(NULL));
    
    for(i = 0; i < ann->layersCount - 1; ++i){
        RandomWeights(ann->innerLayers[i]);   
    }

    printf("Random generated weights.\n");
}

void ExportStoredWeights(Ann* ann, int epoch, int dataSetLength, int good){
    unsigned int i, j,k;
    FILE *file;
    char buffer[100], confStr[32], tmp[7];
    struct stat st = {0};

    strcpy(confStr, "_");
    for(i=0;i<ann->layersCount;i++){
        snprintf(tmp, sizeof(tmp), "%d_", ann->configArr[i]);
        strcat(confStr, tmp);
    }

    if (stat(ann->weights_folder, &st) == -1) {
        mkdir(ann->weights_folder, 0700);
    }

    snprintf(buffer, sizeof(buffer), "%s%.4d%s%.5d_%.5d.bin", ann->weights_folder, ann->epoch+epoch, confStr, dataSetLength, good);

    file = fopen(buffer,"wb+");
    if(file==NULL){
        return;
    }

    for(i = 0; i < ann->layersCount - 1; ++i)
    {
        for(j = 0; j < ann->innerLayers[i]->next->count; ++j)
        {
            for (k = 0; k < ann->innerLayers[i]->count + 1; ++k)
            {
                fwrite(&ann->innerLayers[i]->weights[j][k], sizeof(double), 1, file);
            }
                
        }
    }

    fclose(file);

    // snprintf(buffer, sizeof(buffer), "%s%.4d%s%.5d_%.5d.txt", ann->weights_folder, ann->epoch+epoch, confStr, dataSetLength, good);

    // file = fopen(buffer,"w+");
    // if(file==NULL){
    //     return;
    // }

    // for(i=0;i<ann->layersCount-1;i++){
    //     for(j=0;j<ann->innerLayers[i]->next->count;j++){
    //         for(k=0;k<ann->innerLayers[i]->count+1;k++){
    //             fprintf(file, "%f, ", ann->innerLayers[i]->weights[j][k]);
    //         }
    //         fprintf(file, "\n");
    //     }
    //     fprintf(file, "\n");
    // }

    // fclose(file);

}

void SetInput(Ann *ann, double *arr)
{
    SetInArr(ann->innerLayers[0], arr);
}