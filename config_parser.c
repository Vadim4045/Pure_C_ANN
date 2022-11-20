#include "config_parser.h"


config* ParseCommand(int argc, char *argv[]){

    FILE *file;
    int i, count=0;
    char tmp[BUFF_BIG], tmp1[BUFF_SMALL], tmp2[BUFF_SMALL];
    config* conf = (config*) calloc(1, sizeof(config));
    if(conf == NULL){
        perror("Config memory allocation");
        return NULL;
    }


    file = fopen(argv[1],"r");
    if(file==NULL){
        perror("Config file open");
        free(conf);
        return NULL;
    }

    while (1)
    {
        i=0;
        while(1) 
        {
            tmp[i] = (char)fgetc(file);

            if(tmp[i] != EOF && tmp[i] != '\n'){
                if(tmp[i] != ' '){
                    i++;
                }
            }else{
                break;
            }
        };

        if(tmp[i] == EOF){
            if(count==8+conf->layers){
                fclose(file);
                return conf;
            }else{
                fclose(file);
                free(conf);
                return NULL;
            }
        }

        if(tmp[0]=='#' || tmp[0] == '\n') continue;

        if(tmp[i] == '\n') tmp[i] = '\0';

        strcpy(tmp1,strtok(tmp, "="));
        strcpy(tmp2,strtok(NULL, "="));

        if(tmp1[0] >='0' && tmp1[0] <= '9'){
            conf->layers_counts[atoi(tmp1)] = atoi(tmp2);
            count++;
        }
        
        if(!strcmp(tmp1, "mu")){
            conf->mu = atof(tmp2);
            count++;
        }
        
        if(!strcmp(tmp1, "data_len")){
            conf->data_len = atoi(tmp2);
            count++;
        }
        
        if(!strcmp(tmp1, "data")){
            strcpy(conf->data, tmp2);
            count++;
        }
        
        if(!strcmp(tmp1, "epochs")){
            conf->epochs = atoi(tmp2);
            count++;
        }
        
        if(!strcmp(tmp1, "mode")){
            conf->mode = atoi(tmp2);
            count++;
        }

        if(!strcmp(tmp1, "weights")){
            strcpy(conf->weights, tmp2);
            count++;
        }
        
        if(!strcmp(tmp1, "layers")){
            conf->layers = atoi(tmp2);
            count++;
        }
        
        if(!strcmp(tmp1, "alfa")){
            conf->alfa = atof(tmp2);
            count++;
        } 
    }
}