#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
// #include <omp.h>:
int sum(int arr[], int n)
{
    int sum = 0;

    for (int i = 0; i < n; i++)
    sum += arr[i];

    return sum;
}
double sum_double (double arr[], int n)
{
    double sum = 0;

    for (int i = 0; i < n; i++)
    sum += arr[i];

    return sum;
}
struct MinId{
    float yval;
    int id;
};
struct MinId findmin(double arr[], int len)
{
    struct MinId minid;
    minid.yval = arr[0];
    minid.id = 0;
//    double TOL = 0.00000001;
    for (int c = 1 ; c< len; c++ )
    {

        if (  arr[c] < minid.yval )
        {
            minid.yval = arr[c];
           minid.id = c;
//            printf("%d,", minid.id);
        }
    }

    return minid;
};

struct ArraySize{
    int Labels , Weights, Rates  ;
    int Ordr , IOrdr , IOrdrMat  , OrdrMat;
    int Walk  , ActvMat  , MeasVec , MeasVec2 ;
    int TotAVec , Errors ;
};
int main(int argc, char *argv[])
{
//    time_t t;
//    srand((unsigned) time(&t));
    int trial_id = atoi(argv[3]);
    srand(trial_id);
    struct ArraySize array_size;
//    const int N = atoi(argv[1]);
    const int N =  atoi(argv[1]);
    const int P =atoi(argv[2]);
    const double Marg = 0.000001;
    const int MaxIter = 10000;
    // printf("N%d P %d Trial %d\n", N, P, trial_id);
    struct MinId minid;
    int *Labels = (int *)malloc(sizeof(int)*(P));
//    int Imb =  sum(Labels, P);
    double *Weights = (double *)malloc(sizeof(double)*(P));
    double *Rate = (double *)malloc(sizeof(double)*(P));
    // printf("%ld\n",sizeof(int));
    array_size.Labels =  (sizeof(int)*(P)/sizeof(int));
    array_size.Weights = (sizeof(double)*(P)/sizeof(int));
    array_size.Rates =  (sizeof(double)*(P)/sizeof(int));

    int *Ordr = (int *)malloc(sizeof(int)*P);
    int *IOrdr = (int *)malloc(sizeof(int)*P);

    int **IOrdrMat;
    int **OrdrMat;
    int FLAG = 1;

    array_size.Ordr = (sizeof(int)*(P)/sizeof(int));
    array_size.IOrdr = (sizeof(int)*(P)/sizeof(int));

    double *Walk =(double *) malloc(sizeof(double)*(P+1));
    array_size.Walk =  (sizeof(double)*(P+1)/sizeof(int));


    double **ActvMat;
    int oldx = P;
    int newx = P;
    double yval = 0;
   double lmbdp = 0.01; // Note that the previous case corresponds
   double lmbdm = 0.01; // to lmbdp=lmbdm=0.5 and attrp=attrm=0.
   double attrp = 1;
   double attrm = 1;


    double *MeasVec;
    double *MeasVec2;


    double *TotAVec;
    double *AVec;

    double Thresh;

    int *Errors;
    int NumError;
//    struct rusage r_usage;

    int count = 0;

    clock_t start, end;
    double cpu_time_used;
    double old_cpu_time_used;
    start = clock();

    OrdrMat = (int**)malloc(sizeof(int*)*N);
    IOrdrMat = (int**)malloc(sizeof(int*)*N);
    ActvMat = (double**)malloc(sizeof(double*)*N);
    for (int i = 0; i < N; ){
         OrdrMat[i] = (int*)malloc(sizeof(int)*P);
        IOrdrMat[i] = (int*)malloc(sizeof(int)*P);
        ActvMat[i] = (double*)malloc(sizeof(double)*P);
        i++;
    }
    memset(Ordr, 0, sizeof(int)*P);
    memset(IOrdr, 0, sizeof(int)*P);
    array_size.OrdrMat = (sizeof(int)*P*N/sizeof(int))+ sizeof(int*)*N/sizeof(int)+ sizeof(int**);
    array_size.IOrdrMat = (sizeof(int)*P*N/sizeof(int))+ sizeof(int*)*N/sizeof(int)+ sizeof(int**);
    array_size.ActvMat =  (sizeof(double)*P*N/sizeof(int))+ sizeof(double*)*N/sizeof(int)+ sizeof(double**);
    TotAVec = (double*)malloc(sizeof(double)*P);
    AVec = (double*)malloc(sizeof(double)*P);
    Errors = (int*)malloc(sizeof(int)*P);
    array_size.TotAVec =  (sizeof(double)*P/sizeof(int)) + sizeof(double*)/sizeof(int);
    array_size.Errors =  (sizeof(int)*P/sizeof(int))+ sizeof(double*)/sizeof(int);

//    FILE *fp = NULL;
//    fp = fopen("D:/mono-syn/monsyn_stats/N1000_P18000_trial1.txt",  "r");
//    if( fp == NULL ) {
//           printf("Error while opening the file.\n");
//           return -1;
//       }
//    int i = 0;
//    int col = 0;
//    int row = 0;
//    int num = 0;
//    while( (fscanf(fp, "%d,", &num) != EOF) && i < N * P) {
//
//        row = i/P;
//        col = i%P;
//        printf("row %d, col %d\n", row, col);
//           OrdrMat[row][col] = num-1;
//        i++;
//       }
//    fclose(fp);

    // BEGIN memory usage by variable

    // printf("array_size.Weights %d kB\n", array_size.Weights/256);
    // printf("array_size.Labels %d kB\n", array_size.Labels/256);
    // printf("array_size.Rates %d kB\n", array_size.Rates/256);
    // printf("array_size.Ordr %d kB\n", array_size.Ordr/256);
    // printf("array_size.IOrdr %d kB \n", array_size.IOrdr/256);
    // printf("array_size.OrdrMat %d kB\n", array_size.OrdrMat/256);
    // printf("array_size.IOrdrMat %d kB\n", array_size.IOrdrMat/256);
    // printf("array_size.ActvMat %d kB\n", array_size.ActvMat/256);
    // printf("array_size.TotAVec %d kB\n", array_size.TotAVec/256);
    // END memory usage by variable
#pragma omp parallel for
    for (int i = 0; i< P; i++){
        Weights[i] = 1;
        Rate[i] = 0;
        if( i< P/2 ) Labels[i] = 1;
        else Labels[i] = -1;
         
    }

//    getrusage(RUSAGE_SELF,&r_usage);// Print the maximum resident set size used (in kilobytes).
//    printf("Memory usage: %ld kilobytes\n",r_usage.ru_maxrss);
//#pragma omp parallel for
    for (int i = 0; i< N; i++){

        for (int j  = 0; j < P; j++){
            Ordr[j] = j;
        }
        for (int j  = 0; j < P; j++){
      
            int k, t;
            k = rand() % (P - j) + j;
            t = Ordr[k]; Ordr[k] = Ordr[j]; Ordr[j] = t; IOrdr[Ordr[j]] = j;
           
        }

        for (int j  = 0; j < P; j++) {
            OrdrMat[i][j] = Ordr[j];
            IOrdrMat[i][j] = IOrdr[j];
          
        }
        /* debug */

//         for (int j  = 0; j < P; j++){
//             IOrdrMat[i][OrdrMat[i][j]] = j;
//         }

        

    }


    free(Ordr);
    free(IOrdr);
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    // printf("generate OrdrMat took %f seconds \n", cpu_time_used);
    old_cpu_time_used = cpu_time_used;
#pragma omp parallel for
    for (int i = 0; i< N; i++){

        memset(ActvMat[i], 0, sizeof( double)*(P));
        }
    for (int k = 0; k < P; k++) TotAVec[k] =(double) 0;
    int j = 0;
    for ( j =0; j< MaxIter; j++){
        
//         printf("--------------------------------\n");
//         printf("iter %d \n", j);
//        getrusage(RUSAGE_SELF,&r_usage);// Print the maximum resident set size used (in kilobytes).
//    printf("Memory usage: %ld kilobytes\n",r_usage.ru_maxrss);


        // memset(ActvMat, 0, sizeof( *ActvMat)*(P*N));/
        // printf("memset(ActvMat, 0, sizeof(double)*(P*N));\n ");
        for (int i = 0; i < N; i++){
            // printf("iter %d, n %d\n", j, i);
//            for (int k = 0; k <= P; k++) {
//                Walk[k] = 0; //IOrdr[j] = j;
//            }
             memset(Walk, 0, sizeof(double)*(P+1));
            // printf("memset(Walk, 0, sizeof(double)*(P+1));;\n ");
            oldx = P;
            newx = P;
            yval = 0;
            count = 0;
            /*
            Walk(P+1)    = 0;
            Walk(P:-1:1) = cumsum(-Weights(IOrdMat(P:-1:1,i)).*Labels(IOrdMat(P:-1:1,i)));
            */
#pragma omp parallel for
            for(int k = P-1; k >= 0;k--){
                Walk[P-1-k] = -(double)Weights[OrdrMat[i][k]]*(double)Labels[OrdrMat[i][k]];
                
            }
            for(int p = 1; p < P;){
               Walk[p] = Walk[p - 1] + Walk[p];
                p++;
            }
            double *tWalk = malloc(sizeof(double)*(P));
#pragma omp parallel for
             for(int p = 0; p < P;p++){
                tWalk[p] = Walk[P-1-p];
             }
//             for(int p = 0; p < P; p++){
//                Walk[p] = tWalk[p];
//             }
              memcpy(Walk, tWalk, sizeof(double)*(P+1));
             // printf("memcpy(Walk, tWalk, sizeof(double)*(P+1));\n ");
             free(tWalk);
             // printf("free(tWalk);\n");

            Walk[P] = 0;


//             Walk[P] = 0;
//             for(int p = 0; p < P; p++) Walk[p] = - Walk[p];

             // for(int p = 0; p <= P; p++) printf("Walk[%d] %.4f\n", p, Walk[p]);

             do{

            MeasVec = malloc(sizeof(double)*(oldx));
           if( MeasVec == NULL) printf("Fail to allocate MeasVec");
           MeasVec2 = malloc(sizeof(double)*(oldx));
           if( MeasVec2 == NULL) printf("Fail to allocate MeasVec2");
 //                 getrusage(RUSAGE_SELF,&r_usage);// Print the maximum resident set size used (in kilobytes).
    // printf("Memory usage: %ld kilobytes\n",r_usage.ru_maxrss);


//                 for (int k = oldx - 1; k >= 0; k--){
//                     // MeasVec[oldx-2-k] = (lmbdm + lmbdp)* (k+1);
//                     // MeasVec2[oldx-2-k] = -(lmbdm*attrm + lmbdp*attrp)*(k+1);
//                     MeasVec[oldx-1-k] =   (double)(k+1);
//                     MeasVec2[oldx-1-k] =  - (double)(k+1);
//                     // printf( "MeasVec2[%d]  %.4f\n", oldx-1-k, MeasVec2[oldx-1-k]);
//
//                 }
                 // printf("right\n");
#pragma omp parallel for
                 for (int k = oldx - 1; k >= 0; k--){
                    MeasVec[oldx-1-k] = (lmbdm + lmbdp)* (k+1);
                    MeasVec2[oldx-1-k] = -(lmbdm*attrm + lmbdp*attrp)*(k+1);
//                     MeasVec[oldx-1-k] =   (double)(k+1);
//                     MeasVec2[oldx-1-k] =  - (double)(k+1);
                    // printf( "MeasVec2[%d]  %.4f\n", oldx-1-k, MeasVec2[oldx-1-k]);
//
                }
                 // printf("right\n");
// #pragma omp parallel for
                 for (int k = 0; k< oldx -1 ; k++){
                     // printf( "MeasVec2[%d]  %.4f\n", k, MeasVec2[k]);
                     MeasVec[k] = MeasVec[k]- (lmbdp - lmbdm) *(Walk[k] - Walk[oldx-1]);
                     MeasVec2[k] = MeasVec2[k] + (1+ lmbdp*attrp - lmbdm*attrm)*(Walk[k] - Walk[oldx-1]);
                     // MeasVec2[k] =   -(double)(oldx - k)+ (Walk[k] - Walk[oldx]);
                     MeasVec2[k] = MeasVec2[k]/MeasVec[k];
                      

                 } 
                 // printf("right2, oldx %d\n",oldx);
                 minid = findmin(MeasVec2, oldx); // wrong here
                 yval = minid.yval;
                 newx = minid.id;
//                 if (!(newx > 0)) newx = 0;

                 // printf("yval %f, newx %d\n", yval, newx);
                 if (yval >= 0){

//                     for (int l = 0; l< oldx; l++) ActvMat[i][l] = 0;
                     memset(ActvMat[i], 0, oldx );
                     break;

                 }
                 // printf("right3, newx %d, oldx %d\n",newx, oldx);
#pragma omp parallel for
                 for (int k = newx; k< oldx;k++) {
                    ActvMat[i][k] = -yval;
                     
                }
                 // for (int k = 0; k<P; k++) printf("ActvMat[%d][%d], %.4f\n", i, k,ActvMat[i][k] );
                 oldx = newx;


                 // printf("right4\n");
                 count ++;
                 end = clock();
                cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

                // if (count % 5000==0)  printf("iter %d, n %d, walk %d: %f s, total: %f s\n",j,i,count, cpu_time_used - old_cpu_time_used, cpu_time_used);
                old_cpu_time_used = cpu_time_used;
//                 if (  j>0){
//                     printf("count =50");
//                 }
                free(MeasVec);
            free(MeasVec2);

             }while(newx > 0 );
            
        }
//        printf("N %d while loop %d\n",i+1, count);
#pragma omp parallel for
        for (int k = 0; k< P; k++){
            TotAVec[k] = 0;

            for (int n = 0; n< N; n++){

                TotAVec[k] = TotAVec[k] + ActvMat[n][IOrdrMat[n][k]];
                
            }
//            printf("\n");
            // printf("TotAVec[%d], %.4f\n",k, TotAVec[k]);
        }
//        for (int n = 0; n< N; n++){
//            for (int k = 0; k< P; k++)  printf("%f\t",ActvMat[n][IOrdrMat[n][k]]);
//            printf("\n");
//        }
        // printf("right5\n");

        end = clock();
        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        // printf("ActvMat: %f s, total: %f s\n", cpu_time_used - old_cpu_time_used, cpu_time_used);
        old_cpu_time_used = cpu_time_used;

        Thresh = sum_double(TotAVec, P);
        Thresh = Thresh/(double)P;
#pragma omp parallel for
        for (int k = 0; k< P; k++) Errors[k] = ((TotAVec[k] - Thresh)*(double)Labels[k]) < Marg*Thresh;

        NumError = sum(Errors, P);
        printf("NumError %d\n ", NumError);

        if (NumError == 0) {
            printf("correct!\n");
            FLAG = 0;
            break;
           
        }

        if ( j % 10 == 0){
            FILE *fp2 = NULL;
        //    char *filepath = "/Users/yuru/Documents/GitHub/Monsyn/";
            char filename[100];

            sprintf(filename, "./monsyn4/N%dP%dtrial%d.txt",N,P,trial_id);
            remove(filename);
            fp2 = fopen(filename,  "aw");
            if( fp2 == NULL ) {
                   printf("Error while opening the file.\n");
                   return -1;
            }
            fprintf(fp2, "%d,%d,%d,%d,%d,%d",N, P, trial_id, sum(Errors, P), j, FLAG);
            fclose(fp2);

            char filename1[100];
            sprintf(filename1, "./monsyn4/N%dP%dtrial%dinput.txt",N,P,trial_id);
            remove(filename1);
            FILE *f = fopen(filename1, "wb");
            if( f == NULL ) {
                   printf("Error while opening the file.\n");
                   return -1;
            }
            for (int i = 0; i< N; i++){
                for (int k = 0; k < P; k++) {
                    fprintf(f, "%d ", OrdrMat[i][k]);
                }
                fprintf(f, "\n");
            }
            fclose(f);

            char filename2[100];
            sprintf(filename2, "./monsyn4/N%dP%dtrial%dfunc.txt",N,P,trial_id);
            remove(filename2);
            FILE *f2 = fopen(filename2, "wb");
            if( f2 == NULL ) {
                   printf("Error while opening the file.\n");
                   return -1;
    }
    for (int i = 0; i< N; i++){
        for (int k = 0; k < P; k++) {
            fprintf(f2, "%.4f ", ActvMat[i][k]);
        }
        fprintf(f2, "\n");
    }
    fclose(f2);
        }




#pragma omp parallel for
        for (int k = 0; k< P; k++) {
            Rate[k] = Errors[k] * (Rate[k] + 0.2* Errors[k]);
            Weights[k] = Weights[k] + Rate[k];
            // printf("Weights[%d], %.4f\n",k, Weights[k]);
        }
        // printf("Weights: %f s, total: %f s\n", cpu_time_used - old_cpu_time_used, cpu_time_used);
    }

    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    // printf("total %f seconds to execute \n", cpu_time_used);
    old_cpu_time_used = cpu_time_used;

//    getrusage(RUSAGE_SELF,&r_usage);// Print the maximum resident set size used (in kilobytes).
//printf("Memory usage: %ld kilobytes\n",r_usage.ru_maxrss);

    //for (int i = 0; i < N; i++){
        // free(OrdrMat[i]);
       // free(IOrdrMat[i]);
      //  free(ActvMat[i]);
    //}
    // free(IOrdr);
    // free(Ordr);
    //free(Walk);
    //free(Weights);
    //free(Rate);
    FILE *fp2 = NULL;
//    char *filepath = "/Users/yuru/Documents/GitHub/Monsyn/";
    char filename[100];

    sprintf(filename, "./monsyn4/N%dP%dtrial%d_lmbdsmall.txt",N,P,trial_id);
    remove(filename);
    fp2 = fopen(filename,  "aw");
    if( fp2 == NULL ) {
           printf("Error while opening the file.\n");
           return -1;
    }
    fprintf(fp2, "%d,%d,%d,%d,%d,%d",N, P, trial_id, sum(Errors, P), j, FLAG);
    fclose(fp2);

    char filename1[100];
    sprintf(filename1, "./monsyn4/N%dP%dtrial%dinput_lmbdsmall.txt",N,P,trial_id);
    remove(filename1);
    FILE *f = fopen(filename1, "wb");
    if( f == NULL ) {
           printf("Error while opening the file.\n");
           return -1;
    }
    for (int i = 0; i< N; i++){
        for (int k = 0; k < P; k++) {
            fprintf(f, "%d ", OrdrMat[i][k]);
        }
        fprintf(f, "\n");
    }
    fclose(f);

    char filename2[100];
    sprintf(filename2, "./monsyn4/N%dP%dtrial%dfunc_lmbdsmall.txt",N,P,trial_id);
    remove(filename2);
    FILE *f2 = fopen(filename2, "wb");
    if( f2 == NULL ) {
           printf("Error while opening the file.\n");
           return -1;
    }
    for (int i = 0; i< N; i++){
        for (int k = 0; k < P; k++) {
            fprintf(f2, "%.4f ", ActvMat[i][k]);
        }
        fprintf(f2, "\n");
    }
    fclose(f2);
}