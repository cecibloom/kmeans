/*****************************************************************************/
/*IMPORTANT:  READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.         */
/*By downloading, copying, installing or using the software you agree        */
/*to this license.  If you do not agree to this license, do not download,    */
/*install, copy or use the software.                                         */
/*                                                                           */
/*                                                                           */
/*Copyright (c) 2005 Northwestern University                                 */
/*All rights reserved.                                                       */

/*Redistribution of the software in source and binary forms,                 */
/*with or without modification, is permitted provided that the               */
/*following conditions are met:                                              */
/*                                                                           */
/*1       Redistributions of source code must retain the above copyright     */
/*        notice, this list of conditions and the following disclaimer.      */
/*                                                                           */
/*2       Redistributions in binary form must reproduce the above copyright   */
/*        notice, this list of conditions and the following disclaimer in the */
/*        documentation and/or other materials provided with the distribution.*/ 
/*                                                                            */
/*3       Neither the name of Northwestern University nor the names of its    */
/*        contributors may be used to endorse or promote products derived     */
/*        from this software without specific prior written permission.       */
/*                                                                            */
/*THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS    */
/*IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED      */
/*TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY, NON-INFRINGEMENT AND         */
/*FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL          */
/*NORTHWESTERN UNIVERSITY OR ITS CONTRIBUTORS BE LIABLE FOR ANY DIRECT,       */
/*INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES          */
/*(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR          */
/*SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)          */
/*HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,         */
/*STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN    */
/*ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE             */
/*POSSIBILITY OF SUCH DAMAGE.                                                 */
/******************************************************************************/

/*************************************************************************/
/**   File:         example.c                                           **/
/**   Description:  Takes as input a file:                              **/
/**                 ascii  file: containing 1 data point per line       **/
/**                 binary file: first int is the number of objects     **/
/**                              2nd int is the no. of features of each **/
/**                              object                                 **/
/**                 This example performs a fuzzy c-means clustering    **/
/**                 on the data. Fuzzy clustering is performed using    **/
/**                 min to max clusters and the clustering that gets    **/
/**                 the best score according to a compactness and       **/
/**                 separation criterion are returned.                  **/
/**   Author:  Wei-keng Liao                                            **/
/**            ECE Department Northwestern University                   **/
/**            email: wkliao@ece.northwestern.edu                       **/
/**                                                                     **/
/**   Edited by: Jay Pisharath                                          **/
/**              Northwestern University.                               **/
/**                                                                     **/
/**   ================================================================  **/
/**																		**/
/**   Edited by: Sang-Ha  Lee											**/
/**				 University of Virginia									**/
/**																		**/
/**   Description:	No longer supports fuzzy c-means clustering;	 	**/
/**					only regular k-means clustering.					**/
/**					Simplified for main functionality: regular k-means	**/
/**					clustering.											**/
/**                                                                     **/
/*************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <sys/types.h>
#include <fcntl.h>
#include <omp.h>
#include "getopt.h"

#include "kmeans.h"

extern double wtime(void);

int num_omp_threads = 1;
FILE *fileResults;

/*---< usage() >------------------------------------------------------------*/
void usage(char *argv0) {
    char *help =
        "Usage: %s [switches] \n"
        "       -i filename     	: file containing data to be clustered\n"
        "       -c filename     	: file containing configuration\n"    
        "       -b                 	: input file is in binary format\n"
	"       -n no. of threads	: number of threads\n";
    
    fprintf(stderr, help, argv0);
    exit(-1);
}


/**
 * This function computes the quality measure based on the number of False Positives 
 * and False Negatives the algorithm found based on one ideal configuration. It
 * performs the computation between 2 sets, the so called "golden" and the approximation.
 * A retrieved set defined as "Membership" contains for each point in the dataset, a reference 
 * to the cluster it belongs after the clusterization has been done. The reference (membership)
 * is simply defined as the number of the cluster. The membership set is composed then by 
 * true positives (TP) and false positives (FP).
 * Then there is of course, false negatives (FN) and true negatives (TN), as a whole
 * they all conform the universe of points.
 * 
 * To perform the calculations we considered:
 * 
 * Given    G(k)    :   Ideal configuration for cluster k
 *          A(k)    :   Approximation for cluster k
 *          A'(k)   :   Subset containg TP
 * 
 * FN = len(G(k)) - len(A'(k))  
 * FP = len(G(k)) - len(A'(k))  
 * 
 * So every point that it's being assigned a wrong cluster has a penalty of 2: the first for
 * being a false negative, with respect to its true cluster, and the other for being a false
 * positive, with respect to the cluster to whom has been mistakenly assigned.
 * 
 * 
 * @param golden - Ideal configuration.
 * @param approx - Approximation to be compared against golden
 * @return falses - sum of FN + FP for all k
 * 
 */
int quality3(int    *golden, int    *approx, int npoints){
    int fn_fp = 0;
    int tp = 0;
    int t=0;
    int i;
    
    for (i = 0; i < npoints; i++){
        if (approx[i] == golden[i]){
                tp++;
        } else {
            t++;
        }        
    }
    
    fn_fp = 2 * (npoints - tp);
    
    return fn_fp;
}

/*---< main() >--------------------------------------------------------------------------*/
int main(int argc, char **argv) {
            int     opt;
    extern  char   *optarg;
    extern  int     optind;
            char   *filename = 0;   
            float  *buf;
            float **attributes;
            float **cluster_centres=NULL;
            int     i, j;
                
            int     numAttributes;
            int     numObjects;        
            char    line[1024];           
            int     isBinaryFile = 0;
            double  timing;		   

            int     numApprox = 0;       // Fisrt line is consider as the golden, other are taken as approximations 
            int     *ks;
            float   *thresholds;
            int     *loops;
            char    *config_filename = 0;   
            int     **memberships;
           
            while ( (opt=getopt(argc,argv,"i:c:b:n:?"))!= EOF) {
                    switch (opt) {
                case 'i': filename=optarg;
                          break;
                case 'c': config_filename=optarg;
                          break;          
                case 'b': isBinaryFile = 1;
                          break;
                case 'n': num_omp_threads=atoi(optarg);
                          break;          
                case '?': usage(argv[0]);
                          break;
                default: usage(argv[0]);
                          break;
            }
    }


    if (filename == 0 && config_filename == 0) usage(argv[0]);

    numAttributes = numObjects = 0;

    /* from the input file, get the numAttributes and numObjects ------------*/

    // Reading dataset
        FILE *infile;
        if ((infile = fopen(filename, "r")) == NULL) {
            fprintf(stderr, "Error: no such file (%s)\n", filename);
            exit(1);
        }
        while (fgets(line, 1024, infile) != NULL)
            if (strtok(line, " \t\n") != 0) // Split string into tokens, the delimiters are \t and \n, if there´s at least one
                numObjects++; //Increase in one (per line) the numObjects variable
        rewind(infile); // After finishing reading the file rewind it (Set position of stream to the beginning)
        
        
        while (fgets(line, 1024, infile) != NULL) {
            if (strtok(line, " \t\n") != 0) {
                /* ignore the id (first attribute): numAttributes = 1; */
                while (strtok(NULL, " ,\t\n") != NULL) numAttributes++; // Assuming all lines contain the same number of attributes
                break;
            }
        }
        
        if (numObjects == 0) {
            printf("Error: empty dataset\n");
            exit(1);
        }
        
        /* allocate space for attributes[] and read attributes of all objects */
        buf           = (float*) malloc(numObjects*numAttributes*sizeof(float)); //I get this is the # of cells
        attributes    = (float**)malloc(numObjects*             sizeof(float*)); //Number of lines
        attributes[0] = (float*) malloc(numObjects*numAttributes*sizeof(float));
        for (i=1; i<numObjects; i++)
            attributes[i] = attributes[i-1] + numAttributes; //each has the attr of a line
        rewind(infile);
        i = 0;
        while (fgets(line, 1024, infile) != NULL) {
            if (strtok(line, " \t\n") == NULL) continue; 
            for (j=0; j<numAttributes; j++) {
                buf[i] = atof(strtok(NULL, " ,\t\n")); //a cell has the value of an attr
                i++;
            }
        }
        fclose(infile);
        
        // Reading Configuration
        
        FILE *infileconfig;
        if ((infileconfig = fopen(config_filename, "r")) == NULL) {
            fprintf(stderr, "Error: no such file (%s)\n", config_filename);
            exit(1);
        }
        
        while (fgets(line, 1024, infileconfig) != NULL)
            if (strtok(line, " \t\n") != 0) 
                numApprox++;
        rewind(infileconfig);
        
        printf("\nNumber of computations: %d\n",numApprox);
        if (numApprox < 2) {
            printf("\nError: You need to specify one configuration set for the 'golden' and at least one for the approximation\n");
            exit(1);
        }
        
        ks          =   (int*)malloc(numApprox * sizeof(int));
        thresholds  =   (float*)malloc(numApprox * sizeof(float));
        loops       =   (int*)malloc(numApprox * sizeof(int));
        
        i = 0;
        
        while (fgets(line, 1024, infileconfig) != NULL) {
            if (strtok(line, " \t\n") == NULL) continue; 
            ks[i] = atoi(strtok(NULL, " ,\t\n")); //storing all values of K
            thresholds[i] = atof(strtok(NULL, " ,\t\n")); //sotring all values of Threshold
            loops[i] = atoi(strtok(NULL, " ,\t\n")); //storing all values of Loops
            i++;
        }
        
        fclose(infileconfig);
        
        
	printf("I/O completed\n");	

	memcpy(attributes[0], buf, numObjects*numAttributes*sizeof(float)); //copia en at[0] de buf toda la celda

        time_t rawtime;
        char buffer [255];

        time (&rawtime);
        sprintf(buffer,"../results_%s.txt",ctime(&rawtime));
        
        fileResults = fopen(buffer, "w");
        if (fileResults == NULL){
            printf("Error opening file!\n");
            exit(1);
        }
 
        fprintf(fileResults, "\n┌————————————————————————————————————————————————— GLOBAL CONFIGURATION —————————————————————————————————————————————————┐");
        fprintf(fileResults, "\n│%35s%-35s%-1s%16d%35s"," ","Number of Observations","=", numObjects,"│");
        fprintf(fileResults, "\n│%35s%-35s%-1s%16d%35s"," ","Number of Attributes","=", numAttributes,"│");
        fprintf(fileResults, "\n│%35s%-35s%-1s%16d%35s"," ","Number of Threads","=", num_omp_threads,"│");
        fprintf(fileResults, "\n└————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————┘");

        int k;
        
        memberships    = (int**)malloc(numApprox*             sizeof(int*)); //Number of lines
        memberships[0] = (int*) malloc(numApprox*numObjects*sizeof(int));

        k=0;
        for (k=1; k<numApprox; k++)
            memberships[k] = memberships[k-1] + numObjects; //each has the attr of a line
        
        for (i=0; i<numApprox; i++) {

            fprintf(fileResults, "\n\n\n\n ××××××××××××××××××××××××××××××  SETTING N° %d  ×××××××××××××××××××××××××××××",i+1);
            fprintf(fileResults, "\n×%15s%-31s%-1s%16d%14s"," ","Number of Clusters","=",ks[i],"×");
            fprintf(fileResults, "\n×%15s%-31s%-1s%16f%14s"," ","Threshold","=", thresholds[i],"×");
            fprintf(fileResults, "\n×%15s%-31s%-1s%16d%14s"," ","Number of Loops","=", loops[i],"×");
            fprintf(fileResults, "\n ×××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××\n\n");

            timing = omp_get_wtime();
            cluster_centres = NULL;
            
            cluster(numObjects,
                    numAttributes,
                    attributes,                         
                    ks[i],
                    thresholds[i],
                    loops[i],
                    &cluster_centres ,
                    memberships[i]
                   );
            timing = omp_get_wtime() - timing;
            fprintf(fileResults,"………………………………………………………………………………………………………………………………………………………………………………………………………\n");
            fprintf(fileResults,"\t\t\t\t\t\tProcess Time: %f\n", timing);
            fprintf(fileResults,"………………………………………………………………………………………………………………………………………………………………………………………………………");
            
            if (i > 0){
                fprintf(fileResults,"\n\n………………………………………………………………………………………………………………………………………………………………………………………………………\n");
                fprintf(fileResults,"\t\t\t\t\t\tFN + FP: %d\n", quality3(memberships[0], memberships[i], numObjects));
                fprintf(fileResults,"………………………………………………………………………………………………………………………………………………………………………………………………………\n");
            }
            
            /*
            k=0;
            for (k = 0; k < ks[i]; k++){
                fprintf(fileResults,"\n\n……………………………………………………………………………………………………………………………\n");
                fprintf(fileResults,"\t\t\t\tCentroid[%d]\n", k+1);
                fprintf(fileResults,"……………………………………………………………………………………………………………………………\n");
                for (j = 0; j < numAttributes; j++) {
                    if (j%5 == 0) fprintf(fileResults,"\n");
                    fprintf(fileResults,"%15f", cluster_centres[k][j]);
                }
            }
             */
        }
        fclose(fileResults);

        free(attributes);
        free(cluster_centres[0]);
        free(cluster_centres);
        free(buf);
        free(ks);
        free(thresholds);
        free(loops);
        free(memberships);
        return(0);
}

