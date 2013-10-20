/******************************************************************
**   tool.h
     header file for matrix structures and basic functions used by gmeans.cc
     struct doc_mat;          sparse matrix
     struct dense_mat;        dense matrix
     int* read_mat;           read sparse matrix
     int* read_mat;           read dense matrix
     void print_help;         print help file
     void output_clustering;  output the clustering of vectors
     void extractfilename;    extract file name from a path
**
*****************************************************************/

#if !defined(_TOOLS_H_)
#define _TOOLS_H_

#include "Matrix.h"

#define MAX_DESC_STR_LENGTH	9
#define FILENAME_MAXLENGTH      256
#define SCALE_SURFIX_LENGTH     128


struct doc_mat
{
	char strDesc[MAX_DESC_STR_LENGTH];	// description string
	int n_row, n_col;
	int n_nz;
	int *col_ptr;
	int *row_ind;
	float *val;
};
//struct for a dense matrix
struct dense_mat
{
  int n_row, n_col, n_nz;
  float **val;
};

extern long memory_consume;

int* read_mat(char *fname, dense_mat *mat, int & n_Empty_Docs, int format);
int* read_mat(char *fname, char *scaling, doc_mat *mat, int & n_Empty_Docs);
void print_help();
void output_clustering(int *cluster, int n, int m, char *filename,char *suffix, int format, char *docs);
void extractfilename(char *path, char *name);


#endif // !defined(_TOOLS_H)
