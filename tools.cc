#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream.h>
#include <math.h>
#include <time.h>
#include "tools.h"

void print_help()
{
	printf("USAGE: gmeans [switches] matrix-file\n");
	printf("\t-a algorithm\n");
	printf("\t   s: spherical k-means algorithm (default)\n");
	printf("\t   b: information bottleneck algorithm\n");
	printf("\t   e: euclidean k-means algorithm\n");
	printf("\t   k: kullback_leibler k-means algorithm \n");  
	printf("\t   d: diametric k-means algorithm \n");
	printf("\t-c number-of-clusters (default is 1)\n");
	printf("\t-D delta, this float number sets threshold for first variations (default 10e-6\n");
	printf("\t-d dump the clustering process (default is false)\n");
	printf("\t-E #-of-Kmeans-iterations, this integer sets when to use similarity estimate (default is 5)\n");
	printf("\t-e epsilon, this float number set threshold for k-means (default is .001)\n");
	printf("\t-F matrix format\n");
	printf("\t   s: sparse matrix in CCS (default)\n");
	printf("\t   d: dense matrix, i.e. each column represents a data point \n");
	printf("\t   t: dense matrix transpose, i.e. each row represents a data point \n");
	printf("\t-f number-of-first-variations (default is 0)\n");
	printf("\t-i [S|s|p|r|f|c]\n");
	printf("\t   initialization method:\n");
	printf("\t      p -- random perturbation\n");
	printf("\t      r -- random cluster ID\n");
	printf("\t      f -- read from cluster ID seeding file\n");
	printf("\t      S -- choose the 1st centroid the farthest point from the center of the whole data set, well separate all centroids (defualt)\n");
	printf("\t      s -- same as 'S' but get the 1st centroid randomly\n");
	printf("\t      c -- randomly pick vectors as centroids\n");
	printf("\t-L use Laplacian for Kullback_leibler distance measure (default is false)\n");
	printf("\t      c alpha -- use prior\n");
	//printf("\t      p       -- use datamatrix-based prior\n");
	printf("\t      n       -- no prior \n");
	/*
	printf("\t-M [a|b|d|m] model selection criterion\n");
	printf("\t      a -- AIC\n");
	printf("\t      b -- BIC\n");
	printf("\t      d -- MDL (essentially it is BIC\n");
	printf("\t      m -- MML\n");
	*/
	printf("\t-O the name of output matrix\n");
	printf("\t-P set prior values\n");
	printf("\t      u -- uniform priors (default)\n");
	printf("\t      f filename -- read priors from a file\n");
	printf("\t-p perturbation-magnitude (default is .5)\n");
	printf("\t-S skip spherical K-Means and only use first variation to cluster(default is false)\n");
	printf("\t-s perturbation with a positive integer seed\n");
	printf("\t-T true label file\n");
	printf("\t-t scaling scheme (default is tfn for sparse matrix)\n");
	printf("\t-U cluster_file (to valuate a given clustering, default is false)\n");
	printf("\t-u upper bound of cluster number (default is equal to '-c')\n");
	printf("\t-V verify obj. func. (default is false)\n");
	printf("\t-v version-number\n");
	printf("\t-W wordfile, produces word clustering in the specified file\n");
	printf("\t-w Omiga value is between 0 and -1 (default is 0)\n");
	
	//printf("\t-D concept decompostion\n");
	
	//printf("\t-s\n");
	//printf("\t   suppress output\n");
	//printf("\t-n\n");
	//printf("\t   no dump\n");
	
	//printf("\t-p perturbation-magnitude\n");
	//printf("\t   the distance between initial concept vectors\n");
	//printf("\t     and the centroid will be less than this.\n");
	//printf("\t-N number-of-samples\n");
	
	
	
	//printf("\t-K dimension reduction parameter\n");
	//printf("\t-S sample-size\n");
	//printf("\t   in ratio to the total number of documents\n");
	//printf("\t-E encoding-scheme\n");
	//printf("\t   1: normalized term frequency (default)\n");
	//printf("\t   2: normalized term frequency inverse document frequency\n");
	//printf("\t-o objective-function\n");
	//printf("\t   1: nonweighted (default)\n");
	//printf("\t   2: weighted\n");
}

int* read_mat(char *fname, char *scaling, doc_mat *mat, int & n_Empty_Docs)
// read in a sparse matrix from files into 'mat' and return #empty vectors and an array of empty vector IDs
{
  char filename[FILENAME_MAXLENGTH];
  //clock_t start_clock, finish_clock;

  sprintf(filename, "%s%s", fname, "_dim");
  std::ifstream dimfile(filename);
  if(dimfile==0)
    cout<<"Dim file "<<filename<<" can't open.\n"<<endl;
  sprintf(filename, "%s%s", fname, "_row_ccs");
  std::ifstream rowfile(filename);
  if(rowfile==0)
    cout<<"Row file "<<filename<<" can't open.\n"<<endl;
  sprintf(filename, "%s%s", fname, "_col_ccs");
  std::ifstream colfile(filename);
  if(colfile==0)
    cout<<"Column file "<<filename<<" can't open.\n"<<endl;
  sprintf(filename, "%s%s", fname, "_");
  sprintf(filename, "%s%s", filename, scaling);
  sprintf(filename, "%s%s", filename, "_nz");
  std::ifstream nzfile(filename);
  if(nzfile==0)
    cout<<"Entry file "<<filename<<" can't open.\n"<<endl;
  
  int i;
  
  if(dimfile==0|| colfile==0|| rowfile==0|| nzfile==0)
    {
      cout<<"Matrix file "<< fname<<"_* can't open."<<endl;
      exit(1);
    }
  //data.width(MAX_DESC_STR_LENGTH);
  //data >> mat->strDesc;
  //data >> mat->n_col >> mat->n_row >> mat->n_nz;
  //start_clock = clock();
  cout<<"Reading dimension file..."<< endl;
  dimfile >>mat->n_row>> mat->n_col>>mat->n_nz;
  dimfile.close();
  
  mat->col_ptr = new int[mat->n_col+1];
  mat->row_ind = new int[mat->n_nz];
  mat->val = new float[mat->n_nz];

  //space used for storing the sparse matrix is as follows
  memory_consume+=(mat->n_col+1+mat->n_nz)*sizeof(int)+mat->n_nz*sizeof(float);
  
  //it is necessary to handle empty vectors separately
  int pre=-1;
  int *empty_Docs_ID = new int[mat->n_col], *e_D_ID;
  n_Empty_Docs =0;
  
  cout<<"Reading column pointer file..."<<endl;
  for (i = 0; i < mat->n_col+1; i++)
    {
      colfile >> (mat->col_ptr)[i];
      if ( (mat->col_ptr)[i] == pre)
	{
	  empty_Docs_ID[n_Empty_Docs]= i-1;
	  n_Empty_Docs ++;
	}
      pre = (mat->col_ptr)[i];
    }
  colfile.close();
  
  e_D_ID = new int [n_Empty_Docs+1];
  
  for(i=0; i<n_Empty_Docs;i++)
    e_D_ID[i]= empty_Docs_ID[i];
  
  e_D_ID [i] = mat->n_col;
  
  delete [] empty_Docs_ID;
  
  cout<<"Reading row index file..."<<endl;
  for (i = 0; i < mat->n_nz; i++)
    rowfile >> (mat->row_ind)[i];
  rowfile.close();
  
  cout<<"Reading non-zero value file..."<<endl;
  for (i = 0; i < mat->n_nz; i++)
    nzfile >> (mat->val)[i];
  nzfile.close();
  cout<<endl;
  //finish_clock = clock();
  //cout<<"Reading file time: "<<(finish_clock - start_clock)/1e6<<" seconds."<<endl;
  cout<<"Empty docs found: "<<n_Empty_Docs<<endl;
  
  if (n_Empty_Docs>0)
    cout<<"They are vector ";
  
  for(i=0; i<n_Empty_Docs; i++)
    cout <<e_D_ID[i]<<", ";
  cout<<endl;
  return  e_D_ID;
}

int* read_mat(char *fname, dense_mat *mat, int & n_Empty_Docs, int format)
  //read in a dense matrix; but since either rows are to be clustered or columns are to be clustered,
  //we use 'format' to identify that.
  //for dense matrix we assume there is NO empty vector.
{
  int i, j;
  int *e_D_ID;
  std::ifstream inputfile(fname);
  if(inputfile==0)
    {
      cout<<"Matrix file "<<fname<<" can't open.\n"<<endl;
      exit(1);
    }
  
  cout<<"Reading matrix file..."<< endl;
  switch (format)
    {
      //if you want to cluster columns like the word-doc case
    case DENSE_MATRIX:
      inputfile >>mat->n_row>> mat->n_col;
      mat->n_nz = mat->n_row * mat->n_col;
      
      mat->val = new (float*) [mat->n_row];
      for (i=0; i<mat->n_row; i++)
	mat->val[i]= new float [mat->n_col];
      
      memory_consume+=(mat->n_row * mat->n_col)*sizeof(float);
  
      n_Empty_Docs =0;
      for (i = 0; i < mat->n_row; i++)
	{
	  for (j=0; j < mat->n_col; j++)
	    inputfile >> mat->val[i][j];
	}
      break;
      //if you want to cluster rows like the gene-expression case
    case DENSE_MATRIX_TRANS:
      inputfile >>mat->n_col>> mat->n_row;
      mat->n_nz = mat->n_row * mat->n_col;
      
      mat->val = new (float*) [mat->n_row];
      for (i=0; i<mat->n_row; i++)
	mat->val[i]= new float [mat->n_col];

      memory_consume+=(mat->n_row * mat->n_col)*sizeof(float);
  
      n_Empty_Docs =0;
      for (i = 0; i < mat->n_col; i++)
	{
	  for (j=0; j < mat->n_row; j++)
	    inputfile >> mat->val[j][i];
	}
      break;
    }

  e_D_ID = new int [n_Empty_Docs+1];
  e_D_ID [0] = mat->n_col;
  return  e_D_ID;
}

void output_clustering(int *cluster, int n, int final_cluster_num, char *output_matrix, char *suffix, int format, char *docs_matrix)
{
  char browserfilepost[FILENAME_MAXLENGTH];
  int i, j;

  
  sprintf(browserfilepost, "_doctoclus.%d", final_cluster_num);
  
  if(format == SPARSE_MATRIX)
    {
      strcat(output_matrix, "_");
      strcat(output_matrix, suffix);
    }
  strcat(output_matrix, browserfilepost);
  strcat(docs_matrix, "_docs");
  
  std::ofstream o_m(output_matrix);
  std::ifstream docs(docs_matrix);
  
  char ch;
  o_m<<n<<endl;
  
  for (i=0; i <n; i++)
    {
      docs>>j;
      docs>>ch;
      o_m<<cluster[i];
      o_m<<"\t";
      if (docs.is_open())
	{
	  docs>>docs_matrix; //path of the file; reuse docs_matrix string
	  o_m<<docs_matrix<<endl;
	}
      else
	o_m<<endl;
      
    }
  
  o_m.close();
  if (docs.is_open())
    docs.close();
}

void extractfilename(char *path, char *name)
{
  int length, i, j;
  length = strlen(path);
  for(i= length-1; i>=0; i--)
    if ((path[i] == '/') || (path[i] == '\\'))
      {
	i++;
	for (j=i; j<length; j++)
	  name[j-i]=path[j];
	break;
      }
    else if (i==0)
      {
	for (j=i; j<length; j++)
	  name[j-i]=path[j];
	break;
      }
}


