#include <stdio.h>
#include <iostream.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include "RandomGenerator.h"
//#include "powermethod.h"

#define Power_Method_Epsilon          0.0001

float euclidian_distance(float x[], float y[], int dim)
{
  float dis =0;
  for (int i=0; i< dim; i++)
    dis += (x[i]-y[i])*(x[i]-y[i]);
  return dis;
}

float norm_2(float x[], int dim)
{
  float norm =0;
  for (int i=0; i< dim; i++)
    norm += x[i]*x[i];
  return sqrt(norm);
}

void Ax (float A[][3], float x[], int dim1, int dim2, float result[])
{
  for (int i=0; i<dim1; i++)
    {
      result[i] =0;
      for (int j=0; j<dim2; j++)
	result[i] += A[i][j]*x[j];
    }
}

void Ax(float *vals, int *rowinds, int *colptrs, int dim1, int dim2, float *x, float *result)
  /* compute Ax for a sparse matrix A and a dense vector x
     suppose A is (dim1 by dim2) matrix and x is (dim2 by 1) vector */ 
{
  for (int i = 0; i < dim1; i++)
    result[i] = 0.0;
  for (int i = 0; i < dim2; i++)
    {
      for (int j = colptrs[i]; j < colptrs[i+1]; j++)
	result[rowinds[j]] += vals[j] * x[i];
    }
}

void A_trans_A(int *colptrs, int *rowinds, float *vals,  int n_row, int flag, int * index, int *pointers, float **A_t_A, int & nz_counter)
/* computes A'A given doc_IDs in the same cluster; index[] contains doc_IDs for all docs, 
     pointers[] gives the range of doc_ID belonging to the same cluster;
     the resulting matrix is stored in dense format A_t_A( d by d matrix)
     The right SVs of A are the corresponding EVs of A'A 
     Notice that Gene expression matrix is usually n by d where n is #genes; this matrix
     format is different from that of document matrix. 
     In the main memory the matrix is stored in the format of document matrix.
  */
{
  int i, j, k, l, clustersize = pointers[1]-pointers[0]; 
  
  for (i=0; i<n_row; i++)
    for (j=0; j<n_row; j++)
      A_t_A[i][j] = 0;
  nz_counter=0;
  cout<<"clustersize "<<clustersize<<endl;

  if (flag >0)
    {
      for (k=0; k<clustersize; k++)
	for (l = colptrs[index[k+pointers[0]]]; l < colptrs[index[k+pointers[0]]+1]; l++)
	  for (j = colptrs[index[k+pointers[0]]]; j < colptrs[index[k+pointers[0]]+1]; j++)
	    {
	      if ( A_t_A[rowinds[l]][rowinds[j]] <= 0.000001 )
		nz_counter ++;
	      A_t_A[rowinds[l]][rowinds[j]] += vals[l]*vals[j];
	      cout<<l<<" "<<j<<" "<<rowinds[l]<<" "<<rowinds[j]<<" "<<A_t_A[rowinds[l]][rowinds[j]]<<endl;
	    }
    }
  else
    {
      for (k=0; k<clustersize; k++)
	for (l = colptrs[k+pointers[0]]; l < colptrs[k+pointers[0]+1]; l++)
	  for (j = colptrs[k+pointers[0]]; j < colptrs[k+pointers[0]+1]; j++)
	    {
	      if ( A_t_A[rowinds[l]][rowinds[j]] <= 0.000001 )
		nz_counter ++;
	      A_t_A[rowinds[l]][rowinds[j]] += vals[l]*vals[j];
	    }
    }
}

float x_dot_y(float x[], float y[], int dim1)
{
  float dot=0;
  for (int i=0; i<dim1; i++)
    dot += x[i]*y[i];
  return dot;
}

/*
void A_T_A(float A[][3], int dim1, int dim2, float result[][3])
{
  for (int j=0; j<dim2; j++)
    for (int k=0; k<dim2;k++)
      {
	result[j][k] =0;
	for (int i=0; i<dim1; i++)
	  result[j][k] += A[i][j]*A[i][k];
      }
}

int main(int argc, char **argv)

void power_method(float **A_t_A, int dim, float * CV)
{
  //float a[][3]={{24, 9, 1}, {5, 101, 98}};
  RandomGenerator_MT19937 rand_gen;
  float *x=new float[dim], *y=new float[dim], dis= 10* Epsilon, norm, *temp=new float[dim], Lamda, *old_x=new [dim];
  

  rand_gen.Set((unsigned)time(NULL));
  for (int i=0; i<dim; i++)
    x[i] = old_x[i] = rand_gen.GetUniformInt(10000);
  while (dis >Epsilon)
    {
      Ax(A_t_A, x, dim, dim, y);
      norm = L2_norm(y, dim);
      for (int i=0; i<dim; i++)
	x[i] = y[i]/norm;
      Ax(A_t_A, x, dim, dim, temp);
      Lamda = x_dot_y(x, temp, dim);
      dis = distance (x, old_x, dim);
      for (int i=0; i<dim; i++)
	old_x[i] = x[i];
      //cout<< dis<<endl;
    }
  //cout <<"**************"<<endl;
  //for (int i=0; i<3; i++)
  //cout<< x[i]<<endl;
}
*/
void power_method(float *vals, int *rowinds, int *colptrs, int dim, float * CV, float & Lamda)
  // power_method works for square matrix only; so we have only 1 value for dimension

{
 
  RandomGenerator_MT19937 rand_gen;
  float *x=new float[dim], *y=new float[dim], norm, *temp=new float[dim], *old_x=new float[dim], dis=0;
  int iter=0;

  rand_gen.Set((unsigned)time(NULL));
  for (int i=0; i<dim; i++)
    //x[i] = old_x[i] = rand_gen.GetUniformInt(10000);
    x[i]=old_x[i] =1.0;
  do
    {
      //old_dis =dis;
      Ax(vals, rowinds, colptrs, dim, dim, x, y);
      cout<<"Ax=";
      for (int i=0; i<dim; i++)
	cout<<y[i]<<" ";
      cout<<endl;
      norm = norm_2(y, dim);
      for (int i=0; i<dim; i++)
	x[i] = y[i]/norm;
      cout<<"After norm x=";
      for (int i=0; i<dim; i++)
	cout<<x[i]<<" ";
      cout<<endl;
      Ax(vals, rowinds, colptrs, dim, dim, x, temp);
      cout<<"temp=";
      for (int i=0; i<dim; i++)
	cout<<temp[i]<<" ";
      cout<<endl;
      Lamda = x_dot_y(x, temp, dim);
      cout<<"Lamda="<<Lamda<<endl;
      dis = euclidian_distance (x, old_x, dim);
      for (int i=0; i<dim; i++)
	old_x[i] = x[i];
      cout <<dis<<" "<<endl;
      iter ++;
    }
  while ( dis > Power_Method_Epsilon  );
  if (CV != NULL)
    for (int i=0; i<dim; i++)
      CV[i] = x[i];
  cout <<"Powermethod is done after "<<iter<<" iterations"<<endl;
}

int main(int argc, char **argv)
{
  
  float vals[]={3.0, 4.0, 1.0, 2.0, 1.0, 1.0, 3.0};
  int row_ind[]={0, 2, 2, 0, 1, 2, 3}, colptrs[]={0, 2, 3, 4, 6, 7};
  int n_Clusters =1, n_col=5, n_row =4, nz_counter=0;
  int *pointer = new int[n_Clusters+1], *index = new int[n_col], *range =new int[2];
  float **A_t_A;
  int *AtA_rowind, *AtA_colptr;
  float *AtA_val;
  int i, j, k, l;
  int cluster_size[1]={5};
  int cluster[5]={0, 0, 0, 0, 0};

  A_t_A =new (float*) [n_row];
  for (i=0; i<n_row; i++)
    A_t_A [i] = new float [n_row];

  pointer[0] =0;
  for (i=1; i<n_Clusters; i++)
    pointer[i] = pointer[i-1] + cluster_size[i-1];
  for (i=0; i<n_Clusters; i++)
    cout<<pointer[i]<<" ";
  cout<<endl;

  for (i=0; i<n_col; i++)
    {
      index[pointer[cluster[i]]] = i;
      pointer[cluster[i]] ++;
    }
  for (i=0; i<n_col; i++)
    cout<<index[i]<<" ";
  cout<<endl;

  pointer[0] =0;
  for (i=1; i<=n_Clusters; i++)
    pointer[i] = pointer[i-1] + cluster_size[i-1];
  for (i=0; i<=n_Clusters; i++)
    cout<<pointer[i]<<" ";
  cout<<endl;

  range[0] = pointer[0];
  range[1] = pointer[1];
  A_trans_A(colptrs, row_ind, vals, n_row, 1, index, range, A_t_A, nz_counter);

  cout<<endl<<"A_t_A is:"<<endl;
  for (i=0; i<n_row; i++)
    {
      for (j=0; j<n_row; j++)
	cout<<A_t_A[i][j]<<" ";
      cout<<endl;
    }
  cout<<"nz_counter "<<nz_counter<<endl;
  AtA_val = new float [nz_counter];
  AtA_rowind = new int [nz_counter];
  AtA_colptr = new int [n_row+1];
  AtA_colptr[0] = 0;
  k=0;
  for (j=0; j<n_row; j++)
    {
      l=0;
      for (i=0; i<n_row; i++)
	if (A_t_A[i][j] > 0)
	  {
	    AtA_val[k] = A_t_A[i][j];
	    AtA_rowind[k++] = i;
	    l++;
	  }
      AtA_colptr[j+1] = AtA_colptr[j] + l;
    }
  for (i=0; i<nz_counter; i++)
    cout<<AtA_val[i]<<" ";
  cout<<endl;
  for (i=0; i<nz_counter; i++)
    cout<<AtA_rowind[i]<<" ";
  cout<<endl;
  for (j=0; j<=n_row; j++)
    cout<<AtA_colptr[j]<<" ";
  cout<<endl;

  float *CV= new float[n_row], cluster_quality=0;

  power_method(AtA_val, AtA_rowind, AtA_colptr, n_row, CV, cluster_quality);
  for(i=0; i<n_row; i++)
    cout<<CV[i]<<" ";
  cout<<endl;
  cout<<"cluster_quality="<<cluster_quality<<endl;
}
