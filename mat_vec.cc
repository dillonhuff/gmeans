/*	mat_vec.cc
	Implementation of the mat_vec.cc
	Copyright(c) 2003 Yuqiang Guan 
*/

#include <iostream>
#include <math.h>
#include "mat_vec.h"

using namespace std;

/* The procedure dqrbasis outputs an orthogonal basis spanned by the rows
   of matrix a (using the QR Factorization of a ).  
void dqrbasis( int m, int n, float **a, float **q , float *work)
{
  int i,j;

  for(i = 0; i < m;i++){
    dmatvec( i, n, q, a[i], work );
    dmatvecat( i, n, q, work, q[i] );
    for(j = 0; j < n;j++)
      q[i][j] = a[i][j] - q[i][j];
    dvec_l2normalize( n, q[i] );
  }
}

 Does y = A * x, A is mxn, and y & x are mx1 and nx1 vectors
   respectively  
void dmatvec(int m, int n, float **a, float *x, float *y)
{
  float yi;
  int i,j;

  for(i = 0;i < m;i++){
    yi = 0.0;
    for(j = 0;j < n;j++){
      yi += a[i][j] * x[j];
    }
    y[i] = yi;
  }
}

 Does y = A' * x, A is mxn, and y & x are nx1 and mx1 vectors
   respectively  
void dmatvecat(int m, int n, float **a, float *x, float *y)
{
  float yi;
  int i,j;

  for(i = 0;i < n;i++){
    yi = 0.0;
    for(j = 0;j < m;j++){
      yi += a[j][i] * x[j];
    }
    y[i] = yi;
  }
}

 The function dvec_l2normsq computes the square of the
   Euclidean length (2-norm) of the double precision vector v 
float dvec_l2normsq( int dim, float *v )
{
  float length,tmp;
  int i;

  length = 0.0;
  for(i = 0;i < dim;i++){
    tmp = *v++;
    length += tmp*tmp;
  }
  return(length);
}

 The function dvec_l2normalize normalizes the double precision
   vector v to have 2-norm equal to 1 
void dvec_l2normalize( int dim, float *v )
{
  float nrm;
  
  nrm = sqrt(dvec_l2normsq( dim, v ));
  if( nrm != 0 ) dvec_scale( 1.0/nrm, dim, v );
}

void dvec_scale( float alpha, int n, float *v )
{
  int i;

  for(i = 0;i < n;i++){
    *v++ = *v * alpha;
  }
}
*/


float normalize_vec(float vec[], int n)
{
  float norm;
  int i;
  norm = 0.0;
  for (i = 0; i < n; i++)
    norm += vec[i] * vec[i];
  if (norm != 0) 
    {
      norm = sqrt(norm);
      for (i = 0; i < n; i++)
	vec[i] /= norm;
    }
  return norm;
}

float normalize_vec_1(float vec[], int n)
{
  float norm;
  int i;
  norm = 0.0;
  for (i = 0; i < n; i++)
    norm += fabs(vec[i]);
  if (norm != 0) 
    {
      for (i = 0; i < n; i++)
	vec[i] /= norm;
    }
  return norm;
}

void average_vec(float vec[], int n, int num)
{
  int i;
  for (i=0; i< n; i++)
    vec[i] = vec[i] / num;
} 

float Kullback_leibler(float *x, float *y, int n)
  // in nats NOT in bits
{
  float result = 0.0;
  for (int i=0; i<n; i++)
    if (x[i] >0.0)
      if ( y[i] >0.0 )
	result += x[i]*log(x[i]/y[i]);
      else
	return HUGE_NUMBER;
  return result;
}

float euclidian_distance(float *v1, float *v2, int n)
{
  float result = 0.0;
  int j;
  
  for (j = 0; j < n; j++)
    result += (v1[j] - v2[j]) * (v1[j] - v2[j]);
  return sqrt(result);
}

float norm_1(float *x, int n)
{
  float result =0.0;
  for (int i=0; i<n; i++)
    result += fabs(x[i]);
  return result;
}

float dot_mult(float *v1, float *v2, int n)
{
  float result = 0.0;
  int j;
  for (j = 0; j < n; j++)
    result += v1[j] * v2[j];
  return result;
}

float norm_2 (float vec[], int n)
  //compute squared L2 norm of vec
{
  float norm;
  int i;
  norm = 0.0;
  for (i = 0; i < n; i++)
    norm += vec[i] * vec[i];
  return norm; 
}

float KL_norm (float vec[], int n)
{
  float norm;
  int i;
  norm =0.0;
  for (i = 0; i < n; i++)
    {
      if (vec[i] > 0.0)
	norm += vec[i] * log(vec[i]) / log(2.0);
    }
  return norm;
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

void Ax (float **A, float *x, int dim1, int dim2, float *result)
  //for dense matrix, A is a matrix and x is a vector
{
  for (int i=0; i<dim1; i++)
    {
      result[i] =0;
      for (int j=0; j<dim2; j++)
	result[i] += A[i][j]*x[j];
    }
}

float x_dot_y(float *x, float *y, int dim1)
{
  float dot=0;
  for (int i=0; i<dim1; i++)
    dot += x[i]*y[i];
  return dot;
}

void power_method(float **A, int dim, float * CV, float *init, float & Lamda)
  // for dense square matrix A of dim by dim, initialized with vector init
{
 
  //RandomGenerator_MT19937 rand_gen;
  float *x=new float[dim], *y=new float[dim], norm, *temp=new float[dim], *old_x=new float[dim], dis=0;
  

  for (int i=0; i<dim; i++)
    x[i] = old_x[i] = init[i];
  do
    {
      Ax(A, x, dim, dim, y);
      norm = sqrt(norm_2(y, dim));
      for (int i=0; i<dim; i++)
	x[i] = y[i]/norm;
      Ax(A, x, dim, dim, temp);
      Lamda = x_dot_y(x, temp, dim);
      dis = euclidian_distance (x, old_x, dim);
      for (int i=0; i<dim; i++)
	old_x[i] = x[i];
      //cout <<dis<<" ";
    }
  while ( dis  > Power_Method_Epsilon  );
  if (CV != NULL)
    for (int i=0; i<dim; i++)
      CV[i] = x[i];
  //cout <<"Powermethod is done"<<endl;
}

void power_method(float *vals, int *rowinds, int *colptrs, int dim, float * CV, float *init, float & Lamda)
  // power_method works for square matrix only; so we have only 1 value for dimension

{
 
  //RandomGenerator_MT19937 rand_gen;
  float *x=new float[dim], *y=new float[dim], norm, *temp=new float[dim], *old_x=new float[dim], dis=0;
  
  //rand_gen.Set((unsigned)time(NULL));
  for (int i=0; i<dim; i++)
    x[i] = old_x[i] = init[i];
  do
    {
      Ax(vals, rowinds, colptrs, dim, dim, x, y);
      norm = sqrt(norm_2(y, dim));
      for (int i=0; i<dim; i++)
	x[i] = y[i]/norm;
      Ax(vals, rowinds, colptrs, dim, dim, x, temp);
      Lamda = x_dot_y(x, temp, dim);
      dis = euclidian_distance (x, old_x, dim);
      for (int i=0; i<dim; i++)
	old_x[i] = x[i];
      //cout <<dis<<" ";
    }
  while ( dis > Power_Method_Epsilon  );
  if (CV != NULL)
    for (int i=0; i<dim; i++)
      CV[i] = x[i];
  //cout <<"Powermethod is done"<<endl;
}

float mutual_info(float ** matrix, int r, int c)
  // compute mutual information for a dense matrix of size r by c
{
  float sum=0.0, mi=0.0, *margin_c, *margin_r;
  int i, j;
  
  margin_c = new float [c];
  margin_r = new float [r];
  for (i=0; i<r; i++)
    margin_r[i] =0.0;
  for (j=0; j<c; j++)
    margin_c[j] =0.0;

  for (i=0; i<r; i++)
    for (j=0; j<c; j++)
      {
	margin_r[i] += matrix[i][j];
	margin_c[j] += matrix[i][j];
      }
  for (i=0; i<r; i++)
    sum += margin_r[i];
  
  for (i=0; i<r; i++)
    for (j=0; j<c; j++)
      if (matrix[i][j]>0)
	mi += matrix[i][j]* log(matrix[i][j]/(margin_r[i]*margin_c[j]));
	
  mi = mi/sum + log(sum);
  mi /= log(2.0);
  delete [] margin_c;
  delete [] margin_r;
  return mi;
}

float mutual_info(float ** matrix, int r, int c, float *prior)
  // compute mutual information for a dense matrix of size r by c
  // each row is L1-normalized
{
  float mi=0.0, *margin_c;
  int i, j;
  
  margin_c = new float [c];
  for (i=0; i<c; i++)
    margin_c[i] =0.0;

  for (i=0; i<r; i++)
    for (j=0; j<c; j++)
      {
	margin_c[j] += matrix[i][j]*prior[i];
      }
  
  for (i=0; i<r; i++)
    {
      float temp =0;
      for (j=0; j<c; j++)
	if (matrix[i][j]>0)
	  temp += matrix[i][j]* log(matrix[i][j]/margin_c[j]);
      mi += temp*prior[i];
    }
 
  mi /= log(2.0);
  
  delete [] margin_c;
  return mi;
}
