/* Implementation of DenseMatrix class
 * Copyright (c) 2003, Yuqiang Guan
 */

#include <assert.h>
#include <stdlib.h>
#include <math.h>
#include <iostream.h>
#include <fstream.h>

#include "DenseMatrix.h"

DenseMatrix::DenseMatrix ( int row, int col, float ** val) : Matrix (row, col)
  /* m_row, m_col, and m_val give the dimensionality and entries of matrix.
     norm[] may be used to store the L2 norm of each vector and
     L1_norm[] may be used to store the L1 norm of each vector
  */
{
  m_row = row;
  m_col = col;
  m_val = val;
  memory_used += (m_row+m_col)*sizeof(float);
  
}

DenseMatrix::~DenseMatrix ()
{
  if (norm != NULL)
    delete[] norm;
  if (L1_norm != NULL)
    delete[] L1_norm;
}

float DenseMatrix::dot_mult(float *x, int i) 
  /* this function computes the dot-product between the ith vector in the dense matrix
     and vector x; result is returned.
  */
{
  float result=0.0;
  for (int j = 0; j< m_row; j++)
    result += m_val[j][i]*x[j];
  return result;
}

void DenseMatrix::trans_mult(float *x, float *result) 
  /* Suppose A is the dense matrix, this function computes A^T x;
     the results is stored in array result[]
  */
{
  for (int i = 0; i < m_col; i++)
    result[i] = dot_mult(x, i);
}

void DenseMatrix::squared_trans_mult(float *x, float *result) 
  /* Suppose A is the dense matrix, this function computes A^T x;
     the results is stored in array result[]
  */
{
  for (int i = 0; i < m_col; i++)
    result[i] = squared_dot_mult(x, i) ;
}


float DenseMatrix::squared_dot_mult(float *x, int i) 
  /* this function computes the dot-product between the ith vector in the dense matrix
     and vector x; result is returned.
  */
{
  float result=0.0;
  for (int j = 0; j< m_row; j++)
    result += m_val[j][i]*x[j];
  return result*result;
}

void DenseMatrix::A_trans_A(int flag, int * index, int *pointers, float ** A_t_A)
  /* computes A'A given doc_IDs in the same cluster; index[] contains doc_IDs for all docs, 
     pointers[] gives the range of doc_ID belonging to the same cluster;
     the resulting matrix is stored in A_t_A[][] ( d by d matrix)
     Notice that Gene expression matrix is usually n by d where n is #genes; this matrix
     format is different from that of document matrix. 
     In the main memory the matrix is stored in the format of document matrix.
  */
{
  int i, j, k, clustersize = pointers[1]-pointers[0];

  if (flag >0)
    {
      for (i=0; i<m_row; i++)
	for (j=0; j<m_row; j++)
	  {
	    A_t_A[i][j] = 0;
	    for (k=0; k<clustersize; k++)
	      A_t_A[i][j] += m_val[i][index[k+pointers[0]]] * m_val[j][index[k+pointers[0]]];
	  }
    }
  else
    {
      for (i=0; i<m_row; i++)
	for (j=0; j<m_row; j++)
	  {
	    A_t_A[i][j] = 0;
	    for (k=0; k<clustersize; k++)
	      A_t_A[i][j] += m_val[i][k+pointers[0]] * m_val[j][k+pointers[0]];
	  }
    }
}

void DenseMatrix::right_dom_SV(int *cluster, int *cluster_size, int n_Clusters, float ** CV, float *cluster_quality, int flag)
{
  int i, *pointer = new int[n_Clusters+1], *index = new int[m_col], *range =new int[2];
  float ** A_t_A;

  pointer[0] =0;
  for (i=1; i<n_Clusters; i++)
    pointer[i] = pointer[i-1] + cluster_size[i-1];
  for (i=0; i<m_col; i++)
    {
      index[pointer[cluster[i]]] = i;
      pointer[cluster[i]] ++;
    }

  pointer[0] =0;
  for (i=1; i<=n_Clusters; i++)
    pointer[i] = pointer[i-1] + cluster_size[i-1];

  A_t_A =new (float*) [m_row];
  for (i=0; i<m_row; i++)
    A_t_A [i] = new float [m_row];

  if (flag <0)
    for (i=0; i<n_Clusters; i++)
      {
	range[0] = pointer[i];
	range[1] = pointer[i+1];
	A_trans_A(1, index, range, A_t_A); 
	/*for (int k=0; k< m_row; k++)
	  {
	    for (int j=0; j< m_row; j++)
	      cout<<A_t_A[k][j]<<" ";
	    cout<<endl;
	    }*/
	power_method(A_t_A, m_row, CV[i], CV[i], cluster_quality[i]);
	
      }
  else if ((flag >=0) && (flag <n_Clusters))
    {
      range[0] = pointer[flag];
      range[1] = pointer[flag+1];
      A_trans_A(1, index, range, A_t_A);
      power_method(A_t_A, m_row, CV[flag], CV[flag], cluster_quality[flag]);
    }
  else if ((flag >= n_Clusters) && (flag <2*n_Clusters))
    {
      range[0] = pointer[flag-n_Clusters];
      range[1] = pointer[flag-n_Clusters+1];
      A_trans_A(1, index, range, A_t_A);
      power_method(A_t_A, m_row, NULL, CV[flag-n_Clusters], cluster_quality[flag-n_Clusters]);
    }

  delete [] pointer;
  delete [] index;
  delete [] range;
  for (i=0; i< m_row; i++)
    delete [] A_t_A[i];
  delete [] A_t_A;
}


float DenseMatrix::euc_dis(float *x, int i, float norm_x)
  /* Given squared L2-norms of the vecs and v, norm[i] and norm_v,
     compute the Euc-dis between ith vec in the matrix and v,
     result is returned.
     Used (x-c)^T (x-c) = x^T x - 2 x^T c + c^T c
  */
{
  float result=0.0;
  for (int j=0; j< m_row; j++)
    result += x[j]*m_val[j][i];
  result *= -2.0;
  result += norm[i]+norm_x;
  return result;
}

void DenseMatrix::euc_dis(float *x, float norm_x, float *result)
  /* Given squared L2-norms of the vecs and x, norm[i] and norm_x,
     compute the Euc-dis between each vec in the matrix with x,  
     results are stored in array 'result'. 
     Since the matrix is dense, not taking advantage of 
     (x-c)^T (x-c) = x^T x - 2 x^T c + c^T c
     but the abstract class defition needs the parameter of 'norm_x'
  */
{
  for (int i = 0; i < m_col; i++)
    result[i] = euc_dis(x, i, norm_x);
}


void DenseMatrix::Kullback_leibler(float *x, float *result, int laplace)
{

  for (int i=0; i<m_col; i++)
    result [i] = Kullback_leibler(x, i, laplace);
}

float DenseMatrix::Kullback_leibler(float *x, int i, int laplace)
  // Given the KL-norm of vec[i], norm[i], (already considered prior)
  //   compute KL divergence between vec[i] in the matrix with x,
  //   result is returned. 'sum_log' is sum of logs of x[j]
  //   Take advantage of KL(p, q) = 
  //  \sum_i p_i log(p_i) - \sum_i p_i log(q_i) = norm[i] - \sum_i p_i log(q_i)
  // KL norm is in unit of nats NOT bits
  
{
  float result=0, row_inv_alpha=alpha/m_row;
  if (priors[i] >0)
    {
      switch(laplace)
	{
	case NOLAPLACE:
	  for (int j = 0; j < m_row; j++)
	    {
	      if(x[j] >0.0)
		result += m_val[j][i] * log(x[j]);
	      else if ( m_val[j][i]>0.0 )
		return HUGE_NUMBER; // if KL(vec[i], x) is inf. give it a huge number 1.0e20
	    }
	  result = norm[i]-result;
	  break;
	case CENTER_LAPLACE:
	  for (int j = 0; j < m_row; j++)
	    result += m_val[j][i] * log(x[j]+row_inv_alpha);
	  
	  result = norm[i]-result+log(1+alpha);
	  break;
	case PRIOR_LAPLACE:
	  for (int j = 0; j < m_row; j++)
	    result += (m_val[j][i]+row_inv_alpha) * log(x[j]);
	  
	  result = norm[i]-result/(1+alpha);
	  break;
	}
    }
  return result;
}

void DenseMatrix::Kullback_leibler(float *x, float *result, int laplace, float L1norm_x)
{

  for (int i=0; i<m_col; i++)
    result [i] = Kullback_leibler(x, i, laplace, L1norm_x);
}

float DenseMatrix::Kullback_leibler(float *x, int i, int laplace, float L1norm_x)
  // Given the KL-norm of vec[i], norm[i], (already considered prior)
  //   compute KL divergence between vec[i] in the matrix with x,
  //   result is returned. 'sum_log' is sum of logs of x[j]
  //   Take advantage of KL(p, q) = 
  //  \sum_i p_i log(p_i) - \sum_i p_i log(q_i) = norm[i] - \sum_i p_i log(q_i)
  // KL norm is in unit of nats NOT bits
  
{
  float result=0.0, row_inv_alpha=alpha/m_row;
  if (priors[i] >0)
    {
      switch(laplace)
	{
	case NOLAPLACE:
	  for (int j = 0; j < m_row; j++)
	    {
	      if(x[j] >0.0)
		result += m_val[j][i] * log(x[j]);
	      else if ( m_val[j][i]>0.0 )
		return HUGE_NUMBER; // if KL(vec[i], x) is inf. give it a huge number 1.0e20
	    }
	  
	  result = norm[i]-result+log(L1norm_x);
	  break;
	case CENTER_LAPLACE:
	  for (int j = 0; j < m_row; j++)
	    result += m_val[j][i] * log(x[j]+row_inv_alpha*L1norm_x);
	  
	  result = norm[i]-result+log((1+alpha)*L1norm_x);
	  break;
	case PRIOR_LAPLACE:
	  for (int j = 0; j < m_row; j++)
	    result += (m_val[j][i]+row_inv_alpha) * log(x[j]);
	  
	  result = norm[i]-result/(1+alpha);
	  break;
	}
    }
  return result;
}

float DenseMatrix::Jenson_Shannon(float *x, int i, float prior_x)
  // Given the prior of vec[i]
  //   compute KL divergence between vec[i] in the matrix with x,
  //   result in nats is returned. 
  //   
  
{
  float result=0.0, * p_bar,p1, p2; 
 
  if ((priors[i] >0) && (prior_x >0))
    {
      p1=priors[i]/(priors[i]+prior_x);
      p2=prior_x/(priors[i]+prior_x);
      p_bar = new float [m_row];
      for (int j=0; j< m_row; j++)
	p_bar[j] = p2*x[j] + p1*m_val[j][i];
      
      result = p1* Kullback_leibler(p_bar, i, NOLAPLACE)
	+ ::Kullback_leibler(x, p_bar, m_row)*p2; 
      
      //result /= L1_norm[i] + l1n_x;
      delete [] p_bar;
    }
  return result; // the real JS value should be this result devided by L1_norm[i]+l1n_x
}

void DenseMatrix::Jenson_Shannon(float *x, float *result, float prior_x)
{
  int i;
  for (i=0; i<m_col; i++)
    result[i] = Jenson_Shannon(x, i, prior_x);
  
}

void DenseMatrix::ComputeNorm_2()
  /* compute the squared L2 norms of each vector in the dense matrix
   */
{
  if (norm == NULL)
   {
     norm = new float [m_col];
     memory_used += m_col*sizeof(float);
   }
  for (int i = 0; i < m_col; i++)
    {
      norm[i] =0.0;
      for (int j = 0; j < m_row; j++)
	norm[i] += (m_val[j][i]) * (m_val[j][i]);
    }
}

void DenseMatrix::ComputeNorm_1()
 /* compute the L1 norms of each vector in the dense matrix
   */
{

  if (L1_norm == NULL)
    {
      L1_norm = new float [m_col];
      memory_used += m_col*sizeof(float);
    }
  for (int i = 0; i < m_col; i++)
    {
      L1_norm[i] =0.0;
      for (int j = 0; j < m_row; j++)
	L1_norm[i] += fabs(m_val[j][i]);
      L1_sum +=L1_norm [i];
    }
}

void DenseMatrix::ComputeNorm_KL(int laplace)
  /* compute the KL norms of each vector p_i in the dense matrix
     i.e. \sum_x p_i(x) \log p_i(x)
     the norm[i] is in unit of nats NOT bits
   */
{
  float row_inv_alpha = alpha/m_row;
  Norm_sum =0;
  if (norm == NULL)
    {
      norm = new float [m_col];
      memory_used += m_col*sizeof(float);
    }
  
  switch (laplace)
    {
    case NOLAPLACE:
    case CENTER_LAPLACE:
      for (int i = 0; i < m_col; i++)
	{
	  norm[i] =0.0;
	  for (int j = 0; j < m_row; j++)
	    if (m_val[j][i] > 0.0)
	      norm[i] += (m_val[j][i]) * log(m_val[j][i]);
	  Norm_sum +=norm[i]*priors[i];
	}
      break;
    case PRIOR_LAPLACE:
      for (int i = 0; i < m_col; i++)
	{
	  norm[i] =0.0;
	  for (int j = 0; j < m_row; j++)
	    norm[i] += (m_val[j][i]+row_inv_alpha) * log(m_val[j][i]+row_inv_alpha);
	  norm[i] = norm[i]/(1+alpha) +log(1+alpha);
	  Norm_sum +=norm[i]*priors[i];
	}
    }
   Norm_sum /= log(2.0);
}

void DenseMatrix::normalize_mat_L2()
  /* L2 normalize each vector in the dense matrix to have L2 norm 1
   */
{
  int i, j;
  float norm;

  for (i = 0; i < m_col; i++)
    {
      norm =0.0;
      for (j = 0; j < m_row; j++)
	norm += (m_val[j][i]) * (m_val[j][i]);
      if ( norm >0.0)
	{
	  norm =sqrt(norm);
	  for (j = 0; j < m_row; j++)
	    m_val[j][i] /= norm;
	}
    }
}


void DenseMatrix::normalize_mat_L1()
  /* L1 normalize each vector in the dense matrix to have L1 norm 1
   */
{
  int i, j;
  float norm;
  
  for (i = 0; i < m_col; i++)
    {
      norm =0.0;
      for (j = 0; j < m_row; j++)
	norm += fabs(m_val[j][i]);
      if (norm >0)
	for (j = 0; j < m_row; j++)
	  m_val[j][i] /= norm;
    }
}


void DenseMatrix::ith_add_CV(int i, float *CV)
{
  for ( int j = 0; j < m_row; j++)
    CV[j] += m_val[j][i];
}

void DenseMatrix::ith_add_CV_prior(int i, float *CV)
{
  for ( int j = 0; j < m_row; j++)
    CV[j] += priors[i]*m_val[j][i];
}

void DenseMatrix::CV_sub_ith(int i, float *CV)
{
  for ( int j = 0; j < m_row; j++)
    CV[j] -= m_val[j][i];
}

void DenseMatrix::CV_sub_ith_prior(int i, float *CV)
{
  for ( int j = 0; j < m_row; j++)
    CV[j] -= priors[i]*m_val[j][i];
}

float DenseMatrix::MutualInfo()
{
  float *rowSum= new float [m_row], MI=0.0;
  int i, j;
 
  for (i=0; i<m_row; i++)
    {
      rowSum[i] =0.0;
      for (j=0; j<m_col; j++)
	rowSum[i] += m_val[i][j]*priors[i];
    }
  
  for (i=0; i<m_row; i++)
    {
      float temp=0;
      for (j=0; j<m_col; j++)
	if (m_val[i][j] >0)
	  temp += m_val[i][j]*log(m_val[i][j]/rowSum[i]);
      MI += temp * priors[i];
    }
  delete [] rowSum;
  return (MI/log(2.0));
}

float DenseMatrix::exponential_kernel(float *x, int i, float norm_x, float sigma_squared)
  // this function computes the exponential kernel distance between i_th data with the centroid v
{
  float result=0.0;
  for (int j=0; j< m_row; j++)
    result += x[j]*m_val[j][i];
  result *= -2.0;
  result += norm[i]+norm_x;
  result = exp(result*0.5/sigma_squared);
  return result;
}

void DenseMatrix::exponential_kernel(float *x, float norm_x, float *result, float sigma_squared)
  //this function computes the exponential kernel distance between all data with the centroid x
{

  for (int i = 0; i < n_col; i++)
    result[i] = exponential_kernel(x, i, norm_x, sigma_squared);
}

float DenseMatrix::i_j_dot_product(int i, int j)
//this function computes  dot product between vectors i and j
{
  float result =0;
  for (int l=0; l<m_row; l++)
    result += m_val[l][i]*m_val[l][j];
  return result;
}


/*
// Does y = A * x, A is mxn, and y & x are mx1 and nx1 vectors
//   respectively  
void DenseMatrix::dmatvec(int m, int n, float **a, float *x, float *y)
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

// Does y = A' * x, A is mxn, and y & x are nx1 and mx1 vectors
//   respectively  
void DenseMatrix::dmatvecat(int m, int n, float **a, float *x, float *y)
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

// The procedure dqrbasis outputs an orthogonal basis spanned by the rows
//   of matrix a (using the QR Factorization of a ).  
void DenseMatrix::dqrbasis( float **q)
{
  int i,j;
  float *work = new float [m_row];

  for(i = 0; i < m_row;i++){
    dmatvec( i, m_col, q, m_val[i], work );
    dmatvecat( i, m_col, q, work, q[i] );
    for(j = 0; j < m_col;j++)
      q[i][j] = m_val[i][j] - q[i][j];
    dvec_l2normalize( m_col, q[i] );
  }
}

// The function dvec_l2normsq computes the square of the
//  Euclidean length (2-norm) of the double precision vector v 
float DenseMatrix::dvec_l2normsq( int dim, float *v )
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
// The function dvec_l2normalize normalizes the double precision
//   vector v to have 2-norm equal to 1 
void DenseMatrix::dvec_l2normalize( int dim, float *v )
{
  float nrm;

  nrm = sqrt(dvec_l2normsq( dim, v ));
  //if( nrm != 0 ) dvec_scale( 1.0/nrm, dim, v );
}
*/
