/* Implementation of SparseMatrix class
 * Copyright (c) 2003, Yuqiang Guan
 */

#include <assert.h>
#include <stdlib.h>
#include <iostream.h>
#include <math.h>
#include "SparseMatrix.h"


SparseMatrix::SparseMatrix(int row, int col, int nz, float *val, int *rowind, int *colptr) : Matrix (row, col)
{
  n_row = row;
  n_col = col;
  n_nz  = nz;
  vals = val;
  rowinds = rowind;
  colptrs = colptr;
  memory_used += (n_row+n_col)*sizeof(int)+n_nz*sizeof(float);
}


SparseMatrix::~SparseMatrix()
{
  if (norm != NULL)
    delete[] norm;
  if (L1_norm != NULL)
    delete[] L1_norm;
}


float SparseMatrix::operator()(int i, int j) const
{
  assert(i >= 0 && i < n_row && j >= 0 && j < n_col);
  
  for (int t = colptrs[j]; t < colptrs[j+1]; t++)
    if (rowinds[t] == i) return vals[t];
  return 0.0;
}

float SparseMatrix::dot_mult(float *v, int i)
  /*compute the dot-product between the ith vector (in the sparse matrix) with vector v
    result is returned.
   */
{
  float result=0.0;
  for (int j = colptrs[i]; j < colptrs[i+1]; j++)
    result += vals[j] * v[rowinds[j]];
  return result;
}

void SparseMatrix::trans_mult(float *x, float *result) 
  /*compute the dot-product between every vector (in the sparse matrix) with vector v
    results are stored in array 'result'.
   */
{
  for (int i = 0; i < n_col; i++)
      result[i] = dot_mult(x, i);
}


float SparseMatrix::squared_dot_mult(float *v, int i)
  /*compute the dot-product between the ith vector (in the sparse matrix) with vector v
    result is returned.
   */
{
  float result=0.0;
  for (int j = colptrs[i]; j < colptrs[i+1]; j++)
    result += vals[j] * v[rowinds[j]];
  return result*result;
}


void SparseMatrix::squared_trans_mult(float *x, float *result) 
  /*compute the dot-product between every vector (in the sparse matrix) with vector v
    results are stored in array 'result'.
   */
{
  for (int i = 0; i < n_col; i++)
    result[i] = squared_dot_mult(x, i);
}



void SparseMatrix::A_trans_A(int flag, int * index, int *pointers, float **A_t_A, int & nz_counter)
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
  
  if (flag >0)
    {
      for (k=0; k<clustersize; k++)
	for (l = colptrs[index[k+pointers[0]]]; l < colptrs[index[k+pointers[0]]+1]; l++)
	  for (j = colptrs[index[k+pointers[0]]]; j < colptrs[index[k+pointers[0]]+1]; j++)
	    {
	      A_t_A[rowinds[l]][rowinds[j]] += vals[l]*vals[j];
	    }
    }
  else
    {
      for (k=0; k<clustersize; k++)
	for (l = colptrs[k+pointers[0]]; l < colptrs[k+pointers[0]+1]; l++)
	  for (j = colptrs[k+pointers[0]]; j < colptrs[k+pointers[0]+1]; j++)
	    {
	      A_t_A[rowinds[l]][rowinds[j]] += vals[l]*vals[j];
	    }
    }
  for (i=0; i<n_row; i++)
    for (j=0; j<n_row; j++)
      if (A_t_A[i][j] >0)
	nz_counter++;
}

void SparseMatrix::dense_2_sparse(int* AtA_colptr, int *AtA_rowind, float *AtA_val, float **A_t_A)
{
  int i, l, j, k;
  k=0;
  AtA_colptr[0] = 0;
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
}

void SparseMatrix::right_dom_SV(int *cluster, int *cluster_size, int n_Clusters, float ** CV, float *cluster_quality, int flag)
  /* 
     flag == -1, then update all the centroids and the cluster qualities; 
     0 <= flag < n_Clusters , then update the centroid and the cluster quality specified by 'flag';
     n_Clusters <= flag <2*n_Clusters, then update the cluster quality specified by 'flag-n_Clusters';
  */
{
  int i, *pointer = new int[n_Clusters+1], *index = new int[n_col], *range =new int[2], nz_counter;
  int *AtA_rowind =NULL, *AtA_colptr=NULL;
  float *AtA_val =NULL;
  float **A_t_A;
  A_t_A =new (float*) [n_row];
  for (i=0; i<n_row; i++)
    A_t_A [i] = new float [n_row];

  pointer[0] =0;
  for (i=1; i<n_Clusters; i++)
    pointer[i] = pointer[i-1] + cluster_size[i-1];
  for (i=0; i<n_col; i++)
    {
      index[pointer[cluster[i]]] = i;
      pointer[cluster[i]] ++;
    }

  pointer[0] =0;
  for (i=1; i<=n_Clusters; i++)
    pointer[i] = pointer[i-1] + cluster_size[i-1];

  if (flag <0) // compute eigval and eigvec for all clusters
    {
      for (i=0; i<n_Clusters; i++)
	{
	  range[0] = pointer[i];
	  range[1] = pointer[i+1];
	  A_trans_A(1, index, range, A_t_A, nz_counter);
	  AtA_val = new float [nz_counter];
	  AtA_rowind = new int [nz_counter];
	  AtA_colptr = new int [n_row+1];
	  dense_2_sparse(AtA_colptr, AtA_rowind, AtA_val, A_t_A);
	  /*
	  for (int l=0; l< n_row; l++)
	    {
	      for (int j=0; j< n_row; j++)
		cout<<A_t_A[l][j]<<" ";
	      cout<<endl;
	    }
	 
	  for (int l=0; l<= n_row; l++)
	    cout<<AtA_colptr[l]<<" ";
	  cout<<endl;
	  for (int l=0; l<nz_counter; l++)
	    cout<<AtA_rowind[l]<<" ";
	  cout<<endl;
	  for (int l=0; l<nz_counter; l++)
	    cout<<AtA_val[l]<<" ";
	  cout<<endl<<endl;
	  */
	  power_method(AtA_val, AtA_rowind, AtA_colptr, n_row, CV[i], CV[i], cluster_quality[i]);
	  //for (int l=0; l<n_row; l++)
	  //  cout<<CV[i][l]<<" ";
	  //cout<<cluster_quality[i]<<endl;
	  delete [] AtA_val;
	  delete [] AtA_rowind;
	  delete [] AtA_colptr;
	}
      for (i=0; i< n_row; i++)
	delete [] A_t_A[i];
      delete [] A_t_A;
    } 
  else if ((flag >=0) && (flag<n_Clusters)) //compute eigval and eigvec for cluster 'flag'
    {
      range[0] = pointer[flag];
      range[1] = pointer[flag+1];
      A_trans_A(1, index, range, A_t_A, nz_counter);
      AtA_val = new float [nz_counter];
      AtA_rowind = new int [nz_counter];
      AtA_colptr = new int [n_row+1];
      dense_2_sparse(AtA_colptr, AtA_rowind, AtA_val, A_t_A);
      for (i=0; i< n_row; i++)
	delete [] A_t_A[i];
      delete [] A_t_A;
      power_method(AtA_val, AtA_rowind, AtA_colptr, n_row, CV[flag], CV[flag], cluster_quality[flag]);
      delete [] AtA_val;
      delete [] AtA_rowind;
      delete [] AtA_colptr;
    }
  else if ((flag >= n_Clusters) && (flag <= 2*n_Clusters))
    // compute eigval ONLY for cluster 'flag-n_Clusters', eigvec for  cluster 'flag-n_Clusters' no change
    {
      range[0] = pointer[flag-n_Clusters];
      range[1] = pointer[flag-n_Clusters+1];
      A_trans_A(1, index, range, A_t_A, nz_counter);
      AtA_val = new float [nz_counter];
      AtA_rowind = new int [nz_counter];
      AtA_colptr = new int [n_row+1];
      dense_2_sparse(AtA_colptr, AtA_rowind, AtA_val, A_t_A);
      for (i=0; i< n_row; i++)
	delete [] A_t_A[i];
      delete [] A_t_A;
      power_method(AtA_val, AtA_rowind, AtA_colptr, n_row, NULL, CV[flag-n_Clusters], cluster_quality[flag-n_Clusters]);
      delete [] AtA_val;
      delete [] AtA_rowind;
      delete [] AtA_colptr;
    }

  delete [] pointer;
  delete [] index;
  delete [] range;
}


float SparseMatrix::euc_dis(float *v, int i, float norm_v)
  /* Given L2-norms of the vecs and v, norm[i] and norm_v,
     compute the squared Euc-dis between ith vec in the matrix and v,
     result is returned.
     Take advantage of (x-c)^T (x-c) = x^T x - 2 x^T c + c^T c
  */
{
  float result=0.0;
  for (int j = colptrs[i]; j < colptrs[i+1]; j++)
    result += vals[j] * v[rowinds[j]];
  result *= -2.0;
  result += norm[i]+norm_v;
  return result;
}

void SparseMatrix::euc_dis(float *x, float norm_x, float *result)
  /* Given L2-norms of the vecs and x, norm[i] and norm_x,
     compute the squared Euc-dis between each vec in the matrix with x,  
     results are stored in array 'result'. 
     Take advantage of (x-c)^T (x-c) = x^T x - 2 x^T c + c^T c
  */
{
  for (int i = 0; i < n_col; i++)
    result[i] = euc_dis(x, i, norm_x);
}

float SparseMatrix::Kullback_leibler(float *x, int i, int laplace, float L1norm_x)
  /* Given the L1_norms of vec[i] and x, (vec[i] need be normalized before function-call
     compute KL divergence between vec[i] in the matrix with x,
     result is returned. 
     Take advantage of KL(p, q) = \sum_i p_i log(p_i) - \sum_i p_i log(q_i) = norm[i] - \sum_i p_i log(q_i)
     the KL is in unit of nats NOT bits.
  */
{
  float result=0.0, row_inv_alpha=alpha/n_row;
  if (priors[i] >0)
    {
      switch(laplace)
	{
	case NOLAPLACE:
	  for (int j = colptrs[i]; j < colptrs[i+1]; j++)
	    {
	      if(x[rowinds[j]] >0.0)
		result += vals[j] * log(x[rowinds[j]]);
	      else
		return HUGE_NUMBER; // if KL(vec[i], x) is inf. give it a huge number 1.0e20
	    }
     
	  result = norm[i]-result+log(L1norm_x);
	  break;
	case CENTER_LAPLACE:
	  // this vector alpha is alpha (given by user) divided by |Y|,
	  //row_inv_alpha is to make computation faster
	  for (int j = colptrs[i]; j < colptrs[i+1]; j++)
	    result += vals[j] * log(x[rowinds[j]]+row_inv_alpha*L1norm_x) ;
	  
	  result = norm[i]-result+log((1+alpha)*L1norm_x);
	  break;
	case PRIOR_LAPLACE:
	  // this vector alpha is alpha (given by user) divided by |X|*|Y|,
	  //row_alpha is its L1-norm
	  for (int j = colptrs[i]; j < colptrs[i+1]; j++)
	    result += (vals[j]+row_inv_alpha) * log(x[rowinds[j]]) ;
	  
	  result = norm[i]-result/(1+alpha);
	  break;
	}
    }
  return result;
}

void SparseMatrix::Kullback_leibler(float *x, float *result, int laplace, float L1norm_x)
  // Given the KL-norm of the vecs, norm[i] (already considered prior),
  //   compute KL divergence between each vec in the matrix with x,
  //   results are stored in array 'result'. 
  //   Take advantage of KL(p, q) = \sum_i p_i log(p_i) - \sum_i p_i log(q_i) = norm[i] - \sum_i p_i log(q_i)
  
{
  int i;
  for ( i = 0; i < n_col; i++)
    result[i] = Kullback_leibler(x, i, laplace,L1norm_x);  
}

float SparseMatrix::Kullback_leibler(float *x, int i, int laplace)
  /* Given the L1_norms of vec[i] and x, (vec[i] and x need be normalized before function-call
     compute KL divergence between vec[i] in the matrix with x,
     result is returned. 
     Take advantage of KL(p, q) = \sum_i p_i log(p_i) - \sum_i p_i log(q_i) = norm[i] - \sum_i p_i log(q_i)
     the KL is in unit of nats NOT bits.
  */
{
  float result=0.0, row_inv_alpha=alpha/n_row;
  if (priors[i] >0)
    {
      switch(laplace)
	{
	case NOLAPLACE:
	  for (int j = colptrs[i]; j < colptrs[i+1]; j++)
	    {
	      if(x[rowinds[j]] >0.0)
		result += vals[j] * log(x[rowinds[j]]);
	      else
		return HUGE_NUMBER; // if KL(vec[i], x) is inf. give it a huge number 1.0e20
	    }
     
	  result = norm[i]-result;
	  break;
	case CENTER_LAPLACE:
	  // this vector alpha is alpha (given by user) divided by |Y|,
	  //row_inv_alpha is to make computation faster
	  for (int j = colptrs[i]; j < colptrs[i+1]; j++)
	    result += vals[j] * log(x[rowinds[j]]+row_inv_alpha) ;
	  
	  result = norm[i]-result+log(1+alpha);
	  break;
	case PRIOR_LAPLACE:
	  // this vector alpha is alpha (given by user) divided by |X|*|Y|,
	  //row_alpha is its L1-norm
	  for (int j = colptrs[i]; j < colptrs[i+1]; j++)
	    result += (vals[j]+row_inv_alpha) * log(x[rowinds[j]]) ;
	  
	  result = norm[i]-result/(1+alpha);
	  break;
	}
    }
  return result;
}

void SparseMatrix::Kullback_leibler(float *x, float *result, int laplace)
  // Given the KL-norm of the vecs, norm[i] (already considered prior),
  //   compute KL divergence between each vec in the matrix with x,
  //   results are stored in array 'result'. 
  //   Take advantage of KL(p, q) = \sum_i p_i log(p_i) - \sum_i p_i log(q_i) = norm[i] - \sum_i p_i log(q_i)
  
{
  int i;
  for ( i = 0; i < n_col; i++)
    result[i] = Kullback_leibler(x, i, laplace);  
}

float SparseMatrix::Jenson_Shannon(float *x, int i, float prior_x)
  /* Given the prior of vec[i],
     compute JS divergence between vec[i] in the data matrix with x,
     result in nats is returned. 
  */
{
  float result=0.0, * p_bar, p1, p2;
  
  if ((priors[i] >0) && (prior_x >0))
    {
      p1=priors[i]/(priors[i]+prior_x);
      p2=prior_x/(priors[i]+prior_x);
      p_bar = new float [n_row];
      for (int j=0; j< n_row; j++)
	p_bar[j] = p2*x[j];
      
      for (int j = colptrs[i]; j < colptrs[i+1]; j++)
	p_bar[rowinds[j]] += p1*vals[j];

      result = p1* Kullback_leibler(p_bar, i, NOLAPLACE)
	+ ::Kullback_leibler(x, p_bar, n_row)*p2 ; 
      delete [] p_bar;
    }
  return result; // the real JS value should be this result devided by L1_norm[i]+l1n_x
}

void SparseMatrix::Jenson_Shannon(float *x, float *result, float prior_x)
  /* Given the prior of vec[i] and x; vec[i] and x are all normalized
     compute JS divergence between all vec[i] in the data matrix with x,
     result in nats. 
  */
{  
  int i;
  for ( i = 0; i < n_col; i++)
    result[i] = Jenson_Shannon(x, i, prior_x);
}

void SparseMatrix::ComputeNorm_2()
  /* compute the squared L-2 norms for each vec in the matrix
     first check if array 'norm' has been given memory space
   */
{
  if (norm == NULL)
    {
      norm = new float [n_col];
      memory_used += n_col*sizeof(float);
    }
  for (int i = 0; i < n_col; i++)
    {
      norm[i] =0.0;
      for (int j = colptrs[i]; j < colptrs[i+1]; j++)
	norm[i] += vals[j] * vals[j];
    }
}

void SparseMatrix::ComputeNorm_1()
  /* compute the squared L-2 norms for each vec in the matrix
     first check if array 'norm' has been given memory space
   */
{
  if (L1_norm == NULL)
    {
      L1_norm = new float [n_col];
      memory_used += n_col*sizeof(float);
    }
  for (int i = 0; i < n_col; i++)
    {
      L1_norm[i] =0.0;
      for (int j = colptrs[i]; j < colptrs[i+1]; j++)
	L1_norm[i] += vals[j] ;
    }
}

void SparseMatrix::ComputeNorm_KL(int laplace)
  // the norm[i] is in unit of nats NOT bits
{
  float row_inv_alpha=alpha/n_row;

  if (norm == NULL)
    {
      norm = new float [n_col];
      memory_used += n_col*sizeof(float);
    }
  Norm_sum=0;
  switch (laplace)
    {
    case NOLAPLACE:
    case CENTER_LAPLACE:
      for (int i = 0; i < n_col; i++)
	{
	  norm[i] =0.0;
	  for (int j = colptrs[i]; j < colptrs[i+1]; j++)
	    norm[i] += vals[j] * log(vals[j]);
	  Norm_sum +=norm[i]*priors[i];
	}
      break;
    case PRIOR_LAPLACE:
      for (int i = 0; i < n_col; i++)
	{
	  norm[i] =0.0;
	  for (int j = colptrs[i]; j < colptrs[i+1]; j++)
	    norm[i] += (vals[j]+row_inv_alpha) * log(vals[j]+row_inv_alpha) ;
	  norm[i] += (n_row-(colptrs[i+1]-colptrs[i]))*row_inv_alpha*log(row_inv_alpha) ;
	  norm[i] = norm[i]/(1+alpha) +log(1+alpha);
	  Norm_sum +=norm[i]*priors[i];
	}
    }
  Norm_sum /= log(2.0);
}

void SparseMatrix::normalize_mat_L2()
  /* compute the L_2 norms for each vec in the matrix and L_2-normalize them
     first check if array 'norm' has been given memory space
   */
{
  int i, j;
  float norm;

  for (i = 0; i < n_col; i++)
    {
      norm =0.0;
      for (j = colptrs[i]; j < colptrs[i+1]; j++)
	norm += vals[j] * vals[j];
      if( norm >0.0 )
	{
	  norm = sqrt(norm);
	  for (j = colptrs[i]; j < colptrs[i+1]; j++)
	    vals[j] /= norm;
	}
    }
}

void SparseMatrix::normalize_mat_L1()
  /* compute the L_1 norms for each vec in the matrix and L_1-normalize them
     first check if array 'L1_norm' has been given memory space
   */
{
  int i, j;
  float norm;

  for (i = 0; i < n_col; i++)
    {
      norm =0.0;
      for (j = colptrs[i]; j < colptrs[i+1]; j++)
	norm += fabs(vals[j]);
      if(norm >0)
	{
	  for (j = colptrs[i]; j < colptrs[i+1]; j++)
	    vals[j] /= norm;
	}
    }
}


void SparseMatrix::ith_add_CV(int i, float *CV)
{
  for (int j = colptrs[i]; j < colptrs[i+1]; j++)
    CV[rowinds[j]] += vals[j];
}

void SparseMatrix::ith_add_CV_prior(int i, float *CV)
{
  for (int j = colptrs[i]; j < colptrs[i+1]; j++)
    CV[rowinds[j]] += priors[i]*vals[j];
}

void SparseMatrix::CV_sub_ith(int i, float *CV)
{
  for (int j = colptrs[i]; j < colptrs[i+1]; j++)
    CV[rowinds[j]] -= vals[j];
}

void SparseMatrix::CV_sub_ith_prior(int i, float *CV)
{
  for (int j = colptrs[i]; j < colptrs[i+1]; j++)
    CV[rowinds[j]] -= priors[i]*vals[j];
}

float SparseMatrix::MutualInfo()
{
  float *rowSum= new float [n_row], MI=0.0;
  int i;

  for (i=0; i<n_row; i++)
    rowSum[i] =0.0;

  for (i=0; i<n_col; i++)
    for (int j = colptrs[i]; j < colptrs[i+1]; j++)
      rowSum[rowinds[j]] +=vals[j]*priors[i];
  
  
  for (i=0; i<n_col; i++)
    {
      float temp=0;
      for (int j = colptrs[i]; j < colptrs[i+1]; j++)
	temp += vals[j]*log(vals[j]/rowSum[rowinds[j]]);
      MI += temp *priors[i];
    }
  delete [] rowSum;
  return(MI/log(2.0));
}

float SparseMatrix::exponential_kernel(float *v, int i, float norm_v, float sigma_squared)
  // this function computes the exponential kernel distance between i_th data with the centroid v
{
  float result=0.0;
  for (int j = colptrs[i]; j < colptrs[i+1]; j++)
    result += vals[j] * v[rowinds[j]];
  result *= -2.0;
  result += norm[i]+norm_v;
  result = exp(result*0.5/sigma_squared);
  return result;
}

void SparseMatrix::exponential_kernel(float *x, float norm_x, float *result, float sigma_squared)
  //this function computes the exponential kernel distance between all data with the centroid x
{
  for (int i = 0; i < n_col; i++)
    result[i] = exponential_kernel(x, i, norm_x, sigma_squared);
}

float SparseMatrix::i_j_dot_product(int i, int j)
//this function computes  dot product between vectors i and j
{
  float result =0;

  if (i==j)
    for ( int l= colptrs[i]; l < colptrs[i+1]; l++)
      result += vals[l]*vals[l];
  else
    for ( int l= colptrs[i]; l < colptrs[i+1]; l++)
      for ( int k= colptrs[j]; k < colptrs[j+1]; k++)
	if(rowinds[l] == rowinds[k])
	  result += vals[l]*vals[k];
  return result;
}



