/* Implementation of SparseMatrixDouble class
 * Copyright (c) 2002, Yuqiang Guan
 */

#include <assert.h>
#include <stdlib.h>
#include <math.h>
#include "SparseMatrixDouble.h"


SparseMatrixDouble::SparseMatrixDouble(int row, int col, int nz, float *val, int *rowind, int *colptr) : Matrix (row, col)
{
  //int i;

	n_row = row;
	n_col = col;
	n_nz  = nz;
	norm  = NULL;
	L1_norm = NULL;
	/*vals = new double[n_nz];
	rowinds = new int[n_nz];
	colptrs = new int[n_col+1];

	for (i = 0; i < n_nz; i++)
	{
		vals[i] = val[i];
		rowinds[i] = rowind[i];
	}
	for (i = 0; i < n_col+1; i++)
	colptrs[i] = colptr[i];*/
	vals = val;
	rowinds = rowind;
	colptrs = colptr;
}

/* void SparseMatrixDouble::prepare_trans_mult()
{
	int i, j;

	// setting up data structures for transpose multiplication
	int *rp = new int[n_row];
	rowptrs = new int[n_row+1];
	colinds = new int[n_nz];
	vals2 = new double[n_nz];
	
	for (i = 0; i < n_row+1; i++)
		rowptrs[i] = 0;
	for (i = 0; i < n_nz; i++)
		rowptrs[rowinds[i]+1]++;
	for (i = 1; i < n_row+1; i++)
		rowptrs[i] += rowptrs[i-1];
	
	assert(rowptrs[n_row] == n_nz && colptrs[n_col] ==n_nz);

	for (i = 0; i < n_row; i++)
		rp[i] = rowptrs[i];
	for (i = 0; i < n_col; i++)
		for (j = colptrs[i]; j < colptrs[i+1]; j++)
		{
			colinds[rp[rowinds[j]]] = i;
			vals2[rp[rowinds[j]]] = vals[j];
			rp[rowinds[j]]++;
		}
}*/
/*
SparseMatrixDouble::~SparseMatrixDouble()
{
  delete[] vals;
	delete[] rowinds;
	delete[] colptrs;

	// ... delete data structures for transpose multiplication
	delete[] rowptrs;
	delete[] colinds;
	delete[] vals2; 
}
*/
float SparseMatrixDouble::operator()(int i, int j) const
{
	assert(i >= 0 && i < n_row && j >= 0 && j < n_col);

	for (int t = colptrs[j]; t < colptrs[j+1]; j++)
		if (rowinds[t] == i) return vals[t];
	return 0.0;
}

void SparseMatrixDouble::trans_mult(float *x, float *result) 
{
  for (int i = 0; i < n_col; i++)
    {
      result[i] = 0.0;
      for (int j = colptrs[i]; j < colptrs[i+1]; j++)
	result[i] += vals[j] * x[rowinds[j]];
    }

	// accelerated version
/*	int i;
	for (i = 0; i < n_col; i++)
		result[i] = 0.0;

	for (i = 0; i < n_row; i++)
		if (x[i] != 0)
			for (int j = rowptrs[i]; j < rowptrs[i+1]; j++)
				result[colinds[j]] += vals2[j] * x[i]; */
}

//compute the dot-product between the ith vector (in the sparse matrix) with vector v
float SparseMatrixDouble::dot_mult(float *v, int i)
{
  float result=0.0;
  for (int j = colptrs[i]; j < colptrs[i+1]; j++)
    result += vals[j] * v[rowinds[j]];
  return result;
}

void SparseMatrixDouble::euc_dis(float *x, float *result)
{
  float *temp_vec = new float [n_row];

  for (int i = 0; i < n_col; i++)
    {
      for (int j=0; j< n_row; j++)
	temp_vec[j] =0.0;
      for (int j = colptrs[i]; j < colptrs[i+1]; j++)
	temp_vec[rowinds[j]] = vals[j];
      result[i] =0.0;
      for (int j=0; j< n_row; j++)
	result[i] += (x[j]-temp_vec[j])*(x[j]-temp_vec[j]);
      result[i] = sqrt(result[i]);
    }
}

void SparseMatrixDouble::euc_dis(float *x, float norm_x, float *result)
{

  for (int i = 0; i < n_col; i++)
    {
      result[i] = 0.0;
      for (int j = colptrs[i]; j < colptrs[i+1]; j++)
	result[i] += vals[j] * x[rowinds[j]];

      result[i] *= -2.0;
      result[i] += norm[i]+norm_x;
      result[i] = sqrt(result[i]);
    }
}

float SparseMatrixDouble::euc_dis(float *v, int i, float norm_v)
{
  float result=0.0;
  for (int j = colptrs[i]; j < colptrs[i+1]; j++)
    result += vals[j] * v[rowinds[j]];
  result *= -2.0;
  result += norm[i]+norm_v;
  result = sqrt(result);
  return result;
}

//norm[i] stores the 'KL-norm' of vector[i]
void SparseMatrixDouble::Kullback_leibler(float *x, float *result)
{
  for (int i = 0; i < n_col; i++)
    {
      result[i] = 0.0;
      
      for (int j = colptrs[i]; j < colptrs[i+1]; j++)
	{
	  if (x[rowinds[j]] >0.0)
	    result[i] += vals[j] * log(x[rowinds[j]]) /log(2.0);
	  else
	    {
	      result[i] = HUGE_NUMBER;
	      break;
	    }
	}
      result[i]= norm[i]-result[i];
    }
}

float SparseMatrixDouble::Kullback_leibler(float *x, int i)
{
  float result=0.0;
  for (int j = colptrs[i]; j < colptrs[i+1]; j++)
    {
      if(x[rowinds[j]] >0.0)
	result += vals[j] * log(x[rowinds[j]]) /log(2.0);
      else
	return norm[i]-HUGE_NUMBER;
    }
  result = norm[i]-result;
  return result;
}

void SparseMatrixDouble::ComputeNorm_2()
{
  if (norm == NULL)
    norm = new float [n_col];

  for (int i = 0; i < n_col; i++)
    {
      norm[i] =0.0;
      
      for (int j = colptrs[i]; j < colptrs[i+1]; j++)
	norm[i] += vals[j] * vals[j];
    }
}

void SparseMatrixDouble::ComputeNorm_KL()
{
  if (norm == NULL)
    norm = new float [n_col];

  for (int i = 0; i < n_col; i++)
    {
      norm[i] =0.0;
      
      for (int j = colptrs[i]; j < colptrs[i+1]; j++)
	norm[i] += vals[j] * log(vals[j]) / log(2.0);
    }
}

void SparseMatrixDouble::normalize_mat_L1()
{
  int i, j;
  
  if (L1_norm == NULL)
    L1_norm = new float [n_col];
  
  for (i = 0; i < n_col; i++)
    {
      L1_norm[i] = 0.0;
      for (j = colptrs[i]; j < colptrs[i+1]; j++)
	L1_norm[i] += fabs(vals[j]);
      
      //              assert(norm > 0);
      /*		if (norm == 0)
			{
			cerr << "column " << i << " has 0 norm\n";
			}*/
      if(L1_norm[i]!=0)
	{
	  for (j = colptrs[i]; j < colptrs[i+1]; j++)
	    vals[j] /= L1_norm[i];
	}
    }
}

float SparseMatrixDouble::getNorm(int i)
{
  return norm[i];
}

float SparseMatrixDouble::getL1Norm(int i)
{
  return L1_norm[i];
}

void SparseMatrixDouble::ith_scale_add_CV(int i, float *CV)
{
  for (int j = colptrs[i]; j < colptrs[i+1]; j++)
    CV[rowinds[j]] += vals[j]*L1_norm[i];
}

void SparseMatrixDouble::ith_add_CV(int i, float *CV)
{
  for (int j = colptrs[i]; j < colptrs[i+1]; j++)
    CV[rowinds[j]] += vals[j];
}

void SparseMatrixDouble::CV_sub_ith(int i, float *CV)
{
  for (int j = colptrs[i]; j < colptrs[i+1]; j++)
    CV[rowinds[j]] -= vals[j];
}

void SparseMatrixDouble::CV_sub_ith_scale(int i, float *CV)
{
  for (int j = colptrs[i]; j < colptrs[i+1]; j++)
    CV[rowinds[j]] -= vals[j]*L1_norm[i];
}
