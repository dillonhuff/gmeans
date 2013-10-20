/*	Sparse Matrix header file
 *		SparseMatrix.h
 *	Copyright (c) 2003, Yuqiang Guan
 */

#if !defined(_SPARSE_MATRIX_H_)
#define _SPARSE_MATRIX_H_

#include "Matrix.h"
#include "mat_vec.h"
#include "Constants.h"

class SparseMatrix : public Matrix
{
 protected:
  int	n_row, n_col, n_nz;
  float	*vals;
  int	*rowinds;
  int	*colptrs;
  
  bool row_ind_sorted;
  //void sort_row_ind();
  void A_trans_A(int flag, int * index, int *pointers, float **AtA, int &nz_couter);
  
 public:
  SparseMatrix(int row, int col, int nz, float *val, int *rowind, int *colptr);
  ~SparseMatrix();
  inline float&	val(int i) { return vals[i]; }
  inline int&  	row_ind(int i) { return rowinds[i]; }
  inline int&  	col_ptr(int i) { return colptrs[i]; }
  float	       	operator() (int i, int j) const;
  virtual void  trans_mult(float *x, float *result) ;	
  virtual void  squared_trans_mult(float *x, float *result); 
  virtual float dot_mult(float *v, int i);
  virtual float squared_dot_mult(float *v, int i);
  void dense_2_sparse(int* AtAcolptr, int *AtA_rowind, float *AtA_val, float **AtA);
  virtual void  right_dom_SV(int *cluster, int *cluster_size, int n_Clusters, float ** CV, float *cluster_quality, int flag);
  virtual void  euc_dis(float *x, float norm_x, float *result);
  virtual float euc_dis(float *v, int i, float norm_v);
  virtual void  Kullback_leibler(float *x, float *result,int laplace);
  virtual float Kullback_leibler(float *x, int i, int laplace);
  virtual void  Kullback_leibler(float *x, float *result,int laplace, float l1norm_X);
  virtual float Kullback_leibler(float *x, int i, int laplace, float l1norm_X);
  virtual float Jenson_Shannon(float *x, int i, float l1n_x);
  virtual void  Jenson_Shannon(float *x, float *result, float prior_x);
  virtual void  ComputeNorm_2();
  virtual void  ComputeNorm_1();
  virtual void  ComputeNorm_KL(int l);
  virtual void  normalize_mat_L2();
  virtual void  normalize_mat_L1();
  virtual void  ith_add_CV(int i, float *CV);
  virtual void  CV_sub_ith(int i, float *CV);
  virtual void  CV_sub_ith_prior(int i, float *CV);
  virtual void  ith_add_CV_prior(int i, float *CV);
  virtual float MutualInfo();
  virtual float exponential_kernel(float *v, int i, float norm_v, float sigma_squared);
  virtual void  exponential_kernel(float *x, float norm_x, float *result, float sigma_squared);
  virtual float i_j_dot_product(int i, int j);
};

#endif // !defined(_SPARSE_MATRIX_H_)



