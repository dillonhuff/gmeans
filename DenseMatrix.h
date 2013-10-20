/*	Dense Matrix header file
 *		DenseMatrix.h
 *	Copyright (c) 2003, Yuqiang Guan
 */

#if !defined(_DENSE_MATRIX_H_)
#define _DENSE_MATRIX_H_

#include "Matrix.h"
#include "mat_vec.h"
#include "Constants.h"

class DenseMatrix : public Matrix
{
 protected:
  int m_row, m_col;
  float ** m_val;
  
  void A_trans_A(int flag, int * index, int *pointers, float ** A_t_A);

 public:
  DenseMatrix ( int row, int col, float ** val);
  ~DenseMatrix();
  inline float& val(int i, int j) {return m_val[i][j]; }
  virtual void  trans_mult(float *x, float *result);
  virtual void  squared_trans_mult(float *x, float *result);
  virtual float dot_mult(float *x, int i); 
  virtual float squared_dot_mult(float *x, int i); 
  virtual void  right_dom_SV(int *cluster, int *cluster_size, int n_Clusters, float ** CV, float *cluster_quality, int flag);
  virtual void  euc_dis(float *x, float norm_x, float *result);
  virtual float euc_dis(float *x, int i, float norm_x);
  virtual void  Kullback_leibler(float *x, float *result,int laplace);
  virtual float Kullback_leibler(float *x, int i, int laplace);
  virtual void  Kullback_leibler(float *x, float *result,int laplace, float l1norm_X);
  virtual float Kullback_leibler(float *x, int i, int laplace, float l1norm_X);
  virtual float Jenson_Shannon(float *x, int i, float l1n_x);
  virtual void  Jenson_Shannon(float *x, float *result, float prior_x);
  virtual void  ComputeNorm_KL(int l);
  virtual void  ComputeNorm_2();
  virtual void  ComputeNorm_1();
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
  
  /*void dmatvec(int m, int n, float **a, float *x, float *y);
  void dmatvecat(int m, int n, float **a, float *x, float *y);
  void dqrbasis( float **q);
  float dvec_l2normsq( int dim, float *v );
  void dvec_l2normalize( int dim, float *v );*/
};

#endif // !defined(_DENSE_MATRIX_H_)





