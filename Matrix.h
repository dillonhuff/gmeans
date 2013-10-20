/*	Matrix header file
 *		Matrix.h
 *	Copyright (c) 2003, Yuqiang Guan
 */

#if !defined(_MATRIX_H_)
#define _MATRIX_H_

#include "Constants.h"
/*
#define HUGE_NUMBER                     1.0E20
#define DENSE_MATRIX                    11
#define SPARSE_MATRIX                   12
#define DENSE_MATRIX_TRANS              13
*/
#define Sim_Mat(i,j) (i>=j? Sim_Mat[i][j]:Sim_Mat[j][i])

typedef float *VECTOR_double;

class Matrix
{
  
 protected:
  int n_row, n_col;
  float L1_sum, Norm_sum;
  long memory_used;
  float alpha;
  float   *norm;
  float   *L1_norm, *priors;
  float   **Sim_Mat;
 public:
  int		GetNumRow(); 
  int		GetNumCol();
  float         GetL1Norm(int i);
  float GetNorm(int i);
  float GetL1Sum();
  float GetNormSum();
  long          GetMemoryUsed();
  void          SetAlpha(float a, int laplace);
  float         GetAlpha();
  void SetPrior(int method, char * prior_file, int n_E_Docs, int *e_D_ID);
  float getPrior(int i);
  Matrix(int r, int c);
  virtual ~Matrix() {};
  virtual void trans_mult(float *x, float *result) = 0;
  virtual void squared_trans_mult(float *x, float *result) =0; 
  virtual float dot_mult(float *v, int i) = 0;
  virtual float squared_dot_mult(float *v, int i) = 0;
  virtual void right_dom_SV(int *cluster, int *cluster_size, int n_Clusters, float ** CV, float *cluster_quality, int flag)=0;
  //virtual void A_trans_A(int flag, int * index, int *pointers, float ** A_t_A)=0;
  //virtual void euc_dis(float *x, float *result) = 0;
  virtual void euc_dis(float *x, float norm_x, float *result) = 0;
  //virtual float euc_dis(float *v, int i) = 0;
  virtual float euc_dis(float *v, int i, float norm_v) = 0;
  virtual void  Kullback_leibler(float *x, float *result,int laplace) =0;
  virtual float Kullback_leibler(float *x, int i, int laplace) =0;
  virtual void  Kullback_leibler(float *x, float *result,int laplace, float l1norm_X)=0;
  virtual float Kullback_leibler(float *x, int i, int laplace, float l1norm_X)=0;
  virtual float Jenson_Shannon(float *x, int i, float l1n_x) =0;
  virtual void  Jenson_Shannon(float *x, float *result, float prior_x)=0;
  virtual void ComputeNorm_2() = 0;
  virtual void ComputeNorm_1() = 0 ;
  virtual void ComputeNorm_KL(int l) = 0;
  virtual void normalize_mat_L2() = 0;
  virtual void normalize_mat_L1() = 0;
  virtual void ith_add_CV(int i, float *CV) = 0;
  virtual void CV_sub_ith(int i, float *CV) = 0;
  virtual void  CV_sub_ith_prior(int i, float *CV)=0;
  virtual void  ith_add_CV_prior(int i, float *CV)=0;
  virtual float MutualInfo() =0;
  virtual float exponential_kernel(float *v, int i, float norm_v, float sigma_squared) =0;
  virtual void  exponential_kernel(float *x, float norm_x, float *result, float sigma_squared)=0;
  int  kernel_dot_product(int *pointer, int *index, int j, int n_Clusters, float *kernel_norm);
  //void  polynomial_kernel(int *cluster, int *cluster_size, int n_Clusters, float **result, float c, int d, int flag);
  virtual float i_j_dot_product(int i, int j)=0;
  void  pair_wise_dot_product(float constant, int degree, int alg);
  float phi_x_dot_phi_y(int i, int j, float c, int d);
  float kernel_sum(int *pointer, int *index, int i);
  void  kernel_norm_L2(int *cluster, int *cluster_size, int n_Clusters, float *result, int flag);
  float kernel_euc_dis(int *pointer, int *index, int j, float *kernel_center_sum, int cluster_id);
  int   kernel_euc_dis(int *pointer, int *index,int j, int n_Clusters, float *kernel_center_sum);
  float pair_wise_kernel_euc_dis(int i, int j);
  void  cluster_kernel_sum(int *cluster, int *cluster_size, int n_Clusters, float *result, int flag);
};
#endif // !defined(_MATRIX_H_)
