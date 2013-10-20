/*
definition of dense matrix
*/

#if !defined(_DENSE_MATRIX_DOUBLE_H_)
#define _DENSE_MATRIX_DOUBLE_H_

#include "Matrix.h"

class DenseMatrixDouble : public Matrix
{
 private:

  int m_row, m_col;
  float ** m_val;
  float *norm;
  float *L1_norm;

 public:
  
  DenseMatrixDouble ( int row, int col, float ** val);
  
  /*void dmatvec(int m, int n, float **a, float *x, float *y);
  void dmatvecat(int m, int n, float **a, float *x, float *y);
  void dqrbasis( float **q);
  float dvec_l2normsq( int dim, float *v );
  void dvec_l2normalize( int dim, float *v );*/
  inline int GetNumRow ()
  {
    return m_row;
  };
  inline int GetNumCol ()
    {
      return m_col;
    };
  inline int GetNumNonzeros()
    {
      return m_col*m_row;
    };
  inline float& val(int i, int j) {return m_val[i][j]; }
  virtual void trans_mult(float *x, float *result);
  virtual float dot_mult(float *x, int i); 
  virtual void euc_dis(float *x, float norm_x, float *result);
  virtual float euc_dis(float *x, int i, float norm_x);
  virtual void Kullback_leibler(float* x, float* result);
  virtual float Kullback_leibler(float *x, int i);
  virtual void ComputeNorm_KL();
  virtual void ComputeNorm_2();
  virtual void normalize_mat_L1();
  virtual float getNorm(int i);
  virtual float getL1Norm(int i);
  virtual void ith_scale_add_CV(int i, float *CV);
  virtual void ith_add_CV(int i, float *CV);
  virtual void CV_sub_ith(int i, float *CV);
  virtual void CV_sub_ith_scale(int i, float *CV);
};

#endif // !defined(_DENSE_MATRIX_DOUBLEH_)





