/*	Sparse Matrix header file
 *		SparseMatrix.h
 *	Copyright (c) 2002, Yuqiang Guan
 */

#if !defined(_SPARSE_MATRIX_DOUBLE_H_)
#define _SPARSE_MATRIX_DOUBLE_H_

#include "Matrix.h"


class SparseMatrixDouble : public Matrix
{
private:
	int	n_row, n_col, n_nz;

	float	*vals;
	int	*rowinds;
	int	*colptrs;
	float   *norm;
	float   *L1_norm;

	// data structures for accelerating the computation of transpose multiplication
/*	int	*rowptrs;	// row pointer array
	int	*colinds;
	double  *vals2; */

public:
	SparseMatrixDouble(int row, int col, int nz, float *val, int *rowind, int *colptr);
	//~SparseMatrixDouble();

//	void			prepare_trans_mult();

	inline float&		val(int i) { return vals[i]; }
	inline int&		row_ind(int i) { return rowinds[i]; }
	inline int&		col_ptr(int i) { return colptrs[i]; }

	inline int		GetNumRow() { return n_row; }
	inline int		GetNumCol() { return n_col; }
	inline int		GetNumNonzeros() { return n_nz; }

	float			operator() (int i, int j) const;
	virtual void trans_mult(float *x, float *result) ;	// transpose and multiply by x
	virtual float dot_mult(float *v, int i);
	virtual void euc_dis(float *x, float *result);
	virtual void euc_dis(float *x, float norm_x, float *result);
	virtual float euc_dis(float *v, int i, float norm_v);
	virtual void Kullback_leibler(float *x, float *result);
	virtual float Kullback_leibler(float *x, int i);
	virtual void ComputeNorm_2();
	virtual void ComputeNorm_KL();
	virtual void normalize_mat_L1();
	virtual float getNorm(int i);
	virtual float getL1Norm(int i);
	virtual void ith_scale_add_CV(int i, float *CV);
	virtual void ith_add_CV(int i, float *CV);
	virtual void CV_sub_ith_scale(int i, float *CV);
	virtual void CV_sub_ith(int i, float *CV);
};

#endif // !defined(_SPARSE_MATRIX_DOUBLE_H_)



