/*	mat_vec.h
	subroutines to handle vectors
	Copyright(c) 2000 Shulin Ni
*/

#if !defined(_MAT_VEC_H_)
#define _MAT_VEC_H_

#include "Matrix.h"
//#include "RandomGenerator.h"
// document encoding 
#define NORM_TERM_FREQ			1 // normalized term frequency
#define NORM_TERM_FREQ_INV_DOC_FREQ	2 // normalized term frequency-inverse
					  // document frequency
#define Power_Method_Epsilon            0.0001
/*

// Encode documents using the specified encoding scheme.
// The elements of the input matrix contains the number of occurrences of a word
// in a document.
void encode_mat(SparseMatrix *mat, int scheme = NORM_TERM_FREQ);
void encode_mat(DenseMatrix *mat, int scheme = NORM_TERM_FREQ);


void dmatvec(int m, int n, float **a, float *x, float *y);
void dmatvecat(int m, int n, float **a, float *x, float *y);
void dqrbasis( int m, int n, float **a, float **q , float *work);
float dvec_l2normsq( int dim, float *v );
void dvec_l2normalize( int dim, float *v );
void dvec_scale( float alpha, int n, float *v );
*/

void average_vec(float vec[], int n, int num);
float norm_2 (float vec[], int n);
float norm_1(float *x, int n);
float KL_norm (float vec[], int n);
float euclidian_distance(float *v1, float *v2, int n);
float Kullback_leibler(float *x, float *y, int n);
float normalize_vec(float *vec, int n);
float normalize_vec_1(float *vec, int n);
float dot_mult(float *v1, float *v2, int n);
void Ax (float **A, float *x, int dim1, int dim2, float *result);
void Ax(float *vals, int *rowinds, int *colptrs, int dim1, int dim2, float *x, float *result);
float x_dot_y(float *x, float *y, int dim1);
void power_method(float **A_t_A, int dim, float * CV, float *init, float & Lamda);
void power_method(float *vals, int *rowinds, int *colptrs, int n_col, float * CV, float *init, float & Lamda);
float mutual_info(float ** matrix, int r, int c);
float mutual_info(float ** matrix, int r, int c, float*);
#endif // !defined(_MAT_VEC_H_)
