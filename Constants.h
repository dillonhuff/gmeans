/*	Constants.h
	Define a couple of constants
	Copyright (c) 2003, Yuqiang Guan
*/

#if !defined(_CONSTANTS_H_)
#define _CONSTANTS_H_

#define EUCLIDEAN_K_MEANS		1
#define SPHERICAL_K_MEANS		2
#define KULLBACK_LEIBLER                3
#define DIAMETRIC_K_MEANS		4
#define INFO_BOTTLENECK                 5
#define EXP_KERNEL                      6
#define KERNEL_SPHERICAL_K_MEANS        7
#define KERNEL_EUCLIDEAN_K_MEANS        8

#define AIC                            21
#define BIC                            22
#define MDL                            22
#define MML                            23

#define NOLAPLACE                      31
#define CENTER_LAPLACE                32
#define PRIOR_LAPLACE                 33

#define RANDOM_PERTURB_INIT		1
#define SAMPLE_INIT			2
#define CONCEPT_VECTORS_INIT		3
#define RANDOM_CLUSTER_ID_INIT		4
#define SEEDINGFILE_INIT                5
#define WELL_SEPARATED_CENTROID_INIT    6
#define WELL_SEPARATED_CENTROID_INIT_MODIFY 7
#define ALREADY_ASSIGNED                8

#define DEFAULT_EPSILON			0.001
#define DEFAULT_DELTA                   0.000001
#define DEFAULT_PERTURB			0.1
#define FLOAT_ERROR                     1.0e-6
#define DEFAULT_ALPHA                   0

#define HUGE_NUMBER                     1.0E20
#define DENSE_MATRIX                    11
#define SPARSE_MATRIX                   12
#define DENSE_MATRIX_TRANS              13

#define FILE_PRIOR                      41
#define UNIFORM_PRIOR                   40

#define DEFAULT_DEGREE                  1
#define DEFAULT_CONSTANT                0


#endif // !defined(_CONSTANTS_H_)
