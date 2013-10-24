#include <stdlib.h>

#include "FakeData.h"

/**
 * Populates the matrix so that each column is a
 * data point in one of two separable clusters.
 * The two "centers" of the clusters are reflections
 * of each other across the origin
 */
void FakeData::two_separable_clusters(Matrix *mat) {
  int num_cols = mat->GetNumCol();
  float cluster_1_center[num_cols];
  for (int i = 0; i < num_cols; i++) {
    cluster_1_center[i] = (double) rand() / RAND_MAX;
  }
  normalize_vec(cluster_1_center, num_cols);
  scalar_mult(CENTER_MAGNITUDE, cluster_1_center, num_cols);
  float cluster_2_center[num_cols];
  copy_vec(cluster_2_center, cluster_1_center, num_cols);
  scalar_mult(-1.0, cluster_2_center, num_cols);
  for (int i = 0; i < num_cols/2; i++) {
    mat->ith_add_CV(i, cluster_1_center);
  }
  for (int i = num_cols/2; i < num_cols; i++) {
    mat->ith_add_CV(i, cluster_2_center);
  }
  //Add pseudo-random noise to clusters
  for (int i = 0; i < num_cols; i++) {
    
  }
}

/*
 * Generates a vector of length n with entries
 * uniformly and pseudorandomly distributed in
 * [0,1]
 */
void FakeData::uniform_rand_vec(float *v, int n) {

}
