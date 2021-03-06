/*   Diametric k-means header file
     Diametric_k_means.h
     Copyright (c) 2002, Yuqiang Guan
*/

#if !defined(_DIAMETRIC_K_MEANS_H_)
#define _DIAMETRIC_K_MEANS_H_

#include "Gmeans.h"
#include "mat_vec.h"

class Diametric_k_means : public Gmeans
{
 public:
  Diametric_k_means(Matrix *p_docs, int cluster[], int ini_num_clusters, int est_start, float Omiga, bool random_seeding, int seed, float epsilon);
  ~Diametric_k_means();
 protected:
  float * addition;
  virtual float delta_X ( Matrix *p_Docs, int x, int j);
  virtual float verify_obj_func(Matrix *p_Docs, int n_clus);
  virtual void  well_separated_centroids(Matrix *p_Docs, int i=1);
  virtual void  random_centroids(Matrix *p_Docs);
  virtual void  random_perturb_init(Matrix *p_Docs);	
  virtual int   assign_cluster(Matrix *p_Docs, bool simi_est);
  virtual void  general_k_means(Matrix *p_Docs);
  virtual void  remove_Empty_Clusters();
  virtual int   find_worst_vectors(bool dumped);
  virtual void  initialize_cv(Matrix *p_Docs, char * seeding_file);
  float one_f_v_move(Matrix *p_Docs, one_step track [], int step); 
  virtual float K_L_first_variation(Matrix *p_Docs);
  virtual float Split_Clusters(Matrix *p_Docs, int worst_vector,  float s_bar);
  void  update_centroids(Matrix *p_Docs);
  void Diametric_seeding_file_init(Matrix *p_Docs, char * seeding_file);
  //virtual void  random_cluster_ID();
  //virtual void  seeding_file_init(Matrix *p_Docs, char * seeding_file);
  //virtual void  update_quality_change_mat(Matrix *p_Docs, int c_ID);
  
};

#endif // !defined(_DIAMETRIC_K_MEANS_H_)
