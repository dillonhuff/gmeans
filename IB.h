/*   IB (Information Bottleneck) header file
     IB.h
     Copyright (c) 2003, Yuqiang Guan
*/

#if !defined(_IB_H_)
#define _IB_H_

#include "Gmeans.h"
#include "mat_vec.h"
#define MAXL    30
#define IB_EPSILON 0
#define NUM_RESTARTS 15


class IB : public Gmeans
{
 public:
  IB(Matrix *p_docs, int cluster[], int ini_num_clusters, int est_start, float Omiga, bool random_seeding, int seed, float epsilon);
  ~IB();
 protected:
  float *prior, *new_prior;// *inner_JS;
  
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
  virtual float K_L_first_variation(Matrix *p_Docs);
  virtual float Split_Clusters(Matrix *p_Docs, int worst_vector,  float s_bar);
  
  //virtual void  random_cluster_ID();
  //virtual void  seeding_file_init(Matrix *p_Docs, char * seeding_file);
  //virtual void  update_quality_change_mat(Matrix *p_Docs, int c_ID);
  
};

#endif // !defined(_IB_H_)
