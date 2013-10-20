/*	Gmeans.h
	Header file for Data structures used in k-means algorithm
	Copyright (c) 2002, Yuqiang Guan
*/

#if !defined(_GMEANS_H_)
#define _GMEANS_H_

#include "SparseMatrix.h"
#include "DenseMatrix.h"
#include "RandomGenerator.h"
#include "Constants.h"
#include <iostream>

using namespace std;

extern long memory_consume;

class Gmeans
{
 public:
  Gmeans(Matrix *p_docs, int cluster[], int ini_num_clusters, int est_start, float Omiga, bool random_seeding, int seed, float epsilon);
  virtual ~Gmeans();
  int getClusterNum() { return n_Clusters;};
  void SetEmptyDocs(int n_E_Docs, int *e_D_ID) 
    {
      n_Empty_Docs =n_E_Docs ;
      empty_Docs_ID = e_D_ID;
    };
  void outputClusterSize() 
    { 
      for (int i=0; i< n_Clusters; i++)
	cout << "cluster " << i<< " : " << ClusterSize[i] << "\n";
    };
  void setInitialClustering(int i)         { init_clustering = i; }
  void setIncemental_k_means(int incremental_k_means_moves) { i_k_means_times = incremental_k_means_moves;}
  void setVerify(bool ver)                 { verify = ver;}
  void SetAlgorithm(int alg)               { Alg = alg; }
  void SetEpsilon(float epsilon)           { Epsilon = epsilon; }
  void setDelta(float delta)               { Delta = delta; }
  void SetPerturb(float perturb)           { Perturb = perturb; }
  void SetInitMethod(int init_method)      { Init_Method = init_method; }
  void setSkipSPKM(bool s_SPKM)            { skip_SPKM = s_SPKM;}
  void SetEncodeScheme(int encode_scheme)  { EncodeScheme = encode_scheme; }
  void SetObjFunc(int obj_func)            { ObjFunc = obj_func; }
  void setLaplacian (int l)                { laplacian = l;     }
  void setEvaluate(bool evaluate_only)     { evaluate = evaluate_only;}
  void setClusterID (int *cid) 
    { 
      for (int i=0; i< n_Docs; i++)
	  Cluster[i] = cid [i];
    }
  void setKernelDegree(int d)              { degree =d;}
  void setKernelConstant(float c)          { constant =c;}
  void F_measure (int *label, int n_Class);
  void purity_Entropy_MutInfo();
  void micro_avg_precision_recall();
  void generate_Confusion_Matrix (int *label, int n_Class);
  float GetResult() { return Result; }
  void wordCluster(char *output_matrix, int n_Clusters);
  void setDumpinfo(bool dump){dumpswitch = dump;};
  void general_means(Matrix *p_Docs, int up_bound, int f_v_times, char * seeding_file);
  void read_cate(char *fname);
  void open_dumpfile(char *fname);

 protected:

  int degree;
  float constant;
  std::ofstream obj_inc;
  int EST_START;
  bool stablized;
  bool skip_SPKM;
  bool verify;
  int laplacian;
  bool evaluate;
  float prior, p_log_p;

  struct one_step
  {
    int doc_ID, from_Cluster, to_Cluster;
  };
  int Alg;			// algorithm: classic or general
  int init_clustering;
  bool dumpswitch;
  int f_v_times, i_k_means_times;
  float k_means_threshold, fv_threshold;
  int n_Empty_Docs;
  int *empty_Docs_ID;
	
  int *Cluster;			// which cluster a document belongs to
  int *WordCluster;
  int *label;
  int n_Class;
  VECTOR_double *Concept_Vector;	// concept vectors
	
  VECTOR_double *old_CV, *new_Concept_Vector;
  float *diff, *new_diff;
  int *ClusterSize, *new_ClusterSize, *CategorySize;
  float *cluster_quality, *new_cluster_quality;
  float *CV_norm, *new_CV_norm;
  
  float s_bar;

  RandomGenerator_MT19937 rand_gen;
  int n_Words, n_Docs;	// numbers of words and documents of document matrix
  int Init_Method;		// concept vector initialization method
  int n_Clusters;			// number of clusters, equal to number of concept vectors
  int empty_Clusters;            //
  float Omiga;                   //
	
  float Epsilon;			// stopping criteria of k-means algorithm
  float Perturb;			// maximum norm of the perturb vectors
  float Delta;
  int EncodeScheme;
  int ObjFunc;
  int *mark;

  float **Sim_Mat, **new_Sim_Mat;		// similarity matrix
  float **quality_change_mat, **new_quality_change_mat; // matrix recording quality change for each vec and cluster
  int SPKM_works_time;
  float Result;			// final objective function value
  float pre_Result;
  float initial_obj_fun_val;
  //int ** confusion_matrix;
  
  virtual float delta_X ( Matrix *p_Docs, int x, int j) =0;
  virtual float verify_obj_func(Matrix *p_Docs, int n_clus) =0;
  virtual void  well_separated_centroids(Matrix *p_Docs, int i=1) =0;
  virtual void  random_centroids(Matrix *p_Docs) =0;
  virtual void  random_perturb_init(Matrix *p_Docs) =0;	
  virtual int   assign_cluster(Matrix *p_Docs, bool simi_est) =0;
  virtual void  general_k_means(Matrix *p_Docs) =0;
  virtual void  remove_Empty_Clusters() =0;
  virtual int   find_worst_vectors(bool dumped) =0;
  virtual void  initialize_cv(Matrix *p_Docs, char * seeding_file) =0;
  //virtual float one_f_v_move(Matrix *p_Docs, one_step track [], int step) =0;
  virtual float K_L_first_variation(Matrix *p_Docs) =0;
  virtual float Split_Clusters(Matrix *p_Docs, int worst_vector,  float s_bar) =0;
  //virtual void  update_centroids(Matrix *p_Docs) =0;

  float coherence(int k) ;	
  void random_cluster_ID() ;
  void seeding_file_init(Matrix *p_Docs, char * seeding_file);
  void compute_cluster_size();
  void clear_mark();
  bool empty_vector(int doc);
  bool already_marked(int i);
  void update_quality_change_mat(Matrix *p_Docs, int c_ID);
  void neg_pos_seperation(Matrix *p_Docs);
};

#endif // !defined(_GMEANS_H_)

