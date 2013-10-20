/*   Implementation of Information Bottleneck 
     Copyright (c) 2003, Yuqiang Guan
*/

#include <time.h>
#include <assert.h>
#include <iostream.h>
#include <stdio.h>
#include <fstream.h>
#include <string.h>
#include <math.h>
#include "IB.h"

/* Constructor */

IB::IB(Matrix *p_docs, int cluster[], int ini_num_clusters, int est_start, float Omiga, bool random_seeding, int seed, float epsilon): Gmeans(p_docs, cluster, ini_num_clusters, est_start, Omiga, random_seeding, seed, epsilon)
{
  //inner_JS = new float [n_Docs];
  prior = new float [ini_num_clusters];
}

/* Destructor */
IB::~IB()
{
  delete [] prior;
  //delete [] inner_JS;
}

/* Kullback-Leibler K-means functions */

void IB::general_k_means(Matrix *p_Docs)
{
  float temp_JS=0.0, min_JS, temp_prior;
  int n_Iters, j, k, min_JS_ind, doc_index, n_moves;
  bool done;
  float *temp_CV =new float [n_Words];
  
  if(dumpswitch)
    cout<<endl<<"- Start Information Bottleneck loop. -"<<endl<<endl;
  
  n_Iters =0;
  
  done = false;
  
  
  while (! done)
    {
      n_moves =0;
      k=0;
      for (doc_index=0; doc_index< n_Docs; doc_index++)
	{
	  while (doc_index<empty_Docs_ID[k])
	    {
	      //float temp_mi = Result;
	      //min_JS = inner_JS[doc_index];
	      for (j=0; j<n_Words; j++)
		temp_CV[j] = Concept_Vector[Cluster[doc_index]][j]*prior[Cluster[doc_index]];
	      p_Docs->CV_sub_ith_prior(doc_index, temp_CV);
	      temp_prior = normalize_vec_1(temp_CV, n_Words); // temp_CV is not normalized
	      min_JS =  prior[Cluster[doc_index]]* p_Docs->Jenson_Shannon(temp_CV, doc_index, temp_prior);
	      min_JS_ind = Cluster[doc_index];
					      
	      for (j=0; j<n_Clusters; j++)
		if ( j!= Cluster[doc_index])
		  {
		    temp_JS = (prior[j]+p_Docs->getPrior(doc_index))*Sim_Mat[j][doc_index];
		    
		    if (min_JS > temp_JS)
		      {
			min_JS = temp_JS;
			min_JS_ind = j;
		      }
		  }
	      if ( min_JS_ind != Cluster[doc_index])
		{
		  for (j=0; j<n_Words; j++)
		    Concept_Vector[min_JS_ind][j] *=prior[min_JS_ind];
		  p_Docs->ith_add_CV_prior(doc_index, Concept_Vector[min_JS_ind]);
		  prior[min_JS_ind] = normalize_vec_1(Concept_Vector[min_JS_ind], n_Words);
		  for (j=0; j<n_Words; j++)
		    Concept_Vector[Cluster[doc_index]][j] = temp_CV[j] * temp_prior;
		  prior[Cluster[doc_index]] = normalize_vec_1(Concept_Vector[Cluster[doc_index]], n_Words);
		  n_moves ++;
		  p_Docs->Jenson_Shannon(Concept_Vector[min_JS_ind], Sim_Mat[min_JS_ind], prior[min_JS_ind]);
		  p_Docs->Jenson_Shannon(Concept_Vector[Cluster[doc_index]], Sim_Mat[Cluster[doc_index]],prior[Cluster[doc_index]]); 
		  Cluster[doc_index] = min_JS_ind;
		  /*
		    Result = mutual_info(Concept_Vector, n_Clusters,n_Words);
		    if (temp_mi > Result)
		    {
		    cout<<"ERROR!!! : "<<temp_mi<<" "<<Result<<endl;
		    //exit (0);
		    }
		    else
		    temp_mi = Result;
		  */
		}
	      
	      doc_index ++;
	    }
	  k++;
	}
      if (dumpswitch)  
	{
	  cout <<"Number of assignment changes : "<<n_moves<<endl;
	  generate_Confusion_Matrix (label, n_Class); 
	  Result = mutual_info(Concept_Vector, n_Clusters,n_Words,  prior);
	  cout<<"\nObj. fun. val = : "<<Result<<endl;
	}
      n_Iters ++;
      if ((n_Iters >= MAXL) || (n_moves <= IB_EPSILON * n_Docs))
	done = true;
    }
  
  Result = mutual_info(Concept_Vector, n_Clusters,n_Words,  prior);
  
  if (dumpswitch)  
    {
      find_worst_vectors(false);
      cout<<"Obj. func. ( + Omiga * n_Clusters) = "<<Result<<endl<<"@"<<endl<<endl;
      if (verify)
	cout<<"Verify obj. func. : "<<verify_obj_func(p_Docs,n_Clusters)<<endl;
    }
  else
    cout<<"IB";
  
  if (dumpswitch)
    {
      cout <<"Information Bottleneck loop stoped with "<<n_Iters<<" iterations."<<endl;
      generate_Confusion_Matrix (label, n_Class); 
    }  
  delete [] temp_CV;
}

int IB::assign_cluster(Matrix *p_Docs, bool simi_est)
{
  return 0;
}

/* initialization functions */

void IB::initialize_cv(Matrix *p_Docs, char * seeding_file)
{

  int i, j;
  
  switch (init_clustering)
    {
    case RANDOM_PERTURB_INIT:
      random_perturb_init(p_Docs);
      break;
    case RANDOM_CLUSTER_ID_INIT:
      random_cluster_ID();
      break;
    case CONCEPT_VECTORS_INIT:
      random_centroids(p_Docs);
      break;
    case WELL_SEPARATED_CENTROID_INIT:
      well_separated_centroids(p_Docs, 0);
      break;
    case WELL_SEPARATED_CENTROID_INIT_MODIFY:
      well_separated_centroids(p_Docs, 1);
      break;
    case SEEDINGFILE_INIT:
      seeding_file_init(p_Docs, seeding_file);
      break;
    case ALREADY_ASSIGNED:
      break;
    default:
      random_cluster_ID();
      break;
    }
  // reset concept vectors
  
  for (i = 0; i < n_Clusters; i++)
    for (j = 0; j < n_Words; j++)
      Concept_Vector[i][j] = 0.0;  
     
  for (i = 0; i < n_Docs; i++)
    {  
      if((Cluster[i] >=0) && (Cluster[i] < n_Clusters))//marked cluster[i]<0 points as seeds
	p_Docs->ith_add_CV_prior(i, Concept_Vector[Cluster[i]]);
      else
	Cluster[i] =0;
    }

  compute_cluster_size();

  for (i = 0; i < n_Clusters; i++)
    {
      prior[i] = normalize_vec_1(Concept_Vector[i], n_Words);
      //Concept_Vector is not normalized
    }
  
  for (i = 0; i < n_Clusters; i++) 
    p_Docs->Jenson_Shannon(Concept_Vector[i], Sim_Mat[i], prior[Cluster[i]]);
  
  Result = mutual_info(Concept_Vector, n_Clusters,n_Words,  prior);

  if(dumpswitch || evaluate)
    {
      outputClusterSize();
      cout<<"Initial Obj. func.: "<<Result<<endl;
      if (n_Class >0)
	cout<<"Initial confusion matrix :"<<endl;
      generate_Confusion_Matrix (label, n_Class);
      
    }

  if(evaluate)
    {
      purity_Entropy_MutInfo();
      F_measure(label, n_Class);
      micro_avg_precision_recall();
      cout<<endl;
      cout <<"* Evaluation done. *"<<endl;
      exit (0);
    }
}

void IB::well_separated_centroids(Matrix *p_Docs, int choice=1)
{
  
}


void IB::random_centroids(Matrix *p_Docs)
{
  
}

void IB::random_perturb_init(Matrix *p_Docs)
{

}

/* first variations functions */

float IB::K_L_first_variation(Matrix *p_Docs)
  /* we are gonna use sum vector instead of concept vector in FV */
{

  return 0;
}


float IB::delta_X ( Matrix *p_Docs, int x, int c_ID)
  //if Cluster[x] <0, we compute quality change of cluster c_ID due to adding x into it;
  //if c_ID <0, we compute quality change of cluster Cluster[x] due to deleting x from it.

{ 
  return 0;
}


/* split functions */


float IB::Split_Clusters(Matrix *p_Docs, int worst_vector,  float threshold)
{
  
  return 0;
}

/* tool functions */

void IB::remove_Empty_Clusters()
  /* remove empty clusters after general k-means and fv because fv gets rid of empty clusters*/
{
  int *cluster_label;
  int i,j, tmp_label;

  cluster_label = new int [n_Clusters];

  tmp_label =0;
  empty_Clusters=0;

  for(i=0; i<n_Clusters; i++)
    {
    if(ClusterSize[i] == 0)
      {
	empty_Clusters ++;
      }
    else
      {
	cluster_label[i]= tmp_label;
	tmp_label++;
      }
    }
  if(empty_Clusters !=0)
    {
      
      cout<<empty_Clusters<<" empty clusters generated."<<endl<<"Cluster IDs have been changed."<<endl;
      for(i=0;i<n_Docs; i++)
	Cluster[i] = cluster_label[Cluster[i]];
      for (i=0; i< n_Clusters; i++)
	  if ((ClusterSize[i] != 0) && (cluster_label[i] != i))
	    {
	      for (j=0; j<n_Words; j++)
		Concept_Vector[cluster_label[i]][j] = Concept_Vector[i][j];
	      ClusterSize[cluster_label[i]] = ClusterSize[i];
	      prior[cluster_label[i]] = prior[i];
	    }
      for(i= n_Clusters- empty_Clusters ; i< n_Clusters; i++)
	{
	  delete [] Concept_Vector[i];
	}

      n_Clusters -=empty_Clusters;
    }
  delete [] cluster_label;
}


float IB::verify_obj_func(Matrix *p_Docs, int n_clus)
{
  return 0;
}

int IB::find_worst_vectors(bool dumped)
{
  
  return 0;
}

