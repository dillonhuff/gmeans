/*   Implementation of Kullback-Leibler K-means 
     Copyright (c) 2002, Yuqiang Guan
*/

#include <time.h>
#include <assert.h>
#include <iostream.h>
#include <stdio.h>
#include <fstream.h>
#include <string.h>
#include <math.h>
#include "Kullback_leibler_k_means.h"

/* Constructor */

Kullback_leibler_k_means::Kullback_leibler_k_means(Matrix *p_docs, int cluster[], int ini_num_clusters, int est_start, float Omiga, bool random_seeding, int seed, float epsilon): Gmeans(p_docs, cluster, ini_num_clusters, est_start, Omiga, random_seeding, seed, epsilon)
{

  Sim_Mat = new VECTOR_double[n_Clusters];
  for (int j = 0; j < n_Clusters; j++)
    Sim_Mat[j] = new float[n_Docs]; 
  memory_consume+=n_Clusters*n_Docs*sizeof(float);
  prior = new float [ini_num_clusters];
  C_log_C = cluster_quality;
  memory_consume+=(ini_num_clusters)*sizeof(float);
  k_means_threshold= p_docs->MutualInfo()*epsilon;
  
}

/* Destructor */
Kullback_leibler_k_means::~Kullback_leibler_k_means()
{
  delete [] prior;
}

/* Kullback-Leibler K-means functions */

void Kullback_leibler_k_means::general_k_means(Matrix *p_Docs)
{

  int n_Iters, i;
  bool no_assignment_change;

  if(dumpswitch)
    cout<<endl<<"- Start Kullback-Leibler K-Means loop. -"<<endl<<endl;
  n_Iters =0;
  no_assignment_change =true;
  stablized = false;
  fv_threshold = -1.0*p_Docs->MutualInfo()*Delta;
  do
    {
      do
	{
	  pre_Result = Result;
	  n_Iters ++;
	  
	  if ( assign_cluster(p_Docs, stablized) == 0 )
	    {
	      if (dumpswitch)
		cout<<"No points are moved in the last step "<<endl<<"@"<<endl<<endl;
	      p_Docs->SetAlpha(p_Docs->GetAlpha()/2.0, laplacian);
	    }
	  else
	    {
	      compute_cluster_size();
	      no_assignment_change = false;
	      p_Docs->SetAlpha(p_Docs->GetAlpha()/2.0, laplacian);
	      update_centroids(p_Docs);
	      
	      if(stablized)
		{
		  
		}
	      else
		{
		  for ( i=0; i< n_Clusters; i++)
		    p_Docs->Kullback_leibler(Concept_Vector[i], Sim_Mat[i], laplacian, prior[i]);
		}
	      
	      Result = p_Docs->GetNormSum() - coherence(n_Clusters)/ log(2.0); 
	      if(dumpswitch)
		{
		  find_worst_vectors(false);
		  cout<<"Obj. func. ( + Omiga * n_Clusters) = "<<Result<<endl<<"@"<<endl<<endl;
		  if (verify)
		    cout<<"Verify obj. func. : "<<verify_obj_func(p_Docs,n_Clusters)<<endl;
		  if (obj_inc.good())
		    obj_inc<<"\t"<<Result/p_Docs->MutualInfo()<<"\t"<<p_Docs->GetAlpha()*2.0<<endl;
		}
	      else
		cout<<"K";
	    }
	}
      while ((pre_Result - Result) > k_means_threshold);
      if ((dumpswitch) && (obj_inc.good()))
	obj_inc<<"%%%%% K-means stops!"<<endl;
    }
  while (p_Docs->GetAlpha() >Epsilon);
  cout<<endl; 
      //p_Docs->SetAlpha(0.0, laplacian);

  
  if (dumpswitch)
  {
    cout <<"Kullback-Leibler K-Means loop stoped with "<<n_Iters<<" iterations."<<endl;
    generate_Confusion_Matrix (label, n_Class); 
    
  }  

  if (stablized)
    {
    }
  
  if ((!no_assignment_change) && (f_v_times >0))
    for (i=0; i<n_Clusters; i++)
      update_quality_change_mat(p_Docs, i);
}

int Kullback_leibler_k_means::assign_cluster(Matrix *p_Docs, bool simi_est)
{

  int i,j, k, multi=0, changed=0,  temp_Cluster_ID;
  float temp_sim;

  k=0;

  if(simi_est)
    {
      
    }
  else
    {
      for (i = 0; i < n_Docs; i++)
	{
	  while (i<empty_Docs_ID[k])
	    {
	      temp_sim = Sim_Mat[Cluster[i]][i];
	      temp_Cluster_ID = Cluster[i];
	      
	      for (j = 0; j < n_Clusters; j++)
		if (j != Cluster[i])
		  {
		    multi++;
		    if (Sim_Mat[j][i] <temp_sim )
		      {
			temp_sim = Sim_Mat[j][i];
			temp_Cluster_ID = j;
		      }
		  }
	      if (temp_Cluster_ID != Cluster[i])
		{
		  Cluster[i] = temp_Cluster_ID;
		  Sim_Mat[Cluster[i]][i] = temp_sim;
		  changed++;
		}   
	      i++;
	    }
	  k++;
	} 
    }
  if(dumpswitch)
    {
      cout << multi << " dot product computation\n";
      cout << changed << " assignment changes\n";
    }
  return changed;
}

void Kullback_leibler_k_means::compute_sum_log(int i)
{
  int j;
  C_log_C[i] =0;
  for (j = 0; j < n_Words; j++)
    if (Concept_Vector[i][j]>0)
      C_log_C[i] +=Concept_Vector[i][j]*log(Concept_Vector[i][j]);
  if (prior[i] >0)
    C_log_C[i] -= prior[i]*log(prior[i]);
  
}

void Kullback_leibler_k_means::update_centroids(Matrix *p_Docs)
{

  int i, j, k;
  
  for (i = 0; i < n_Clusters; i++)
    for (j = 0; j < n_Words; j++)
      Concept_Vector[i][j] = 0.0;
  
  k=0;
  switch (laplacian)
    {
    case NOLAPLACE:
    case CENTER_LAPLACE:
      for (i = 0; i < n_Docs; i++)
	{  
	  while (i<empty_Docs_ID[k])
	    {
	      p_Docs->ith_add_CV_prior(i, Concept_Vector[Cluster[i]]);
	      i++;
	    }
	  k++;
	}
      break;
    case PRIOR_LAPLACE:
      for (i = 0; i < n_Clusters; i++)
	for (j = 0; j < n_Words; j++)
	  Concept_Vector[i][j] = +ClusterSize[i];  
      
      for (i = 0; i < n_Docs; i++)
	{  
	  while (i<empty_Docs_ID[k])
	    {
	      p_Docs->ith_add_CV_prior(i, Concept_Vector[Cluster[i]]);
	      i++;
	    }
	  k++;
	}
      break;
    }

  for (i = 0; i < n_Clusters; i++)
    {
      prior[i] = norm_1(Concept_Vector[i], n_Words);
      compute_sum_log(i);
    }
}


/* initialization functions */

void Kullback_leibler_k_means::initialize_cv(Matrix *p_Docs, char * seeding_file)
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

  switch (laplacian)
    {
    case NOLAPLACE:
    case CENTER_LAPLACE:
      for (i = 0; i < n_Docs; i++)
	if((Cluster[i] >=0) && (Cluster[i] < n_Clusters))//marked cluster[i]<0 points as seeds
	  p_Docs->ith_add_CV_prior(i, Concept_Vector[Cluster[i]]);
      	else
	  Cluster[i] =0;
      break;
    case PRIOR_LAPLACE:
      for (i = 0; i < n_Clusters; i++)
	for (j = 0; j < n_Words; j++)
	  Concept_Vector[i][j] = +ClusterSize[i];  
      
      for (i = 0; i < n_Docs; i++)
	if((Cluster[i] >=0) && (Cluster[i] < n_Clusters))//marked cluster[i]<0 points as seeds
	  p_Docs->ith_add_CV_prior(i, Concept_Vector[Cluster[i]]);
	else
	  Cluster[i] =0;
      break;
    }

  for (i = 0; i < n_Clusters; i++)
    {
      prior[i]= norm_1(Concept_Vector[i], n_Words);  
      compute_sum_log(i);
    }
  
  compute_cluster_size();
  
  
  if (laplacian == PRIOR_LAPLACE)
    for (i = 0; i < n_Clusters; i++)
      for (j = 0; j < n_Words; j++)
	Concept_Vector[i][j] += ClusterSize[i];  

  for (i = 0; i < n_Clusters; i++)
    p_Docs->Kullback_leibler(Concept_Vector[i], Sim_Mat[i], laplacian, prior[i]);
 
  
  //cout<<coherence(n_Clusters)/log(2.0)<<endl; 
  //cout<<p_Docs->GetNormSum()<<endl;
  
  Result = p_Docs->GetNormSum() - coherence(n_Clusters)/log(2.0); 
  
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
      cout<<"The mutual information of the original matrix is : " <<p_Docs->MutualInfo()<<endl;
      cout<<"The mutual information of the final matrix is    : "<<p_Docs->MutualInfo()-Result<<" ("
	  <<(p_Docs->MutualInfo()-Result)*100.0/p_Docs->MutualInfo()<<"%)"<<endl;
      cout<<endl;
      cout <<"* Evaluation done. *"<<endl;
      exit (0);
    }

  if (f_v_times >0)
    {
      quality_change_mat = new (float *)[n_Clusters];
      for (int j = 0; j < n_Clusters; j++)
	quality_change_mat [j] = new float[n_Docs]; 
      memory_consume+=(n_Clusters*n_Docs)*sizeof(float);
      for (i = 0; i < n_Clusters; i++)
	update_quality_change_mat (p_Docs, i);
    } 
  memory_consume+=p_Docs->GetMemoryUsed();
}

void Kullback_leibler_k_means::well_separated_centroids(Matrix *p_Docs, int choice=1)
{
int i, j, k, max_ind, *cv = new int [n_Clusters];
float max, cos_sum;
  bool *mark = new bool [n_Docs];

  for (i=0; i< n_Docs; i++)
    mark[i] = false;
  for (i=0; i< n_Empty_Docs; i++)
    mark[empty_Docs_ID[i]] = true;

  switch (laplacian)
    {
    case NOLAPLACE:
    case CENTER_LAPLACE:
      for (i = 0; i < n_Clusters; i++)
	for (j = 0; j < n_Words; j++)
	  Concept_Vector[i][j] = 0.0;
     break;
    case PRIOR_LAPLACE:
      for (i = 0; i < n_Clusters; i++)
	for (j = 0; j < n_Words; j++)
	  Concept_Vector[i][j] = 1.0;
      break;
    }
  switch (choice)
    {
    case 0:
      do{
	cv[0] = rand_gen.GetUniformInt(n_Docs);
      }while(mark[cv[0]]);
      if(dumpswitch)
	{
	  cout<<"Cluster centroids are chosen to be well separated from each other."<<endl;
	  cout<<"Start with a random chosen vector"<<endl;
	} 
      break;
    case 1:
    default:
      float *v;
      v = new float[n_Words];
      switch (laplacian)
	{
	case NOLAPLACE:
	case CENTER_LAPLACE:
	  for (i = 0; i < n_Words; i++)
	    v[i] = 0.0;
	  break;
	case PRIOR_LAPLACE:
	  for (i = 0; i < n_Words; i++)
	    v[i] = n_Docs;
	  break;
	}
      for (i = 0; i < n_Docs; i++)
	p_Docs->ith_add_CV_prior(i, v);
      
      float temp, temp_norm;
      max_ind=0;
      max =0.0;
      temp_norm = norm_1(v, n_Words);
      k=0;
      for(i=0; i<n_Docs; i++)
	{
	  while (i<empty_Docs_ID[k])
	    {
	      temp = p_Docs->Kullback_leibler(v, i, laplacian, temp_norm);
	      if ( temp > max )
		{
		  max = temp;
		  max_ind = i;
		}
	      i++;
	    }
	  k++;
	}
      cv[0] = max_ind;
      delete [] v;
      if(dumpswitch)
	{
	  cout<<"Cluster centroids are chosen to be well separated from each other."<<endl;
	  cout<<"Start with a vector farthest from the centroid of the whole data set"<<endl;
	} 
      break;
    }
 
  p_Docs->ith_add_CV_prior(cv[0], Concept_Vector[0]);
  prior[0] = norm_1(Concept_Vector[0], n_Words);
  compute_sum_log(0);
  
  mark[cv[0]] = true;
  
  p_Docs->Kullback_leibler(Concept_Vector[0], Sim_Mat[0], laplacian, prior[0]);
  for (i=1; i<n_Clusters; i++)
    {
      max_ind = 0;
      max = 0;
      for (j=0; j<n_Docs; j++)
	{
	  if(!mark[j])
	    {
	      cos_sum = 0;
	      for (k=0; k<i; k++)
		cos_sum += Sim_Mat[k][j];
	      if (cos_sum > max)
		{
		  max = cos_sum;
		  max_ind = j;
		}
	    }
	}
      cv[i] = max_ind;
      p_Docs->ith_add_CV_prior(cv[i], Concept_Vector[i]);
      prior[i] = norm_1(Concept_Vector[i], n_Words); 
      compute_sum_log(i);
      p_Docs->Kullback_leibler(Concept_Vector[i], Sim_Mat[i], laplacian, prior[i]);
      mark[cv[i]] = true;
    }
  for(i=0; i<n_Docs; i++)
    Cluster[i] =0;
  assign_cluster(p_Docs, false);
  
  if(dumpswitch)
    {
      cout<<"Vectors chosen to be the centroids are : ";
      for (i=0; i<n_Clusters; i++)
	cout<<cv[i]<<" ";
      cout<<endl;
    }
  delete [] cv;
  delete [] mark;
}


void Kullback_leibler_k_means::random_centroids(Matrix *p_Docs)
{
  int i, j, k, *cv = new int [n_Clusters];
  bool *mark= new bool[n_Docs];
  k=0;
  for (i = 0; i < n_Docs; i++)
    {
      while (i<empty_Docs_ID[k])
	{
	  mark[i] = false;
	  i++;
	}
      mark[i]=true;
      k++;
    }
  
  switch (laplacian)
    {
    case CENTER_LAPLACE:
    case NOLAPLACE:
      for (i = 0; i < n_Clusters; i++)
	for (j = 0; j < n_Words; j++)
	  Concept_Vector[i][j] = 0.0;  
      break;
    case PRIOR_LAPLACE:
      for (i = 0; i < n_Clusters; i++)
	for (j = 0; j < n_Words; j++)
	  Concept_Vector[i][j] = 1.0;  
      break;
    }
  
  for (i=0; i<n_Clusters; i++)
    {
      do{
	cv[i] = rand_gen.GetUniformInt(n_Docs);
      }
      while (mark[cv[i]]);
      mark[cv[i]] = true;
      p_Docs->ith_add_CV_prior(cv[i], Concept_Vector[i]);
      prior[i] = norm_1(Concept_Vector[i], n_Words); 
      compute_sum_log(i);
    }
  
  if(dumpswitch)
    {
      cout<<"Randomly chose centroids"<<endl;
      cout<<"Vectors chosen to be the centroids are : ";
      for (i=0; i<n_Clusters; i++)
	cout<<cv[i]<<" ";
      cout<<endl;
    }
  delete [] mark;
  delete [] cv;
  
  for (i = 0; i < n_Clusters; i++)
    p_Docs->Kullback_leibler(Concept_Vector[i], Sim_Mat[i], laplacian, prior[i]);
     
  for(i=0; i<n_Docs; i++)
    Cluster[i] =0;
  assign_cluster(p_Docs, false);
}

void Kullback_leibler_k_means::random_perturb_init(Matrix *p_Docs)
{

  VECTOR_double v, temp_v;
  int i, j;
  float temp_norm;
  
  v = new float[n_Words];
  temp_v = new float[n_Words];
  
  switch (laplacian)
    {
    case NOLAPLACE:
    case CENTER_LAPLACE:
      for (i = 0; i < n_Words; i++)
	v[i] = 0.0;
      break;
    case PRIOR_LAPLACE:
      for (i = 0; i < n_Words; i++)
	v[i] = n_Docs;
      break;
    }
  for (i = 0; i < n_Docs; i++)
    p_Docs->ith_add_CV_prior(i, v);
  normalize_vec_1(v, n_Words);
  
  for (i = 0; i < n_Clusters; i++)
    for (j = 0; j < n_Words; j++)
      Concept_Vector[i][j] = 0.0;  
     
  for (i = 0; i < n_Clusters; i++)
    {
      for (j = 0; j < n_Words; j++)
	temp_v[j] = rand_gen.GetUniform() - 0.5;
      normalize_vec_1(temp_v, n_Words);
      temp_norm = Perturb * rand_gen.GetUniform();
      for (j = 0; j < n_Words; j++)
	temp_v[j] *= temp_norm;
      for (j = 0; j < n_Words; j++)
	Concept_Vector[i][j] += fabs(v[j]*(1+ temp_v[j]));
      normalize_vec_1(Concept_Vector[i], n_Words); 
      prior[i]=1.0/n_Clusters;
      compute_sum_log(i);
    }
  
  delete [] v;
  delete [] temp_v;

  for (i = 0; i < n_Clusters; i++)
    p_Docs->Kullback_leibler(Concept_Vector[i], Sim_Mat[i], laplacian, prior[i]);
      
  for(i=0; i<n_Docs; i++)
    Cluster[i] =0;
  assign_cluster(p_Docs, false);
  if(dumpswitch)
    {
      cout<<"Generated the centroid of whole data set then perturb around it"<<endl; 
    }
}

/* first variations functions */

float Kullback_leibler_k_means::K_L_first_variation(Matrix *p_Docs)
{

  int i, j, max_change_index=-1;
  float *change= new float [f_v_times+i_k_means_times], *total_change =new float [f_v_times+i_k_means_times], max_change=0.0, pre_change;
  float *old_prior = new float[n_Clusters], *old_C_log_C = new float[n_Clusters];
  one_step *track = new (one_step) [f_v_times+i_k_means_times]; 
  bool *been_converted= new bool [n_Clusters];
  int *original_Cluster =new int [f_v_times+i_k_means_times], *old_ClusterSize =new int [n_Clusters];


  for (i = 0; i < n_Clusters; i++)
    for (j = 0; j < n_Words; j++)
      old_CV[i][j] = Concept_Vector[i][j];
  for (i=0; i< n_Clusters; i++){
    old_prior[i] = prior[i];
    old_C_log_C[i] = C_log_C[i];
    old_ClusterSize[i] = ClusterSize[i];
  }

  cout<<endl<<"+ first variation +"<<endl;
  pre_change =0.0;
  float temp_result = Result;
  int act_f_v_times = f_v_times+i_k_means_times;
  clear_mark ();
  for (i=0; i< f_v_times || i<i_k_means_times; i++)
    {
      
      change[i] = one_f_v_move(p_Docs, track, i);
      if ( track[i].doc_ID <0)
	{
	  cout <<"This current move occurs at a vector that's already chosen to move. So stop first variation."<<endl;
	  act_f_v_times = i;
	  break;
	}
      mark[i] = track[i].doc_ID;
     
      if ( dumpswitch)
	{
	  cout<<"Vector "<<track[i].doc_ID<<" was moved from C"<<track[i].from_Cluster<<" to C"<<track[i].to_Cluster;
	  if(n_Class>0)
	    cout<<endl<<"And it is in category "<<label[track[i].doc_ID]<<endl;
	  else
	    cout<< endl;
	  cout<<"Change of objective fun. of step "<<i+1<<" is : "<<change[i]<<endl;
	  temp_result+=change[i];
	  
	}
      else
	cout<<"F";
      original_Cluster[i] = Cluster[track[i].doc_ID];
      Cluster[track[i].doc_ID] = track[i].to_Cluster;

      switch (laplacian)
	{
	case NOLAPLACE:
	case CENTER_LAPLACE:
	  p_Docs->CV_sub_ith_prior(track[i].doc_ID, Concept_Vector[track[i].from_Cluster]);
	  p_Docs->ith_add_CV_prior(track[i].doc_ID, Concept_Vector[track[i].to_Cluster]);
	  break;
	case PRIOR_LAPLACE:
	 for(j=0; j<n_Words; j++)
	   Concept_Vector[track[i].from_Cluster][j] -= 1.0;
	  p_Docs->CV_sub_ith(track[i].doc_ID, Concept_Vector[track[i].from_Cluster]);
	  for(j=0; j<n_Words; j++)
	    Concept_Vector[track[i].to_Cluster][j] += 1.0;
	  p_Docs->ith_add_CV(track[i].doc_ID, Concept_Vector[track[i].to_Cluster]);
	  break;
	}
      prior[track[i].from_Cluster] = norm_1(Concept_Vector[track[i].from_Cluster], n_Words);
      compute_sum_log(track[i].from_Cluster);
      prior[track[i].to_Cluster] = norm_1(Concept_Vector[track[i].to_Cluster], n_Words);
      compute_sum_log(track[i].to_Cluster);
      p_Docs->Kullback_leibler(Concept_Vector[track[i].from_Cluster], Sim_Mat[track[i].from_Cluster], laplacian, prior[track[i].from_Cluster]);
      p_Docs->Kullback_leibler(Concept_Vector[track[i].to_Cluster], Sim_Mat[track[i].to_Cluster], laplacian,prior[track[i].to_Cluster]);
     
      // notice cluster_quality[] is not updated because we don't need them as long as we have current centers 
      total_change[i] = pre_change+change[i];
      pre_change = total_change[i];
      if (max_change > total_change[i])
	{
	  max_change = total_change[i];
	  max_change_index = i;
	}
      
      ClusterSize[track[i].from_Cluster] --;
      ClusterSize[track[i].to_Cluster] ++;
      if (f_v_times>0)
	{
	  update_quality_change_mat(p_Docs, track[i].from_Cluster);
	  update_quality_change_mat(p_Docs, track[i].to_Cluster);
	}
    }
  cout<<endl;
  
  if ( dumpswitch)
    if (max_change < fv_threshold)
      cout<<"Max change of objective fun. "<<max_change<<" occures at step "<<max_change_index+1<<endl;
    else
      cout<<"No change of objective fun."<<endl;
  
  for (i=0; i<n_Clusters; i++)
    been_converted[i] =false;
  if ( max_change >= fv_threshold)
    {
      max_change_index = -1;
      max_change =0.0;
     
      for (i=max_change_index+1; i<act_f_v_times; i++)
	Cluster[track[i].doc_ID] = original_Cluster[i] ;
      for (i=0; i< n_Clusters; i++)
	{
	  ClusterSize[i] = old_ClusterSize[i];
	  for (j = 0; j < n_Words; j++)
	    Concept_Vector[i][j] = old_CV[i][j] ;
	  prior[i] = old_prior[i];
	  C_log_C[i] = old_C_log_C[i];
	  p_Docs->Kullback_leibler(Concept_Vector[i], Sim_Mat[i], laplacian,prior[i]);
	  if (f_v_times>0)
	    update_quality_change_mat(p_Docs, i);
	}
    }
  else
    {      
      if (max_change_index < act_f_v_times-1)
	{
	  for (i=0; i<=max_change_index ;i++)
	    {
	      Cluster[track[i].doc_ID] = track[i].to_Cluster;
	      
	      switch (laplacian)
		{
		case NOLAPLACE:
		case CENTER_LAPLACE:
		  p_Docs->ith_add_CV_prior(track[i].doc_ID, old_CV[track[i].to_Cluster]);
		  p_Docs->CV_sub_ith_prior(track[i].doc_ID, old_CV[track[i].from_Cluster]);
		  break;
		case PRIOR_LAPLACE:
		  for(j=0; j<n_Words; j++)
		    old_CV[track[i].to_Cluster][j] += 1.0;
		  p_Docs->ith_add_CV(track[i].doc_ID, old_CV[track[i].to_Cluster]);
		  for(j=0; j<n_Words; j++)
		    old_CV[track[i].from_Cluster][j] -= 1.0;
		  p_Docs->CV_sub_ith(track[i].doc_ID, old_CV[track[i].from_Cluster]);
		  break;
		}
	      
	      been_converted[track[i].from_Cluster]=true;
	      been_converted[track[i].to_Cluster]=true;
	      old_ClusterSize[track[i].from_Cluster] --;
	      old_ClusterSize[track[i].to_Cluster] ++;
	    }

	  for (i=max_change_index+1; i<act_f_v_times; i++)
	    Cluster[track[i].doc_ID] = original_Cluster[i] ;
	  
	  for (i=0; i< n_Clusters; i++)
	    {
	      ClusterSize[i] = old_ClusterSize[i];
	      for (j = 0; j < n_Words; j++)
		Concept_Vector[i][j] = old_CV[i][j] ;
	      if (been_converted[i] == true)
		{
		  prior[i] = norm_1(Concept_Vector[i], n_Words);
		  compute_sum_log(i);
		}
	      else
		{
		  prior[i] = old_prior[i];
		  C_log_C[i] = old_C_log_C[i];
		}    
	      p_Docs->Kullback_leibler(Concept_Vector[i], Sim_Mat[i], laplacian, prior[i]);
	    }
	  if (f_v_times>0)
	    for (i=0; i<n_Clusters; i++)
	      update_quality_change_mat(p_Docs, i);
	}
      
      if (dumpswitch)
	generate_Confusion_Matrix (label, n_Class);
    }
  
  //cout<<"!!"<<coherence(p_Docs, n_Clusters)<<endl;

  
  delete [] old_prior;
  delete [] old_C_log_C;
  delete [] change;
  delete [] total_change;
  delete [] been_converted;
  delete [] track;
  delete [] original_Cluster;
  delete [] old_ClusterSize;
  
  return max_change;
  
}

float Kullback_leibler_k_means::one_f_v_move(Matrix *p_Docs, one_step track [], int step)
{

  int i, j, max_change_id=-1, where_to_go=-1;
  float max_diff, temp_diff;
  int k;
  /*
  for(i=0;i<n_Clusters;i++)
    {
      for (j=0;j<n_Docs; j++)
	cout<<quality_change_mat[i][j]<<" ";
      cout<<endl;
    }
  
  for (i=0; i<n_Docs; i++)
    cout<<p_Docs->getPrior(i)<<endl;
  for (i=0; i<n_Docs; i++)
    cout<<p_Docs->GetNorm(i)<<endl;
  */
  k=0;
  max_diff= Result;
  if(f_v_times>0)
    {
      for (i = 0; i < n_Docs; i++)
	{
	  while (i<empty_Docs_ID[k])
	    {
	      if (!already_marked(i))
		//a document can only be moved in one first-variation in the whole clustering
		for (j = 0; j < n_Clusters; j++)
		  {
		    if (j != Cluster[i])
		      {
		    //temp_diff = delta_X ( p_Docs, i, j);
			temp_diff = quality_change_mat[j][i];
			if (temp_diff < max_diff)
			  {
			    max_change_id= i;
			    where_to_go = j;
			    max_diff = temp_diff;
			  }
		      }
		  }
	      i++;
	    }
	  k++;
	}
    }
  else //if(i_k_means_times >0)
    {
      do
	i= rand_gen.GetUniformInt(n_Docs);
      while(already_marked(i) || empty_vector(i));
      max_change_id=i;
      for (j = 0; j < n_Clusters; j++)
	if (j != Cluster[i])
	  {
	    temp_diff = delta_X ( p_Docs, i, j);
	    if (temp_diff < max_diff)
	      {
		where_to_go = j;
		max_diff = temp_diff;
	      }
	  }
    }
  track[step].doc_ID = max_change_id;
  track[step].from_Cluster = Cluster[max_change_id];
  track[step].to_Cluster = where_to_go;
  
  return max_diff;
}

float Kullback_leibler_k_means::delta_X ( Matrix *p_Docs, int x, int c_ID)
  //if Cluster[x] <0, we compute quality change of cluster c_ID due to adding x into it;
  //if c_ID <0, we compute quality change of cluster Cluster[x] due to deleting x from it.

{ 
  float quality_change=0.0, *temp, temp_norm, temp_sum_log;

  if ((Cluster[x] == c_ID) || (ClusterSize[Cluster[x]] ==1))
    return 0;
  temp = new float [n_Words];
  
  if (Cluster[x]>=0){
    for (int j=0; j<n_Words; j++)
      temp [j] = Concept_Vector[Cluster[x]][j];
    
    switch (laplacian)
      {
      case NOLAPLACE:
      case CENTER_LAPLACE:
	p_Docs->CV_sub_ith_prior(x, temp);
	break;
      case PRIOR_LAPLACE:
	for (int j=0; j<n_Words; j++)
	  temp [j] -= 1.0;
	p_Docs->CV_sub_ith(x, temp);
	break;
      }
    temp_norm = norm_1(temp, n_Words);
    temp_sum_log =0.0;
    for (int j=0; j<n_Words; j++)
      if (temp[j]>0)
	temp_sum_log += temp[j]*log(temp[j]);
    temp_sum_log -= temp_norm * log(temp_norm );
    quality_change =C_log_C[Cluster[x]]-temp_sum_log - p_Docs->GetNorm(x)*p_Docs->getPrior(x) ;

  }

  if (c_ID >=0){
    for (int j=0; j<n_Words; j++)
      temp [j] = Concept_Vector[c_ID][j];
    
    switch (laplacian)
      {
      case NOLAPLACE:
      case CENTER_LAPLACE:
      p_Docs->ith_add_CV_prior(x, temp);
      break;
      case PRIOR_LAPLACE:
	for (int j=0; j<n_Words; j++)
	  temp [j] += 1.0;
	p_Docs->ith_add_CV(x, temp);
	break;
      }
    temp_norm = norm_1(temp, n_Words);
    temp_sum_log =0.0;
    for (int j=0; j<n_Words; j++)
      if (temp[j]>0)
	temp_sum_log += temp[j]*log(temp[j]);
    temp_sum_log -= temp_norm * log(temp_norm );
    quality_change += C_log_C[c_ID]-temp_sum_log +  p_Docs->GetNorm(x)*p_Docs->getPrior(x);
  }
  delete [] temp;
  return quality_change / log(2.0);
}


/* split functions */


float Kullback_leibler_k_means::Split_Clusters(Matrix *p_Docs, int worst_vector,  float threshold)
{

  int worst_vector_s_cluster_ID, i, j;
  float change;

  if(dumpswitch)
    {
      cout <<endl<<"- Split the clusters -"<<endl;
      cout <<"The worst vector "<<worst_vector<<" is in cluster "<<Cluster[worst_vector];
      if(n_Class>0)
	cout<<" and in category "<<label[worst_vector]<<endl;
      else
	cout<< endl;
    }
  worst_vector_s_cluster_ID = Cluster[worst_vector];
  change = delta_X(p_Docs, worst_vector, -1) +Omiga;
  if(change < fv_threshold)
    {
      Cluster[worst_vector] = n_Clusters;
      n_Clusters ++;
      ClusterSize[worst_vector_s_cluster_ID] --;

      for (i=0; i<n_Clusters-1; i++)
	delete [] old_CV[i];
      delete [] old_CV;      
      old_CV = new VECTOR_double[n_Clusters];
      for (i = 0; i < n_Clusters; i++)
	old_CV[i] = new float[n_Words];

      new_ClusterSize = new int [n_Clusters];
      new_cluster_quality = new float [n_Clusters];
      new_prior = new float [n_Clusters];
      new_Concept_Vector = new (float*) [n_Clusters];
      for (i = 0; i <n_Clusters; i++)
	new_Concept_Vector[i] = new float [n_Words];
      
      for (i = 0; i < n_Clusters-1; i++)
	{
	  for (j = 0; j < n_Words; j++)
	    new_Concept_Vector[i][j] = Concept_Vector[i][j];
	  new_cluster_quality[i] = cluster_quality[i];
	  new_prior[i] = prior[i];
	  new_ClusterSize[i] =  ClusterSize[i];
	}
      
      new_ClusterSize[n_Clusters-1]=1;
      new_cluster_quality [n_Clusters-1]=0;
      delete [] cluster_quality;
      C_log_C=cluster_quality = new_cluster_quality;
      delete [] ClusterSize;
      ClusterSize = new_ClusterSize;

      for (i = 0; i < n_Clusters-1; i++)
	delete[] Concept_Vector[i];
      delete [] Concept_Vector;
      Concept_Vector = new_Concept_Vector;
      switch (laplacian)
	{
	case NOLAPLACE:
	case CENTER_LAPLACE:
	  for(j=0; j<n_Words; j++)
	    Concept_Vector[n_Clusters-1][j]=0.0;
	  p_Docs->ith_add_CV_prior(worst_vector, Concept_Vector[n_Clusters-1]);
	  p_Docs->CV_sub_ith_prior(worst_vector, Concept_Vector[worst_vector_s_cluster_ID]);
	  break;
	case PRIOR_LAPLACE:
	  for (j=0; j< n_Words; j++)
	    Concept_Vector[n_Clusters-1][j] =1.0;
	  p_Docs->ith_add_CV(worst_vector, Concept_Vector[n_Clusters-1]);
	  for (j=0; j< n_Words; j++)
	    Concept_Vector[worst_vector_s_cluster_ID][j] -=1.0;
	  p_Docs->CV_sub_ith(worst_vector, Concept_Vector[worst_vector_s_cluster_ID]);
	  break;
	}
      new_prior[n_Clusters-1] = p_Docs->getPrior(worst_vector);
      new_prior[worst_vector_s_cluster_ID] = norm_1(Concept_Vector[worst_vector_s_cluster_ID], n_Words);
      delete [] prior;
      prior = new_prior;
 
      compute_sum_log(n_Clusters-1);
      compute_sum_log(worst_vector_s_cluster_ID);

      new_Sim_Mat = new (float *)[n_Clusters];
      for ( j = 0; j < n_Clusters; j++)
	new_Sim_Mat[j] = new float[n_Docs]; 
      for (i=0; i<n_Clusters-1; i++)
	for (j=0; j<n_Docs; j++)
	  new_Sim_Mat[i][j] = Sim_Mat[i][j] ;
     
      p_Docs->Kullback_leibler(Concept_Vector[n_Clusters-1], new_Sim_Mat[n_Clusters-1], laplacian, prior[n_Clusters-1]);
      p_Docs->Kullback_leibler(Concept_Vector[worst_vector_s_cluster_ID], new_Sim_Mat[worst_vector_s_cluster_ID], laplacian, prior[worst_vector_s_cluster_ID]);
      
      for (i=0; i<n_Clusters-1; i++)
	delete [] Sim_Mat[i];
      
      /* there is some problem in linux about this following deletion */
      delete  [] Sim_Mat;
      Sim_Mat = new_Sim_Mat; 
      
      if (dumpswitch)
	generate_Confusion_Matrix (label, n_Class);
      
      if ( f_v_times > 0)
	{
	  new_quality_change_mat = new (float *)[n_Clusters];
	  for ( j = 0; j < n_Clusters; j++)
	    new_quality_change_mat [j] = new float[n_Docs]; 
	  update_quality_change_mat (p_Docs, worst_vector_s_cluster_ID);
	  for (i=0; i<n_Clusters-1; i++)
	    for (j=0; j<n_Docs; j++)
	      new_quality_change_mat [i][j] = quality_change_mat [i][j];
	  for (i=0; i<n_Clusters-1; i++)
	    delete [] quality_change_mat[i];
	  delete []quality_change_mat;  
	  quality_change_mat = new_quality_change_mat;
	  update_quality_change_mat (p_Docs, n_Clusters-1);
	}
    }
  else
    cout<<endl<<"This splitting doesn't benefit objective function. No splitting."<<endl;
  return change;
}

/* tool functions */

void Kullback_leibler_k_means::remove_Empty_Clusters()
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
      int k=0;
      for (i = 0; i < n_Docs; i++)
	{
	  while (i<empty_Docs_ID[k])
	    {
	      Cluster[i] = cluster_label[Cluster[i]];
	      i++;
	    }
	  k++;
	}

      for (i=0; i< n_Clusters; i++)
	if ((ClusterSize[i] != 0) && (cluster_label[i] != i))
	  {
	    for (j=0; j<n_Words; j++)
	      Concept_Vector[cluster_label[i]][j] = Concept_Vector[i][j];
	    for (j=0; j<n_Docs; j++)
	      Sim_Mat[cluster_label[i]][j] = Sim_Mat[i][j];
	    diff[cluster_label[i]] = diff[i];
	    ClusterSize[cluster_label[i]] = ClusterSize[i];
	    cluster_quality[cluster_label[i]]=cluster_quality[i];
	    prior[cluster_label[i]] = prior[i];
	  }
      for(i= n_Clusters- empty_Clusters ; i< n_Clusters; i++)
	{
	  delete [] Concept_Vector[i];
	  delete [] old_CV[i];
	  delete [] Sim_Mat[i];
	}
      
      n_Clusters -=empty_Clusters;
    }
  delete [] cluster_label;
}


float Kullback_leibler_k_means::verify_obj_func(Matrix *p_Docs, int n_clus)
{
  int i, k;
  float value =0.0;
  k=0;
  for (i = 0; i < n_Docs; i++)
    {
      while (i<empty_Docs_ID[k])
	{
	  value += p_Docs->getPrior(i) * p_Docs->Kullback_leibler(Concept_Vector[Cluster[i]], i, laplacian, prior[Cluster[i]]);
	  i++;
	}
      k++;
    }
       
  return value/log(2.0)+n_clus*Omiga;
}

int Kullback_leibler_k_means::find_worst_vectors(bool dumped)
{
  
  int i, k, worst_vector=0;
  float min_sim, *worst_vec_per_cluster;

  worst_vec_per_cluster = new float[n_Clusters];
  min_sim = 0; 
  for (i = 0; i <n_Clusters; i++)
    worst_vec_per_cluster[i] = 0;
     
  k=0;
  for (i=0; i<n_Docs; i++)
    {
      while (i<empty_Docs_ID[k])
	{
	  if ( Sim_Mat[Cluster[i]][i] > worst_vec_per_cluster[Cluster[i]])
	    worst_vec_per_cluster[Cluster[i]] =Sim_Mat[Cluster[i]][i];
	  if ( Sim_Mat[Cluster[i]][i] > min_sim)
	    {
	      worst_vector =i;
	      min_sim = Sim_Mat[Cluster[i]][i] ;
	    }
	  i++;
	}
      k++;
    }

  for (i = 0; i <n_Clusters; i++)
    {
      if (ClusterSize[i]==0)
	{
	  if(dumpswitch)
	    cout <<"Cluster "<<i<<" is empty."<<endl;
	}
      else
	{
	  if(dumpswitch && !dumped)
	    {
	      cout<<"#"<<i<<" : quality/# doc/av_quality/L1_norm/sum_log/Worst dp = ";
	      cout<<cluster_quality[i]<<"/"<<ClusterSize[i]<<"/"<<cluster_quality[i]/ClusterSize[i];
	      cout<<"/"<<prior[i]<<"/"<<C_log_C[i];
	      cout<<"/"<<worst_vec_per_cluster[i]<<endl;
	      
	    }
	}
    }
  
  if(dumpswitch && !dumped)
    {
      cout<<"Vector "<<worst_vector<< " has the worst Kullback-leibler "<<min_sim<<endl;
    }
  delete [] worst_vec_per_cluster;
  
  return worst_vector;
}



