/*   Implementation of Diametric K-means 
     Copyright (c) 2002, Yuqiang Guan
*/

#include <time.h>
#include <assert.h>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "Diametric_k_means.h"

using namespace std;

/* Constructor and Distructor */

Diametric_k_means::Diametric_k_means(Matrix *p_docs, int cluster[], int ini_num_clusters, int est_start, float Omiga, bool random_seeding, int seed, float epsilon) : Gmeans(p_docs, cluster, ini_num_clusters, est_start, Omiga, random_seeding, seed, epsilon)
{
  Sim_Mat = new VECTOR_double[n_Clusters];
  for (int j = 0; j < n_Clusters; j++)
    Sim_Mat[j] = new float[n_Docs]; 
  memory_consume+=n_Clusters*n_Docs*sizeof(float);
  addition = new float [ini_num_clusters];
}

Diametric_k_means::~Diametric_k_means()
{
  delete [] addition;
}


/* Diametric K-means functions */

void Diametric_k_means::general_k_means(Matrix *p_Docs)
{

  int n_Iters, i, j;
  bool no_assignment_change;
 
  if(dumpswitch)
    cout<<endl<<"- Start General K-Means loop. -"<<endl<<endl;
  n_Iters =0;
  no_assignment_change =true;

  do
    {
      pre_Result = Result;
      n_Iters ++;
      if (n_Iters >EST_START) 
	stablized =true;
      if(dumpswitch && stablized)
	cout <<"(Similarity estimation used.)"<<endl;
      
      if ( assign_cluster(p_Docs, stablized) == 0 )
	{
	  if (dumpswitch)
	    cout<<"No points are moved in the last step "<<endl<<"@"<<endl<<endl;
	}
      else
	{
	  compute_cluster_size();
       
	  if( stablized)
	    for (i = 0; i < n_Clusters; i++)
	      for (j = 0; j < n_Words; j++)
		old_CV[i][j] = Concept_Vector[i][j];
	  
	  update_centroids(p_Docs);
	  
	  if(stablized)
	    {
	      for (i = 0; i < n_Clusters; i++)
		{
		  diff[i] = 0.0;
		  addition[i] =0.0;
		  for (j = 0; j < n_Words; j++)
		    {
		      diff[i] += (old_CV[i][j] - Concept_Vector[i][j]) * (old_CV[i][j] - Concept_Vector[i][j]);
		      addition[i] += (old_CV[i][j] + Concept_Vector[i][j]) * (old_CV[i][j] + Concept_Vector[i][j]);
		    }
		  diff[i] = sqrt(diff[i]);
		  addition[i] = sqrt (addition[i]);
		  diff[i] *= addition[i];
		}
	  
	      for ( i=0; i<n_Docs; i++)
		Sim_Mat[Cluster[i]][i] = p_Docs->squared_dot_mult (Concept_Vector[Cluster[i]], i);
	    }
	  else
	    for (i = 0; i < n_Clusters; i++)
	      p_Docs->squared_trans_mult(Concept_Vector[i], Sim_Mat[i]);
	  
	  Result = coherence(n_Clusters); 
	  if(dumpswitch)
	    {
	      find_worst_vectors(false);
	      cout<<"Obj. func. ( + Omiga * n_Clusters) = "<<Result<<endl<<"@"<<endl<<endl;
	    }
	  else
	    cout<<"D";
	}
    }
  while ((Result - pre_Result) > Epsilon*Result);
  cout<<endl; 
  if (dumpswitch)
  {
    cout <<"Diametric K-Means loop stoped with "<<n_Iters<<" iterations."<<endl;
    generate_Confusion_Matrix (label, n_Class); 
  }  

  if (stablized)
    for (i = 0; i < n_Clusters; i++)
      p_Docs->squared_trans_mult(Concept_Vector[i], Sim_Mat[i]);
	
  if ((!no_assignment_change) && (f_v_times >0))
    for (i=0; i<n_Clusters; i++)
      update_quality_change_mat(p_Docs, i);
}

int Diametric_k_means::assign_cluster(Matrix *p_Docs, bool simi_est)
{

  int i,j, k, multi=0, changed=0,  temp_Cluster_ID;
  float temp_sim;

  k=0;

  if(simi_est)
    {
      for (i = 0; i < n_Clusters; i++)
	for (j = 0; j < n_Docs; j++)
	  if (i != Cluster[j])
	    Sim_Mat[i][j] += diff[i];
	 
      for (i = 0; i < n_Docs; i++)
	{
	  while (i<empty_Docs_ID[k])
	    {
	      temp_sim = Sim_Mat[Cluster[i]][i];
	      temp_Cluster_ID = Cluster[i];

	      for (j = 0; j < n_Clusters; j++)
		if (j != Cluster[i])
		  {
		    if (Sim_Mat[j][i] > temp_sim)
		      {
			multi++;
			Sim_Mat[j][i] = p_Docs->squared_dot_mult(Concept_Vector[j], i);
			if (Sim_Mat[j][i] > temp_sim)
			  {
			    temp_sim = Sim_Mat[j][i];
			    temp_Cluster_ID = j;
			  }
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
		    if (Sim_Mat[j][i]>temp_sim )
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


void Diametric_k_means::update_centroids(Matrix *p_Docs)
{
  p_Docs->right_dom_SV(Cluster, ClusterSize, n_Clusters, Concept_Vector, cluster_quality, -1);
}

/* initialization functions */


void Diametric_k_means::Diametric_seeding_file_init(Matrix *p_Docs, char * seeding_file)
{

  std::ifstream gpfile(seeding_file);
  int i, max_class=0;
  char whole_line[256];

  if(gpfile.is_open())
    {
      cout<<"Initial cluster ID file: "<<seeding_file<<endl;
      gpfile>>i;
      gpfile.getline(whole_line, 256, '\n');
      if(i!=n_Docs)
	{
	  cout<<"Wrong number of vectors!!! \nSo using well-separate-centroid initialization ..."<<endl;
	  well_separated_centroids(p_Docs, 1);
	}
      else
	{
	  
	  for (i=0; i<n_Docs; i++)
	    {
	      gpfile>>Cluster[i];
	      Cluster[i] /=2;
	      if (max_class<Cluster[i])
		max_class=Cluster[i];
	      gpfile.getline(whole_line, 256, '\n');
	    }
	  if (max_class != n_Clusters-1)
	    {
	      cout<<"Number of clusters in the seeding file is not correct!!!\nSo using well-separate-centroid initialization ..."<<endl;
	      well_separated_centroids(p_Docs, 1);
	    }
	  else if(dumpswitch)
	    {
	      cout<<"Initial centroids are generated from a seeding file."<<endl;
	    }
	}
      gpfile.close();
    }
  else
    { 
      if (!evaluate)
      {
	cout<<"Can't open "<<seeding_file<<endl<<"So using well-separate-centroid initialization ..."<<endl;
	well_separated_centroids(p_Docs, 1);
      }
      else
	{
	  cout<<"Can't open "<<seeding_file<<" or wrong file format."<<endl;
	  exit (0);
	}
    }
}


void Diametric_k_means::initialize_cv(Matrix *p_Docs, char * seeding_file)
{

  int i;
  
  switch (init_clustering)
    {
    case RANDOM_PERTURB_INIT:
      random_perturb_init(p_Docs);
      break;
    case RANDOM_CLUSTER_ID_INIT:
      random_cluster_ID();
      for (i=0; i<n_Clusters; i++) //make sure Concept_Vector is not zero
	Concept_Vector[i][0] =1.0;
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
      if (evaluate)
	Diametric_seeding_file_init(p_Docs, seeding_file);
      else
	seeding_file_init(p_Docs, seeding_file);
      for (i=0; i<n_Clusters; i++)
	for(int j=0; j<n_Words; j++)
	  Concept_Vector[i][j] =1.0;
      break;
    case ALREADY_ASSIGNED:
      break;
    default:
      well_separated_centroids(p_Docs, 1);
      break;
    }
  // reset concept vectors
  for (i = 0; i < n_Docs; i++)
    {
      if((Cluster[i] <0) || (Cluster[i] > n_Clusters)) //marked cluster[i]<0 points as seeds
	Cluster[i] =0;
    }
  compute_cluster_size();
  p_Docs->right_dom_SV(Cluster, ClusterSize, n_Clusters, Concept_Vector, cluster_quality, -1);

  for (i = 0; i < n_Clusters; i++)
    p_Docs->squared_trans_mult(Concept_Vector[i], Sim_Mat[i]); 

  for (i = 0; i < n_Clusters; i++)
    diff[i] = 0.0;

  // because we need give the coherence here.
  
  Result = coherence(n_Clusters);
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

  if (f_v_times >0)
    {
      quality_change_mat = new float*[n_Clusters];
      for (int j = 0; j < n_Clusters; j++)
	quality_change_mat [j] = new float[n_Docs];
      memory_consume+=(n_Clusters*n_Docs)*sizeof(float);
      for (i = 0; i < n_Clusters; i++)
	update_quality_change_mat (p_Docs, i);
    } 
  memory_consume+=p_Docs->GetMemoryUsed();
}

void Diametric_k_means::well_separated_centroids(Matrix *p_Docs, int choice)
{
  int i, j, k, min_ind, *cv = new int [n_Clusters];
  float min, cos_sum;
  bool *mark = new bool [n_Docs];
  float **vp, cq[1];
  int cs[1];

  for (i=0; i< n_Docs; i++)
    mark[i] = false;
  for (i=0; i< n_Empty_Docs; i++)
    mark[empty_Docs_ID[i]] = true;

  for (i = 0; i < n_Clusters; i++)
    for (j = 0; j < n_Words; j++)
      Concept_Vector[i][j] = 0.0;
     
  switch (choice)
    {
    case 0:
      do{
	cv[0] = rand_gen.GetUniformInt(n_Docs);
      }
      while(mark[cv[0]]);

      if(dumpswitch)
	{
	  cout<<"Cluster centroids are chosen to be well separated from each other."<<endl;
	  cout<<"Start with a random chosen vector"<<endl;
	} 
      break;
    case 1:
    default:
      float *v, min;
      int min_ID=0;
      v = new float[n_Words];
      float temp;
      k=0;
      vp = &(v);
      v[0] =1.0;
      cs[0] = n_Docs;
      for (i=0; i<n_Docs; i++)
	Cluster[i] =0;
      p_Docs->right_dom_SV(Cluster, cs, 1, vp, cq, -1);
      //for (int l=0; l<n_Words; l++)
      //cout<<v[l]<<" ";
      //cout<<endl;
      min =1.0;
      min_ID =0;
      for(i=0; i<n_Docs; i++)
	{
	  while (i<empty_Docs_ID[k])
	    {
	      temp = p_Docs->squared_dot_mult(v, i);
	      if ( temp < min )
		{
		  min = temp;
		  min_ID = i;
		}
	      i++;
	    }
	  k++;
	}
      cv[0] = min_ID;
      delete [] v;
      if(dumpswitch)
	{
	  cout<<"Cluster centroids are chosen to be well separated from each other."<<endl;
	  cout<<"Start with a vector farthest from the centroid of the whole data set"<<endl;
	} 
      break;
    }
 
  p_Docs->ith_add_CV(cv[0], Concept_Vector[0]);
  mark[cv[0]] = true;
  
  p_Docs->squared_trans_mult(Concept_Vector[0], Sim_Mat[0]);
  for (i=1; i<n_Clusters; i++)
    {
      min_ind = 0;
      min = 2.0 * n_Clusters;
      for (j=0; j<n_Docs; j++)
	{
	  if(!mark[j])
	    {
	      cos_sum = 0;
	      for (k=0; k<i; k++)
		cos_sum += Sim_Mat[k][j];
	      if (cos_sum < min)
		{
		  min = cos_sum;
		  min_ind = j;
		}
	    }
	}
      cv[i] = min_ind;
      p_Docs->ith_add_CV(cv[i], Concept_Vector[i]);
      
      p_Docs->squared_trans_mult(Concept_Vector[i], Sim_Mat[i]);
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

void Diametric_k_means::random_centroids(Matrix *p_Docs)
{
  int i, j, *cv = new int [n_Clusters];
  bool *mark= new bool[n_Docs];
  
  for (i=0; i< n_Docs; i++)
    mark[i] = false;

  for (i = 0; i < n_Clusters; i++)
    for (j = 0; j < n_Words; j++)
      Concept_Vector[i][j] = 0.0;
  
  for (i=0; i<n_Clusters; i++)
    {
      do{
	cv[i] = rand_gen.GetUniformInt(n_Docs);
      }
      while (mark[cv[i]]);
      mark[cv[i]] = true;
      p_Docs->ith_add_CV(cv[i], Concept_Vector[i]);
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
    p_Docs->squared_trans_mult(Concept_Vector[i], Sim_Mat[i]);
    
  for(i=0; i<n_Docs; i++)
    Cluster[i] =0;
  assign_cluster(p_Docs, false);
}

void Diametric_k_means::random_perturb_init(Matrix *p_Docs)
{

  VECTOR_double v, temp_v;
  int i, j;
  float temp_norm;
  float **vp, cq[1];
  int cs[1];

  v = new float[n_Words];
  temp_v = new float[n_Words];
  vp = &(v);
  cs[0] = n_Docs;
  for (i=0; i<n_Docs; i++)
    Cluster[i] =0;
  p_Docs->right_dom_SV(Cluster, cs, 1, vp, cq, -1);
  for (i = 0; i < n_Clusters; i++)
    for (j = 0; j < n_Words; j++)
      Concept_Vector[i][j] = 0.0;
  
  for (i = 0; i < n_Clusters; i++)
    {
      for (j = 0; j < n_Words; j++)
	temp_v[j] = rand_gen.GetUniform() - 0.5;
      normalize_vec(temp_v, n_Words);
      temp_norm = Perturb * rand_gen.GetUniform();
      for (j = 0; j < n_Words; j++)
	temp_v[j] *= temp_norm;
      for (j = 0; j < n_Words; j++)
	Concept_Vector[i][j] += fabs(v[j]*(1+ temp_v[j]));
    }
  
  delete [] v;
  delete [] temp_v;

  for (i = 0; i < n_Clusters; i++)
    CV_norm[i] = normalize_vec(Concept_Vector[i], n_Words);
  for (i = 0; i < n_Clusters; i++)
    p_Docs->squared_trans_mult(Concept_Vector[i], Sim_Mat[i]);   
    
  for(i=0; i<n_Docs; i++)
    Cluster[i] =0;
  assign_cluster(p_Docs, false);
  if(dumpswitch)
    {
      cout<<"Generated the centroid of whole data set then perturb around it"<<endl; 
    }
}

/* first variations functions */

float Diametric_k_means::K_L_first_variation(Matrix *p_Docs)
{

  int i, j, max_change_index=-1;
  float *change= new float [f_v_times], *total_change =new float [f_v_times], max_change=0.0, pre_change;
  float *old_CV_norm = new float[n_Clusters];
  one_step *track = new one_step[f_v_times];  
  bool *been_converted = new bool [n_Clusters];
  int *original_Cluster =new int [f_v_times], *old_ClusterSize =new int [n_Clusters];

  for (i = 0; i < n_Clusters; i++)
    for (j = 0; j < n_Words; j++)
      old_CV[i][j] = Concept_Vector[i][j];

  for (i=0; i< n_Clusters; i++)
      old_CV_norm[i] = CV_norm[i];

  for (i=0; i< n_Clusters; i++)
    old_ClusterSize[i] = ClusterSize[i];


  cout<<endl<<"+ first variation +"<<endl;
  pre_change =0.0;
  float temp_result = Result;
  int act_f_v_times = f_v_times;

  for (i=0; i< f_v_times; i++)
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
      ClusterSize[track[i].from_Cluster] --;
      ClusterSize[track[i].to_Cluster] ++;
      p_Docs->right_dom_SV(Cluster, ClusterSize, n_Clusters, Concept_Vector, cluster_quality, track[i].from_Cluster);
      p_Docs->right_dom_SV(Cluster, ClusterSize, n_Clusters, Concept_Vector, cluster_quality, track[i].to_Cluster);
      p_Docs->squared_trans_mult(Concept_Vector[track[i].from_Cluster], Sim_Mat[track[i].from_Cluster]);
      p_Docs->squared_trans_mult(Concept_Vector[track[i].to_Cluster], Sim_Mat[track[i].to_Cluster]);
      
      total_change[i] = pre_change+change[i];
      pre_change = total_change[i];
      if (max_change <total_change[i])
	{
	  max_change = total_change[i];
	  max_change_index = i;
	}
	
      update_quality_change_mat(p_Docs, track[i].from_Cluster);
      update_quality_change_mat(p_Docs, track[i].to_Cluster);
    }
  cout<<endl;
  
  if ( dumpswitch) {
    if (max_change > Delta*Result)
      cout<<"Max change of objective fun. "<<max_change<<" occures at step "<<max_change_index+1<<endl;
    else
      cout<<"No change of objective fun."<<endl;
  }

  for (i=0; i< n_Clusters; i++)
    been_converted[i] = false;
  if ( max_change <= Delta*Result)
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
	  p_Docs->squared_trans_mult(Concept_Vector[i], Sim_Mat[i]);
	}
      for (i=0; i< n_Clusters; i++)
	if (f_v_times>0)
	  update_quality_change_mat(p_Docs, i);
    }
  else
    {      
      if (max_change_index < act_f_v_times-1)
	{
	  for (i=0; i<=max_change_index ;i++)
	    {
	      Cluster[track[i].doc_ID] = track[i].to_Cluster;

	      been_converted[track[i].from_Cluster] = true;
	      been_converted[track[i].to_Cluster] = true;
	      old_ClusterSize[track[i].from_Cluster] --;
	      old_ClusterSize[track[i].to_Cluster] ++;
	    }
	  
	  for (i=max_change_index+1; i<act_f_v_times; i++)
	    Cluster[track[i].doc_ID] = original_Cluster[i] ;
	  for (i=0; i< n_Clusters; i++)
	    ClusterSize[i] = old_ClusterSize[i];
	  

	  for (i=0; i< n_Clusters; i++)
	    {
	      if (been_converted[i] == true)
		p_Docs->right_dom_SV(Cluster, ClusterSize, n_Clusters, Concept_Vector, cluster_quality, i);
	      else
		CV_norm[i] = old_CV_norm[i];
	      
	      for (j = 0; j < n_Words; j++)
		old_CV[i][j] = Concept_Vector[i][j];
	      p_Docs->squared_trans_mult(Concept_Vector[i], Sim_Mat[i]);
	      diff[i]= 0.0;
	    }
	  for (i=0; i<n_Clusters; i++)
	    update_quality_change_mat(p_Docs, i);
	}
      else // max_change_index == act_f_v_times
	{
	  for (i=0; i< n_Clusters; i++)
	    {
	      for (j = 0; j < n_Words; j++)
		old_CV[i][j] = Concept_Vector[i][j];
	      diff[0] =0;
	    }
	}
      
      if (dumpswitch)
	generate_Confusion_Matrix (label, n_Class);
    }
  
  //cout<<"!!"<<coherence(p_Docs, n_Clusters)<<endl;
  
  delete [] change;
  delete [] total_change;
  delete [] old_CV_norm;
  delete [] been_converted;
  delete [] track;
  delete [] original_Cluster;
  
  return max_change;
  
  
}

float Diametric_k_means::one_f_v_move(Matrix *p_Docs, one_step track [], int step)
{

  int i, j, max_change_id=-1, where_to_go=-1;
  float max_diff=-1.0e10, temp_diff;
  int k;

  k=0;
  max_diff=-1.0*HUGE_NUMBER;
  for (i = 0; i < n_Docs; i++)
    {
      while (i<empty_Docs_ID[k])
	{
	  if (!already_marked(i))
	    for (j = 0; j < n_Clusters; j++)
	      {
		if (j != Cluster[i])
		  {
		    //temp_diff = delta_X ( p_Docs, i, j);
		    temp_diff = quality_change_mat[j][i];
		    if (temp_diff > max_diff)
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
    
  track[step].doc_ID = max_change_id;
  track[step].from_Cluster = Cluster[max_change_id];
  track[step].to_Cluster = where_to_go;
  
  return max_diff;
}

float Diametric_k_means::delta_X ( Matrix *p_Docs, int x, int c_ID)
{ 
  float quality_change =0.0;
  int temp_ID;
  float temp_cluster_quality[2];

  if (Cluster[x] == c_ID)
    return 0;
  
    
  temp_cluster_quality[0] = cluster_quality[Cluster[x]];
  temp_cluster_quality[1] = cluster_quality[c_ID];
  temp_ID = Cluster[x];
  Cluster[x] = c_ID;
  ClusterSize[temp_ID] --;
  ClusterSize[c_ID] ++;
  p_Docs->right_dom_SV(Cluster, ClusterSize, n_Clusters, Concept_Vector, cluster_quality, temp_ID+n_Clusters);
  p_Docs->right_dom_SV(Cluster, ClusterSize, n_Clusters, Concept_Vector, cluster_quality, c_ID+n_Clusters);
  quality_change=cluster_quality[temp_ID]+cluster_quality[c_ID]- temp_cluster_quality[0]-temp_cluster_quality[1];
  cluster_quality[temp_ID] = temp_cluster_quality[0];
  cluster_quality[c_ID] = temp_cluster_quality[1];
  Cluster[x] = temp_ID;
  ClusterSize[temp_ID] ++;
  ClusterSize[c_ID] --;
  return quality_change;
}

/* split functions */


float Diametric_k_means::Split_Clusters(Matrix *p_Docs, int worst_vector,  float threshold)
{

  int worst_vector_s_cluster_ID, i, j;
  //float * new_CV_norm, *new_cluster_quality,** new_Concept_Vector;

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

  // reform the cluster controids.

  Cluster[worst_vector] = n_Clusters;
  n_Clusters ++;
  ClusterSize[worst_vector_s_cluster_ID] --;

  new_Concept_Vector = new float*[n_Clusters];
  for (i = 0; i <n_Clusters; i++)
    new_Concept_Vector[i] = new float [n_Words];
  new_CV_norm = new float [n_Clusters];
  new_cluster_quality = new_CV_norm;
  
  for (i = 0; i < n_Clusters-1; i++)
  {
    for (j = 0; j < n_Words; j++)
      new_Concept_Vector[i][j] = Concept_Vector[i][j];
    new_CV_norm[i] = CV_norm[i];
  }
  
  for (j = 0; j < n_Words; j++)
    new_Concept_Vector[n_Clusters-1][j]=0.0;
  p_Docs->ith_add_CV(worst_vector, new_Concept_Vector[n_Clusters-1]);
     
  new_cluster_quality[n_Clusters-1] = 1.0;
  p_Docs->right_dom_SV(Cluster, ClusterSize, n_Clusters, new_Concept_Vector, new_cluster_quality, worst_vector_s_cluster_ID);
  
  float change;
  change = new_cluster_quality[worst_vector_s_cluster_ID] - cluster_quality[worst_vector_s_cluster_ID]+new_cluster_quality[n_Clusters-1]+Omiga;
  
 
  if(change > threshold)
    {
      //float *new_diff, **new_Sim_Mat;
      
      for (i=0; i<n_Clusters-1; i++)
	  delete [] old_CV[i];
      delete [] old_CV;
  
      old_CV = new VECTOR_double[n_Clusters];
      for (i = 0; i < n_Clusters; i++)
	old_CV[i] = new float[n_Words];
      
      for (i = 0; i < n_Clusters-1; i++)
	delete[] Concept_Vector[i];
      delete [] Concept_Vector;
      Concept_Vector=new_Concept_Vector;
      
      new_diff =new float [n_Clusters];
      for (i=0; i<n_Clusters-1; i++)
	new_diff[i]= diff[i];
      new_diff[n_Clusters-1]=0.0;
      delete [] diff;
      diff = new_diff;
      
      new_Sim_Mat = new float*[n_Clusters];
      for ( j = 0; j < n_Clusters; j++)
	new_Sim_Mat[j] = new float[n_Docs]; 
      for (i=0; i<n_Clusters-1; i++)
	for (j=0; j<n_Docs; j++)
	  new_Sim_Mat[i][j] = Sim_Mat[i][j] ;

      p_Docs->squared_trans_mult(Concept_Vector[n_Clusters-1], new_Sim_Mat[n_Clusters-1]);
      p_Docs->squared_trans_mult(Concept_Vector[worst_vector_s_cluster_ID], new_Sim_Mat[worst_vector_s_cluster_ID]);
	 
      for (i=0; i<n_Clusters-1; i++)
	delete [] Sim_Mat[i];
      delete [] Sim_Mat;
      Sim_Mat = new_Sim_Mat;
      delete [] cluster_quality;
      CV_norm = cluster_quality = new_cluster_quality;

      new_ClusterSize = new int [n_Clusters];
      for (i=0; i<n_Clusters-1; i++)
	new_ClusterSize[i] =  ClusterSize[i];
      new_ClusterSize[n_Clusters-1]=1;
      delete [] ClusterSize;
      ClusterSize = new_ClusterSize;
      if (dumpswitch)
	generate_Confusion_Matrix (label, n_Class);

      if ( f_v_times > 0)
	{
	  new_quality_change_mat = new float*[n_Clusters];
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
    {
      n_Clusters --;
      Cluster[worst_vector]=worst_vector_s_cluster_ID;
      ClusterSize[worst_vector_s_cluster_ID] ++;
      for (i = 0; i < n_Clusters; i++)
	delete[] new_Concept_Vector[i];
      delete [] new_Concept_Vector;
      delete [] new_cluster_quality;
      cout<<endl<<"This splitting doesn't benefit objective function. No splitting."<<endl;
    }
  return change;
}

/* tool functions */

void Diametric_k_means::remove_Empty_Clusters()
  /* remove empty clusters after general k-means */
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
	      cluster_quality[tmp_label]=cluster_quality[i];
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


float Diametric_k_means::verify_obj_func(Matrix *p_Docs, int n_clus)
{
  int i;
  float value =0.0;

  for (i = 0; i < n_Docs; i++)
    value += p_Docs->squared_dot_mult(Concept_Vector[Cluster[i]], i);
      
  return value+n_clus*Omiga;
}

int Diametric_k_means::find_worst_vectors(bool dumped)
{
  
  int i, k, worst_vector=0;
  float min_sim=1.0, *worst_vec_per_cluster;

  worst_vec_per_cluster = new float[n_Clusters];
  min_sim = 1.0; 
  for (i = 0; i <n_Clusters; i++)
    worst_vec_per_cluster[i] = 1.0;
     
  k=0;
  for (i=0; i<n_Docs; i++)
    {
      while (i<empty_Docs_ID[k])
	{
	  if ( Sim_Mat[Cluster[i]][i] < worst_vec_per_cluster[Cluster[i]])
	    worst_vec_per_cluster[Cluster[i]] =Sim_Mat[Cluster[i]][i];
	  if ( Sim_Mat[Cluster[i]][i] < min_sim)
	    {
	      worst_vector =i;
	      min_sim = Sim_Mat[Cluster[i]][i] ;
	    }
	  i++;
	}
      k++;
    }

  s_bar = 0.0;

  for (i = 0; i <n_Clusters; i++)
    {
      if (ClusterSize[i]==0)
	{
	  if(dumpswitch)
	    cout <<"Cluster "<<i<<" is empty."<<endl;
	}
      else
	{
	  float temp;
	  if(dumpswitch && !dumped)
	    {
	      temp=cluster_quality[i]*cluster_quality[i]-2*cluster_quality[i]*worst_vec_per_cluster[i]+1;
	      temp=sqrt(temp);
	      cout<<"#"<<i<<" : quality/# doc/av_quality/Omega_1/Worst dp = ";
	      cout<<cluster_quality[i]<<"/"<<ClusterSize[i]<<"/"<<cluster_quality[i]/ClusterSize[i];
	      cout<<"/"<<cluster_quality[i]-1.0-temp;
	      cout<<"/"<<worst_vec_per_cluster[i]<<endl;
	      
	    }
	  if(s_bar <cluster_quality[i])
	    s_bar = cluster_quality[i];
	}
    }
 if(dumpswitch && !dumped)
   {
     cout<<"Vector "<<worst_vector<< " has the worst dot product "<<min_sim<<endl;
   }
 delete [] worst_vec_per_cluster;
 
 return worst_vector;
}
