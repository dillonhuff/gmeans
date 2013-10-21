/*	Gmeans.cc 
	Implementation of the Gmeans class
	Copyright (c) 2002, Yuqiang Guan
*/

#include <time.h>
#include <assert.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <string.h>
#include <math.h>
#include "mat_vec.h"
#include "Gmeans.h"

using namespace std;

Gmeans::Gmeans(Matrix *p_Docs, int cluster[], int ini_num_clusters, int sim_est_start,
		    float omiga,  bool random_seeding, int seed, float epsilon) 
  
{
  assert(p_Docs != NULL && cluster != NULL && ini_num_clusters > 0 &&  epsilon > 0);

	
  Cluster = cluster;
  n_Clusters = ini_num_clusters;
  
  Epsilon = epsilon;
  Omiga = omiga;
  
  EST_START = sim_est_start;
  n_Words = p_Docs->GetNumRow();
  n_Docs = p_Docs->GetNumCol();

  Perturb = DEFAULT_PERTURB;
  /*
  Sim_Mat = new VECTOR_double[n_Clusters];
  for (int j = 0; j < n_Clusters; j++)
    Sim_Mat[j] = new float[n_Docs]; 
  */
  cluster_quality = new float[n_Clusters];
  CV_norm = cluster_quality;
  
  //memory_consume+=(n_Clusters+n_Clusters*n_Docs)*sizeof(float);
  memory_consume+= n_Clusters*sizeof(float);

  Concept_Vector = new VECTOR_double[n_Clusters];
  for (int i = 0; i < n_Clusters; i++)
    Concept_Vector[i] = new float[n_Words];
  
  old_CV = new VECTOR_double[n_Clusters];
  for (int i = 0; i < n_Clusters; i++)
    old_CV[i] = new float[n_Words];
  
  diff = new float[n_Clusters];
  
  memory_consume+=n_Clusters*n_Words*sizeof(float)*2;
  
  ClusterSize = new  int[n_Clusters];
  
  memory_consume+=(n_Clusters*2+n_Docs)*sizeof(float);

  if(!random_seeding)
    rand_gen.Set((unsigned) seed);
  else
    rand_gen.Set((unsigned)time(NULL));
}

Gmeans::~Gmeans()
{
  
  int i;
  for (i = 0; i < n_Clusters; i++)
    {
      delete [] Sim_Mat[i];
      delete [] Concept_Vector[i];
      delete [] old_CV[i];
    }
  delete [] Sim_Mat;
  delete [] Concept_Vector;
  delete [] old_CV;

  delete [] cluster_quality;
  delete [] diff;
  delete [] ClusterSize;

  if (f_v_times > 0)
  {
    for (i = 0; i < n_Clusters; i++)
      delete [] quality_change_mat[i];
    delete [] quality_change_mat;
  }
}


void Gmeans::compute_cluster_size()
{
  int i, k;
  
  k=0;
  for (i = 0; i < n_Clusters; i++)
    ClusterSize[i] = 0;
  for (i = 0; i < n_Docs; i++)
    {
      while (i<empty_Docs_ID[k])
	{
	  ClusterSize[Cluster[i]]++;
	  i++;
	}
      k++;
    }
}

float Gmeans::coherence(int n_clus)
{
  int i;
  float value = 0.0; 
  
  /*
  int k=0;
 
  for (i = 0; i < n_Docs; i++)
    { 
      while (i<empty_Docs_ID[k])
	{
	  value +=Sim_Mat[Cluster[i]][i] ;
	  i++;
	}
      k++;
    }
  */
  for (i = 0; i < n_clus; i++)
    value +=cluster_quality[i]; 
  
  return value+n_clus*Omiga;
}

void Gmeans::general_means(Matrix *p_Docs, int up_bound, int f_v, char * seeding_file)
{
  int n_Runs, worst_vector;
  float  change;
  bool dumped=false;

  stablized = false;
  Result =0.0;
  f_v_times = f_v;
  if ((f_v_times>0) && (i_k_means_times>0))
    i_k_means_times =0;

  initialize_cv(p_Docs, seeding_file);
  n_Runs =0;
  mark = new int [f_v_times+i_k_means_times];

  cout<<endl<<"Now clustering..."<<endl;

  do
    {
      pre_Result = Result;
      n_Runs ++;
      clear_mark();
      cout<<endl<<"* Run "<<n_Runs<<" *"<<endl;
      if ( skip_SPKM )
	cout<<"General K-Means is skipped"<<endl;
      else
	general_k_means(p_Docs);
      if (dumpswitch)
	dumped = true;
      if (verify)
	cout<<"Verify obj. func. : "<<verify_obj_func(p_Docs,n_Clusters)<<endl;
      worst_vector = find_worst_vectors(dumped);
      dumped =false;
      if ((f_v_times+i_k_means_times > 0) && (n_Clusters > 1))
	{
	  pre_Result = Result;
	  change=K_L_first_variation(p_Docs);
	  Result += change;
	  //if (dumpswitch && obj_inc.good())
	  //  obj_inc<<"1\t0\t"<<Result<<"\t0\t0\t"<<Result<<endl;
	}
      else
	change =0;
      if(change == 0)
	 remove_Empty_Clusters();
      //if((fabs(change) <= FLOAT_ERROR) && (n_Clusters <up_bound))
      if ((n_Clusters <up_bound) && (change ==0))
	{
	  pre_Result = Result;
	  change = Split_Clusters(p_Docs, worst_vector,  FLOAT_ERROR);
	  Result += change;
	  //if (dumpswitch && obj_inc.good())
	  //  obj_inc<<"\t\t\t"<<change<<endl;
	}
      cout<<endl<<"^ obj. change pre-obj : "<<Result<<" "<<change<<" "<<pre_Result<<endl;
      if (verify)
	cout<<"Verify obj. func. : "<<verify_obj_func(p_Docs,n_Clusters)<<endl;
    }
  while (change != 0);
  if (Alg == DIAMETRIC_K_MEANS)
    neg_pos_seperation(p_Docs);
  compute_cluster_size();
  cout<<endl<<endl<<"~ Final clusters are: ~"<<endl<<endl;
  outputClusterSize();
  generate_Confusion_Matrix (label, n_Class);
  
  cout <<endl<<endl<<"* Clustering is done!. *"<<endl<<endl<<endl;
  if (dumpswitch && obj_inc.good())
    obj_inc.close();
  purity_Entropy_MutInfo();
  F_measure(label, n_Class);
  micro_avg_precision_recall();
  cout<<endl;
}

void Gmeans::neg_pos_seperation(Matrix *p_Docs)
{
  int i, j;
  for (i = 0; i < n_Docs; i++)
    {
      if (p_Docs->dot_mult (Concept_Vector[Cluster[i]], i) <=0)
	Cluster[i] = 2*Cluster[i];
      else
	Cluster[i] = 2*Cluster[i]+1;
    }
  
  new_Concept_Vector = new float*[2*n_Clusters];
  for (i = 0; i <2*n_Clusters; i++)
    new_Concept_Vector[i] = new float [n_Words];

  
  for (i = 0; i < 2*n_Clusters; i++)
    for (j = 0; j < n_Words; j++)
      new_Concept_Vector[i][j] = 0.0;
  
  for (i = 0; i < n_Clusters-1; i++)
    delete[] Concept_Vector[i];
  delete [] Concept_Vector;
  Concept_Vector=new_Concept_Vector;    
  
  for (i = 0; i < n_Docs; i++)
    p_Docs->ith_add_CV(i, Concept_Vector[Cluster[i]]);
  
  for (i = 0; i < 2*n_Clusters; i++)
    normalize_vec(Concept_Vector[i], n_Words);

  delete []ClusterSize;
  ClusterSize = new int [2*n_Clusters];
  
  n_Clusters *= 2;
}

void Gmeans::wordCluster(char *output_matrix, int n_C)
{
  char browserfilepost[128];
  //double max; 
  int i, j;

  WordCluster = new int[n_Words];
  int index;
  float max;

  for (j = 0; j < n_Words; j++)
    {
      max = Concept_Vector[0][j];
      index =0;
      
      for (i = 1; i < n_Clusters; i++)
	if (Concept_Vector[i][j] > max) 
	  {
	    max = Concept_Vector[i][j];
	    index =i;
	  }
      WordCluster[j]=index;
    }
  sprintf(browserfilepost, "_wordtoclus.%d", n_C);
  strcat(output_matrix, browserfilepost);
  std::ofstream o_m(output_matrix);
  
  o_m<< n_Words <<endl;

  for (j = 0; j < n_Words; j++)
    o_m<<WordCluster[j]<<endl;
 
  o_m.close();
  cout <<output_matrix<<endl;
}

void Gmeans::F_measure (int *label, int n_Class)
{
  int **confusion_matrix, i, j, k;
  
   // generate the confusion matrix
    if(n_Class >0)
      {
	confusion_matrix = new int* [n_Clusters];
	for (i=0; i<n_Clusters; i++)
	  confusion_matrix[i] = new int [n_Class];
	for(i=0; i<n_Clusters;i++)
	  for (j=0; j<n_Class; j++)
	    confusion_matrix[i][j]=0;
	  
	k=0;
	for (i = 0; i < n_Docs; i++)
	  {
	    while (i<empty_Docs_ID[k])
	      {
		confusion_matrix[Cluster[i]][label[i]]++;
		i++;
	      }
	    k++;
	  }
	float **Recall, **Precision, **F;
	float temp_max, F_value;

	Recall = new float* [n_Clusters];
	for (i=0; i<n_Clusters; i++)
	  Recall[i] = new float[n_Class];
	Precision = new float* [n_Clusters];
	for (i=0; i<n_Clusters; i++)
	  Precision[i] = new float[n_Class];
	F = new float* [n_Clusters];
	for (i=0; i<n_Clusters; i++)
	  F[i] = new float[n_Class];
	for (i = 0; i < n_Clusters; i++)
	  for (j = 0; j < n_Class; j++)
	    Recall[i][j] = confusion_matrix[i][j]*1.0/CategorySize[j];
	for (i = 0; i < n_Clusters; i++)
	  for (j = 0; j < n_Class; j++)
	    Precision[i][j] = confusion_matrix[i][j]*1.0/ClusterSize[i];
	for (i = 0; i < n_Clusters; i++)
	  for (j = 0; j < n_Class; j++)
	    F[i][j]= 2.0*Recall[i][j]*Precision[i][j]/(Recall[i][j]+Precision[i][j]);
	F_value =0.0;
	for (j = 0; j < n_Class; j++)
	  {
	    temp_max =0.0;
	    for (i = 0; i < n_Clusters; i++)
	      if (temp_max < F[i][j])
		temp_max = F[i][j];
	    F_value += temp_max*CategorySize[j];
	  }
	F_value /= n_Docs;
	cout <<endl<<"F-measure value is: "<< F_value<<endl;
	for(i=0; i<n_Clusters;i++)
	  {
	    delete [] Recall[i];
	    delete [] Precision[i];
	    delete [] F[i];
	    delete [] confusion_matrix[i];
	  }
	delete [] F;
	delete [] Precision;
	delete [] Recall;
	delete [] confusion_matrix;
      }
}

void Gmeans::generate_Confusion_Matrix (int *label, int n_Class)
{
  int **confusion_matrix, i, j, k;
  
   // generate the confusion matrix
    if(n_Class >0)
      {
	confusion_matrix = new int* [n_Clusters];
	for (i=0; i<n_Clusters; i++)
	  confusion_matrix[i] = new int [n_Class];
	for(i=0; i<n_Clusters;i++)
	  for (j=0; j<n_Class; j++)
	    confusion_matrix[i][j]=0;
	
	k=0;
	for (i = 0; i < n_Docs; i++)
	  {
	    while (i<empty_Docs_ID[k])
	      {
		confusion_matrix[Cluster[i]][label[i]]++;
		i++;
	      }
	    k++;
	  }
	for (i = 0; i < n_Clusters; i++)
	  {
	    printf("\n#%d", i);
	    for (j = 0; j < n_Class; j++)
	      printf("\t%d", confusion_matrix[i][j]);
	    fprintf(stdout, "\n");
	  }
	for (i=0; i<n_Clusters; i++)
	  delete [] confusion_matrix[i];
	delete [] confusion_matrix;
      }
}

void Gmeans::micro_avg_precision_recall()
  /* for the definition of micro-average precision/recall see paper "Unsupervised document classification
     using sequential information maximization" by N. Slonim, N. Friedman and N. Tishby */

{
  int i, j, k, temp, temp1, **confusion_matrix;
  int *uni_label, *alpha, *beta, *gamma;
  float p_t, r_t;

  if (n_Class >0)
    {
      confusion_matrix = new int* [n_Clusters];
      for (i=0; i<n_Clusters; i++)
	confusion_matrix[i] = new int [n_Class];
      for(i=0; i<n_Clusters;i++)
	for (j=0; j<n_Class; j++)
	  confusion_matrix[i][j]=0;
      
      k=0;
      for (i = 0; i < n_Docs; i++)
	{
	  while (i<empty_Docs_ID[k])
	    {
	      confusion_matrix[Cluster[i]][label[i]]++;
	      i++;
	    }
	  k++;
	}
      
      uni_label = new int [n_Clusters];
      for (i=0; i<n_Clusters; i++)
	{
	  uni_label[i] =0;
	  temp = confusion_matrix[i][0];
	  for (j=1; j<n_Class; j++)
	    if (temp < confusion_matrix[i][j])
	      {
		temp = confusion_matrix[i][j];
		uni_label[i] = j;
	      }
	}
      alpha = new int [n_Class];
      beta  = new int [n_Class];
      gamma = new int [n_Class];
      for (j=0; j<n_Class; j++)
	{
	  alpha[j] =0;
	  beta[j] =0;
	  gamma[j] = 0;
	}
      k=0;
      for (i = 0; i < n_Docs; i++)
	{
	  while (i<empty_Docs_ID[k])
	    {
	      if (uni_label[Cluster[i]] == label[i])
		alpha[label[i]] ++;
	      else
		{
		  beta[uni_label[Cluster[i]]]++;
		  gamma[label[i]] ++;
		}
	      i++;
	    }
	  k++;
	}      
      temp = temp1 =0;
      for (j=0; j<n_Class; j++)
	{
	  temp += alpha[j];
	  temp1 += beta[j];
	}
      temp1 += temp;
      p_t = temp *1.0/temp1;
      temp1 =0;
      for (j=0; j<n_Class; j++)
	temp1 += gamma[j];
      temp1 += temp;
      r_t = temp*1.0/temp1;
      cout<<"Micro-average precision is : "<<p_t<<endl;
      cout<<"Micro-average recall is    : "<<r_t<<endl;
      for(i=0; i<n_Clusters;i++)
	delete [] confusion_matrix[i];
      delete [] confusion_matrix;
      delete [] uni_label;
      delete [] alpha;
      delete [] beta;
      delete [] gamma;

    }    
}


void Gmeans::purity_Entropy_MutInfo()
{
   double sum, mut_info=0.0, max;
   int *sum_row, *sum_col;
   int i, j, k, **confusion_matrix;
   double average_purity=0.0, average_entropy=0.0;

   if (n_Class >0) 
     {
       confusion_matrix = new int* [n_Clusters];
       for (i=0; i<n_Clusters; i++)
	 confusion_matrix[i] = new int [n_Class];
       sum_row = new int [n_Clusters];
       sum_col = new int [n_Class];
       for(i=0; i<n_Clusters;i++)
	 for (j=0; j<n_Class; j++)
	   confusion_matrix[i][j]=0;
	       
       k=0;
       for (i = 0; i < n_Docs; i++)
	 {
	   while (i<empty_Docs_ID[k])
	     {
	       confusion_matrix[Cluster[i]][label[i]]++;
	       i++;
	     }
	   k++;
	 }
			
       for(i=0;i<n_Clusters; i++)
	 {
	   sum=0.0;
	   max=-1;
	   for(j=0;j<n_Class;j++)
	     {
	       if (max<confusion_matrix[i][j])
		 max=confusion_matrix[i][j];
	       if(ClusterSize[i]!=0 && confusion_matrix[i][j]!=0)
		 sum+=(double)confusion_matrix[i][j]/ClusterSize[i]*log((double)ClusterSize[i]/confusion_matrix[i][j])/log((double)n_Class);
	     }
	   if(ClusterSize[i]!=0)
	     {  
	       cout<<"Purity of cluster  "<<i<<" is : "<<(double)max/ClusterSize[i]<<endl;
	       cout<<"Entropy of cluster "<<i<<" is : "<<sum<<endl;
	       average_purity+=(double)max/ClusterSize[i];
	     } 
	   
	   average_entropy+=sum;
	 }
       
       cout<<"Average Purity is       : "<<average_purity/n_Clusters<<endl;
       cout<<"Average Entropy is      : "<<average_entropy/n_Clusters<<endl;
       
       for(i=0; i<n_Clusters; i++)
	{
	  sum_row[i] =0;
	  for(int k=0;k<n_Class;k++)
	    sum_row[i] +=confusion_matrix[i][k];
	  
	}
       for(int k=0;k<n_Class;k++)
	 {
	   sum_col [k]=0;
	   for(i=0; i<n_Clusters; i++)
	     sum_col[k] += confusion_matrix[i][k];
	   
	 }

       for(i=0; i<n_Clusters; i++)
	 for(int k=0;k<n_Class;k++)
	   if (confusion_matrix[i][k] >0)
	     mut_info+=confusion_matrix[i][k]*log(confusion_matrix[i][k]*(n_Docs-n_Empty_Docs)*1.0/(sum_row[i]*sum_col[k]));

       mut_info /=n_Docs-n_Empty_Docs;
       
       float hx, hy, min;
       hx= hy =0;
       for (i=0; i<n_Clusters; i++)
	 if (sum_row[i]>0)
	   hx += sum_row[i]*log(sum_row[i]);    
       hx =log(n_Docs-n_Empty_Docs)-hx/(n_Docs-n_Empty_Docs);
       for (i=0; i<n_Class; i++)
	 if (sum_col[i]>0)
	   hy += sum_col[i]*log(sum_col[i]);
       hy =log(n_Docs-n_Empty_Docs)-hy/(n_Docs-n_Empty_Docs);
       
       //min = hx<hy?hx:hy;
       min = (hx+hy)/2;
       cout<<"(Normalized) Mutual info. of this clustering is : "<<mut_info/min<<endl;
       for(i=0; i<n_Clusters;i++)
	 delete [] confusion_matrix[i];
       delete [] confusion_matrix;
    }
}


void Gmeans::random_cluster_ID()
{
  bool *mark= new bool[n_Clusters];
  int i, j;

  for (i=0; i< n_Clusters; i++)
    mark[i] = false;
  
  int k=0;
      for (i = 0; i < n_Docs; i++)
	{
	  while (i<empty_Docs_ID[k])
	    {
	      Cluster[i] = rand_gen.GetUniformInt(n_Clusters);
	      mark[Cluster[i]] = true;
	      i++;
	    }
	  Cluster[i] =0;
	  mark[Cluster[i]] = true;
	  k++;
	}

  for (i=0; i< n_Clusters && mark[i]; i++);

  if (i < n_Clusters)
    for (j=0; j<n_Clusters; j++)
      Cluster[j] = j;

  delete [] mark;
  if(dumpswitch)
    {
      cout<<"Randomly generated cluster ID for each vector."<<endl;
    }
}

void Gmeans::seeding_file_init(Matrix *p_Docs, char * seeding_file)
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

void Gmeans::open_dumpfile(char *obj_inc_file)
{
  if (dumpswitch)
    {
      obj_inc.open(obj_inc_file);
      if (obj_inc.good())
	{
	  obj_inc<<"Iter.\tSPKM\tFV\tSPLIT\tTOTAL"<<endl;
	  cout<<"Objective function values are to be written in file : "<<obj_inc_file<<endl;
	}
    }

}
  
void Gmeans::read_cate(char *fname)
{
  int i, k, max=0, n_Doc;
  char whole_line[256];

  std::ifstream catefile(fname);
  
  if(catefile==0)
    {
      cout<<"Category file "<<fname<<" can't open.\n"<<endl;
      n_Class =0;
      return;
      //exit(1);
    }
  catefile>>n_Doc;
  catefile.getline(whole_line, 256, '\n');
  
  if(n_Doc != n_Docs)
    {
      cout<<"Wrong number of docs in true label file!!!"<<endl;
      n_Class =0;
      return;
    }

  cout<<"True label file: "<<fname<<endl;
  
  label= new int[n_Docs];
  for(i=0; i<n_Doc; i++)
    {
      catefile>>label[i];
      catefile.getline(whole_line, 256, '\n');
      if(label[i]>max)
	max=label[i];
    }
  
  catefile.close();
  
  if(max+1 >0)
    {
      cout<<endl<<"True category number: "<<max+1<<endl;
      if(n_Empty_Docs >0)
	{
	  cout<<"\nThe empty docs are categorized as:"<<endl;
	  int * temp_Cate = new int [max+1];
	  for (i=0; i< max+1; i++)
	    temp_Cate[i] = 0;
	  
	  for (i=0; i< n_Empty_Docs; i++)
	    {
	      temp_Cate[label[empty_Docs_ID[i]]]++;
	      Cluster[empty_Docs_ID[i]]= -1;
	    }
	  for (i=0; i<max+1; i++)
	    cout << temp_Cate[i]<<"\t";
	  
	  cout<<endl<<endl;
	  delete [] temp_Cate;
	}
      CategorySize = new int [max+1];
      for (i=0; i<max+1; i++)
	CategorySize[i] = 0;
      
      k=0;
      for (i = 0; i < n_Docs; i++)
	{
	  while (i<empty_Docs_ID[k])
	    {
	      CategorySize[label[i]] ++;
	      i++;
	    }
	  k++;
	}
    }
  n_Class= max+1;
  memory_consume += n_Docs *sizeof(int);
}


void Gmeans::clear_mark ()
{
  int i;

  for (i=0; i< f_v_times+i_k_means_times; i++)
    mark[i] =-1;
}

bool Gmeans::already_marked(int doc)
{
  int i;
 
  for (i=0; (i<f_v_times+i_k_means_times)&& (mark[i] >=0); i++)
    if (mark[i] == doc)
      return true;
  
  return false;
}

bool Gmeans::empty_vector(int doc)
{
  int i;
  for (i=0; i<n_Empty_Docs; i++)
    if (doc == empty_Docs_ID[i])
      return true;
 
  return false;
}

void Gmeans::update_quality_change_mat(Matrix *p_Docs, int c_ID)
  // update the quality_change_matrix for a particular cluster 
{
  int k, i;

  k=0;
 
  for (i = 0; i < n_Docs; i++)
    {
      while (i<empty_Docs_ID[k])
	{
	  quality_change_mat[c_ID][i] = delta_X (p_Docs, i, c_ID);
	  i++;
	}
      k++;
    }
}
