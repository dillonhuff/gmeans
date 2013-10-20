#include <time.h>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include "mat_vec.h"
#include "IB.h"
#include "Spherical_k_means.h"
#include "Diametric_k_means.h"
#include "Euclidean_k_means.h"
#include "Kullback_leibler_k_means.h"
#include "timerUtil.h"
#include "tools.h"

#define MAX_VER_DIGIT_LENGTH 	3

//memory usage statistics
long memory_consume=0l;


//main function comes here
int main(int argc, char **argv)
{
  int *cluster;
  doc_mat mat;
  dense_mat d_mat;

  bool no_args = true;
  
  float coherence;
  
  char version[] = "1.0";

  // default parameters for the k-means algorithm
  int alg = SPHERICAL_K_MEANS;
  
  int format = SPARSE_MATRIX;
  int  n_Clusters = 1;
  float epsilon = DEFAULT_EPSILON;
  float delta = DEFAULT_DELTA;
  int init_method = WELL_SEPARATED_CENTROID_INIT_MODIFY;
  float perturb = DEFAULT_PERTURB;
  float Omiga=0.0;
  float alpha=DEFAULT_ALPHA;
  int up_bound=0;
  int * e_D_ID=NULL, n_Empty_Docs=0;
  int first_var_move_num=0, incremental_k_means_moves=0;
  int total_run=1, seeds=0, sim_est_start=5, use_laplacian= NOLAPLACE;
  bool dumpinfo =false, random_seeding=true, skip_SPKM = false, verify = false, evaluate_only=false, generate_Word_Clustering = false;
  
  
  char output_matrix[FILENAME_MAXLENGTH]="";
  char wordCluster[FILENAME_MAXLENGTH]="";
  char suffix[SCALE_SURFIX_LENGTH]="tfn";

  char docs_matrix[FILENAME_MAXLENGTH];
  char seeding_file[FILENAME_MAXLENGTH] ="";  
  char pattern_file[FILENAME_MAXLENGTH] ="";
  char dump_file[FILENAME_MAXLENGTH] ="";
  
  int prior_method = UNIFORM_PRIOR;
  char prior_file[FILENAME_MAXLENGTH] ="";
  //////////////////////////////////////////////////////////////
  //----------------- Read commandline arguments ---------------
  
  
  TimerUtil runtime_est;

  for (argv++; *argv != NULL; argv++)
    {
      if ((*argv)[0] == '-')
	{
	  switch ((*argv)[1])
	    {
	    case 'a':
	      if ((*(++argv))[0] == 's')
		alg = SPHERICAL_K_MEANS;
	      else if((*(argv))[0] == 'e')
		alg = EUCLIDEAN_K_MEANS;
	      else if ((*(argv))[0] == 'k')
		alg = KULLBACK_LEIBLER;
	      else if ((*(argv))[0] == 'd')
		alg = DIAMETRIC_K_MEANS;
	      else if ((*(argv))[0] == 'b')
		alg = INFO_BOTTLENECK;
	      else
		{
		  cout <<"Invalid algorithm "<<*argv<<endl;
		  print_help();
		  exit(0);
		}
	      break;
	    case 'c':
	      n_Clusters = atoi(*(++argv));
	      break;
	   
	    case 'D':
	      delta = atof(*(++argv));
	      break;
	    case 'd':
	      dumpinfo=true;
	      break;
	    case 'E':
	      sim_est_start = atoi(*(++argv));
	      break;
	    case 'e':
	      epsilon = atof(*(++argv));
	      break;
	    case 'F':
	      if ((*(++argv))[0] == 'd')
		format = DENSE_MATRIX;
	      else if((*(argv))[0] == 's')
		format = SPARSE_MATRIX;
	      else if((*(argv))[0] == 't')
		format = DENSE_MATRIX_TRANS;
	      else
		{
		  cout <<"Invalid algorithm "<<*argv<<endl;
		  print_help();
		  exit(0);
		}
	      break;
	    case 'f':
	      first_var_move_num = atoi(*(++argv));
	      break;
	    
	    case 'i':
	      switch ((*(++argv))[0])
		{
		case 'p':
		  init_method = RANDOM_PERTURB_INIT;
		  break;
		case 'c':
		  init_method = CONCEPT_VECTORS_INIT;
		  break;
		case 'r':
		  init_method = RANDOM_CLUSTER_ID_INIT;
		  break;
		case 's':
		  init_method = WELL_SEPARATED_CENTROID_INIT;
		  break;
		case 'S':
		  init_method = WELL_SEPARATED_CENTROID_INIT_MODIFY;
		  break;
		case 'f':
		  init_method = SEEDINGFILE_INIT;
		  strcpy(seeding_file, *(++argv));
		  break;  
		default:
		  printf("Invalid option %s\n", *argv);
		  print_help();
		  exit(0);
		  break;
		}
	      break;
	    case 'I':
	      incremental_k_means_moves = atoi(*(++argv));
	      break;
	    case 'L':
	      switch ((*(++argv))[0])
		{
		case 'c':
		  use_laplacian = CENTER_LAPLACE;
		  alpha = atof(*(++argv));
		  break;
		  //case 'p':
		  //use_laplacian = PRIOR_LAPLACE;
		  //break;
		case 'n':
		  use_laplacian = NOLAPLACE;
		  break;
		}
	      break;
	      /*case 'M':
	      switch ((*(++argv))[0])
		{
		case 'a':
		  model_selection_criterion = AIC;
		  break;
		case 'b':
		  model_selection_criterion = BIC;
		  break;
		case 'd':
		  model_selection_criterion = MDL;
		  break;
		case 'm':
		  model_selection_criterion = MML;
		  break;
		}
		break;*/
	    case 'O':
	      strcpy(output_matrix, *(++argv));
	      strcpy(wordCluster, output_matrix);
	      break;
	    case 'P':
	      switch ((*(++argv))[0])
		{
		case 'u':
		  prior_method = UNIFORM_PRIOR;
		  break;
		case 'f':
		  prior_method = FILE_PRIOR;
		  strcpy(prior_file, *(++argv));
		  break;
		}
	    case 'p':
	      perturb = atof(*(++argv));
	      break;
	    case 'R':
	      total_run = atoi(*(++argv));
	      break;
	    case 'S':
	      skip_SPKM = true;
	      break;
	    case 's':
	      random_seeding = false;
	      seeds = atoi(*(++argv));
	      break;
	    case 'T':
	      strcpy(pattern_file, *(++argv));
	      break;
	    case 't':
	      strcpy(suffix, *(++argv));
	      break;
	    case 'U':
	      evaluate_only = true;
	      init_method = SEEDINGFILE_INIT;
	      strcpy(seeding_file, *(++argv));
	      break;  
	    case 'u':
	      up_bound = atoi(*(++argv));
	      break;
	    case 'V':
	      verify = true;
	      break;
	    case 'v':
	      cout<<"G-Means version "<<version<<endl;
	      exit (0);
	    case 'W':
	      generate_Word_Clustering = true;
	      break;
	    case 'w':
	      Omiga = atof(*(++argv));
	      break;
	    default:
	      printf("Invalid switch %s\n", *argv);
	      print_help();
	      exit(0);
	      break;
	    }
	}
      else
	{
	  sprintf(docs_matrix, "%s",*argv);
	  if (strcmp(output_matrix, "") ==0)
	    extractfilename(*argv, output_matrix);
	  strcpy(wordCluster, output_matrix);
	  strcpy(dump_file, output_matrix);
	  switch (format)
	    {
	    case DENSE_MATRIX:
	    case DENSE_MATRIX_TRANS:
	      e_D_ID= read_mat(*argv, &d_mat, n_Empty_Docs, format);
	      break;
	    case SPARSE_MATRIX:
	    default: 
	      e_D_ID= read_mat(*argv,suffix,  &mat, n_Empty_Docs);
	      strcat(wordCluster, "_");
	      strcat(dump_file, "_");
	      strcat(wordCluster, suffix);
	      strcat(dump_file, suffix);
	    }
	  strcat(dump_file, "_dump");
	  no_args = false;
	}
    }
  
  if (no_args)
    {
      print_help();
      exit(1);
    }
  //////////////////////////////////////////////////////////////

  int final_cluster_num = n_Clusters;
  
  Matrix *matrix;
  Gmeans *K;
  if ((format ==  DENSE_MATRIX) || (format ==  DENSE_MATRIX_TRANS))
    matrix = new DenseMatrix(d_mat.n_row, d_mat.n_col, d_mat.val); 
  else 
    matrix = new SparseMatrix(mat.n_row, mat.n_col, mat.n_nz, mat.val, mat.row_ind, mat.col_ptr);
  
  switch(alg)
    {
    
    case DIAMETRIC_K_MEANS:
      matrix->normalize_mat_L2();
      break;
    case EUCLIDEAN_K_MEANS:
      //normalize_mat_1(&R);
      matrix->ComputeNorm_2();
      break;
    case KULLBACK_LEIBLER:
      matrix->SetAlpha(alpha, use_laplacian);
      matrix->SetPrior(prior_method, prior_file, n_Empty_Docs, e_D_ID);
      matrix->normalize_mat_L1();
      matrix->ComputeNorm_KL(use_laplacian);
      //cout<<matrix->MutualInfo()<<endl;
      break;
    case INFO_BOTTLENECK:
      matrix->SetPrior(prior_method, prior_file, n_Empty_Docs, e_D_ID);
      matrix->normalize_mat_L1();
      matrix->ComputeNorm_KL(NOLAPLACE);
      break;
    case SPHERICAL_K_MEANS:
    default:
      matrix->normalize_mat_L2();
      break;    
     
    }

  if (up_bound <=n_Clusters)
    up_bound = n_Clusters;
  if (matrix->GetNumCol()-n_Empty_Docs < up_bound)
    {
      up_bound = matrix->GetNumCol()-n_Empty_Docs;
      cout <<"The number of clusters you asked for is greater than the number of non-empty vectors. We adjust the number of clusters to be the number of data items."<<endl;
    }
  if (matrix->GetNumCol() < first_var_move_num)
    {
      first_var_move_num = matrix->GetNumCol();
      cout <<"The number of first variations you asked for is greater than the number of data items. We adjust the number of clusters to be the number of data items."<<endl;
    }
  cluster = new int[matrix->GetNumCol()];
  
  switch (alg)
    {
    
    case DIAMETRIC_K_MEANS:
      K= new Diametric_k_means (matrix, cluster, n_Clusters, sim_est_start, Omiga, random_seeding, seeds, epsilon);
      break;
    case EUCLIDEAN_K_MEANS:
      K= new Euclidean_k_means (matrix, cluster, n_Clusters, sim_est_start, Omiga, random_seeding, seeds, epsilon);
      break;
    case KULLBACK_LEIBLER :
      K= new Kullback_leibler_k_means(matrix, cluster, n_Clusters, sim_est_start, Omiga, random_seeding, seeds, epsilon);
      break;
    case INFO_BOTTLENECK:
      K= new IB(matrix, cluster, n_Clusters, sim_est_start, Omiga, random_seeding, seeds, epsilon);
      break;
    case SPHERICAL_K_MEANS:
    default:
      K= new Spherical_k_means (matrix, cluster, n_Clusters, sim_est_start, Omiga, random_seeding, seeds, epsilon);
      break;
    }
  
  K->SetAlgorithm(alg);
  K->SetEmptyDocs(n_Empty_Docs, e_D_ID);
  K->setDumpinfo(dumpinfo);
  K->setInitialClustering(init_method);
  K->setSkipSPKM(skip_SPKM);
  K->setVerify(verify);
  K->setLaplacian(use_laplacian);
  K->setIncemental_k_means(incremental_k_means_moves);
  K->SetEpsilon(epsilon);
  K->setDelta(delta);
  K->setEvaluate(evaluate_only);
 

  if (init_method == RANDOM_PERTURB_INIT)
    K->SetPerturb(perturb);

  K->read_cate(pattern_file);
  K->open_dumpfile(dump_file);
  K->general_means(matrix, up_bound, first_var_move_num,seeding_file);
  
  coherence = K->GetResult();

  final_cluster_num = K->getClusterNum();

  if (generate_Word_Clustering)
    {
      cout<<"\nWriting word cluster file : ";
      K->wordCluster(wordCluster, final_cluster_num);
    }

  output_clustering(cluster,matrix->GetNumCol(), final_cluster_num, output_matrix, suffix, format, docs_matrix);
  cout << "\nobjective function value: " << coherence << "\n\n"; 
  cout << "Final number of clusters : " << final_cluster_num << "\n";	
  cout << "Number of documents  : " << matrix->GetNumCol()<< "\n";
  
  switch (alg)
    {
    
    case SPHERICAL_K_MEANS:
      cout << "Algorithm            : spherical k-means\n";
      break;
    case DIAMETRIC_K_MEANS:
      cout << "Algorithm            : diametric k-means\n";
      break;
    case EUCLIDEAN_K_MEANS:
      cout << "Algorithm            : euclidian k-means\n";
      break;
    case KULLBACK_LEIBLER:
      cout << "Algorithm            : kullback_leibler k-means\n";
      cout<<"The mutual information of the original matrix is : " <<matrix->MutualInfo()<<endl;
      cout<<"The mutual information of the final matrix is    : "<<matrix->MutualInfo()-coherence<<" ("
	  <<(matrix->MutualInfo()-coherence)*100.0/matrix->MutualInfo()<<"%)"<<endl;
      switch (use_laplacian)
	{
	case CENTER_LAPLACE:
	  cout << "Laplace's rule is applied to the centroids.\n";
	  cout<<"The prior used is : "<<alpha<<endl;
	  break;
	case PRIOR_LAPLACE:
	  cout << "Laplace's rule is applied to the input probabilty vectors.\n";
	  break;
	}
      break;
    case INFO_BOTTLENECK:
      cout << "Algorithm            : sequential information bottleneck.\n";
      cout<<"The mutual information of the original matrix is : " <<matrix->MutualInfo()<<endl;
      cout<<"The mutual information of the final matrix is    : "<<coherence<<" ("
	  <<coherence*100.0/matrix->MutualInfo()<<"%)"<<endl;
      break;
    }
 
  cout << "Epsilon              : " << epsilon << " (used for Kmeans)\n";
  cout<<  "Omiga                : "<<Omiga <<" (used for fv and split)\n";
  cout<<"Delta                : "<<delta<<" (used for fv and split)\n";
  cout << "Initialization method: ";
  switch (init_method)
    {
    case RANDOM_PERTURB_INIT:
      cout << "random perturbation\n";
      cout << "perturbation magnitude: " << perturb << "\n";
      break;
    case RANDOM_CLUSTER_ID_INIT:
      cout << "Randomly generated cluster ID for each vector.\n";
      break;
    case SEEDINGFILE_INIT:
      cout<<"Initialized with seeding file "<<seeding_file<<endl;
      break;
    case WELL_SEPARATED_CENTROID_INIT:
      cout<<"Cluster centroids are chosen to be well separated from each other, starting with a random chosen vector"<<endl;
      break;
    case WELL_SEPARATED_CENTROID_INIT_MODIFY:
      cout<<"Cluster centroids are chosen to be well separated from each other, starting with a vector farthest from the centroid of the whole data set"<<endl;
      break;
    case CONCEPT_VECTORS_INIT:
      cout<<"Randomly chose centroids"<<endl;
      break;
    } 
  cout<<"\nOutput clustering file is: "<<output_matrix<<endl<<endl;

  runtime_est.setStopTime(cout, "Computation time: ");
  cout<<"\nMemory consumed : "<<memory_consume<<" bytes"<<endl<<endl<<endl;
  
}


