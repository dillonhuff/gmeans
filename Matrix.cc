/* Implementation of Matrix class
 * Copyright (c) 2003, Yuqiang Guan
 */

#include <assert.h>
#include <stdlib.h>
#include <math.h>
#include <iostream.h>
#include <fstream.h>
#include "Matrix.h"

Matrix::Matrix(int r, int c)
{
  n_row =r; 
  n_col =c;
  memory_used = 0;
  norm  = NULL;
  L1_sum = Norm_sum =0;
  priors= L1_norm = NULL;
}

int Matrix::GetNumRow() { return n_row; }

int Matrix::GetNumCol() { return n_col; }

float Matrix::GetL1Sum() { return L1_sum;}

long Matrix::GetMemoryUsed() { return memory_used;}

float Matrix::GetNorm(int i) { return norm[i]; }

float Matrix::GetL1Norm(int i) { return L1_norm[i]; }

float Matrix::GetNormSum() { return Norm_sum; }

void Matrix::SetAlpha(float a, int laplace)
{ 
  switch(laplace)
    {
    case NOLAPLACE:
      alpha =0;
      break;
    case CENTER_LAPLACE:
      alpha = a;
      break;
    case PRIOR_LAPLACE:
      alpha /= n_col;
      break; 
    }  
}

float Matrix::GetAlpha(){ return alpha;}
  
void Matrix::SetPrior(int method, char * prior_file, int n_E_Docs, int *empty_Docs_ID)
{
  float inv = 1.0/(n_col-n_E_Docs);
  int i;
  char whole_line[256];
  std::ifstream gpfile(prior_file);

  if (priors ==NULL)
    {
      priors = new float [n_col];
      memory_used += n_col *sizeof(float);
    }
  switch(method)
    {
    case FILE_PRIOR:      
      if(gpfile.is_open())
	{
	  cout<<"Prior file: "<<prior_file<<" is used."<<endl;
	  gpfile>>i;
	  gpfile.getline(whole_line, 256, '\n');
	  if(i!=n_col)
	    {
	      cout<<"Wrong number of vectors!!! \nSo using uniform priors ..."<<endl;
	      for (int i=0; i<n_col; i++)
		priors[i] = inv;
	    }
	  else
	    {
	      float sum=0, temp;
	      int k=0;
	      for (i = 0; i < n_col; i++)
		{
		  while (i<empty_Docs_ID[k])
		    {
		      gpfile>>priors[i];
		      gpfile.getline(whole_line, 256, '\n');
		      sum +=priors[i];
		      i++;
		    }
		  gpfile>>temp;
		  k++;
		}
	      inv = 1.0/sum;
	      for (i=0; i<n_col; i++)
		priors[i] *= inv;
	       for (i = 0; i<n_E_Docs; i++)
		 priors[empty_Docs_ID[i]]=0;
	    }
	  gpfile.close();
	}
      else
	{
	  cout<<"Can't open file "<<prior_file<<" !!! \nSo using uniform priors ..."<<endl;
	  for (int i=0; i<n_col; i++)
	    priors[i] = inv;
	}
      break;
    case UNIFORM_PRIOR:
    default:
      int k=0;
      for (i = 0; i < n_col; i++)
	{
	  while (i<empty_Docs_ID[k])
	    {
	      priors[i] = inv;
	      i++;
	    }
	  k++;
	}
      for (i = 0; i<n_E_Docs; i++)
	priors[empty_Docs_ID[i]]=0;
    }
 
}

float Matrix::getPrior(int i) 
{
  if ( (i < n_col) && (i>=0))
    return priors[i];
  else
    return 0;
}

float Matrix::kernel_sum(int *pointer, int *index, int i)
  //compute L2 norm of ith kernel centroid
{
  int start, end;
  float kernel_norm_L2=0;

  start = pointer[i];
  end = pointer[i+1];
  for (int i=start; i<end; i++)
    for (int j = start; j < end; j++)
      kernel_norm_L2 += Sim_Mat(index[i],index[j]);
      
  return kernel_norm_L2;
}

void Matrix::kernel_norm_L2(int *cluster, int *cluster_size, int n_Clusters, float *result, int flag)
//this function computes the L2 norm of kernel centroids 
{
  int i, *pointer = new int[n_Clusters+1], *index = new int[n_col];
  pointer[0] =0;
  for (i=1; i<n_Clusters; i++)
    pointer[i] = pointer[i-1] + cluster_size[i-1];
  for (i=0; i<n_col; i++)
    {
      index[pointer[cluster[i]]] = i;
      pointer[cluster[i]] ++;
    }

  pointer[0] =0;
  for (i=1; i<=n_Clusters; i++)
    pointer[i] = pointer[i-1] + cluster_size[i-1];

  if (flag <0)
    for (i = 0; i < n_Clusters; i++)
      result[i] = sqrt(kernel_sum(pointer, index, i));
  else
    result[flag]= sqrt(kernel_sum(pointer, index, flag));
  delete [] pointer;
  delete [] index;
}


int Matrix::kernel_dot_product(int *pointer, int *index,int j, int n_Clusters, float *kernel_center_norm)
//this function computes the polynomial kernel distance between jth data with all kernel centroids
  //return the closest center index
{
  int start, end, temp_id=0;
  float result, temp =0;
  
  for (int i=0; i<n_Clusters; i++)
    {
      start = pointer[i];
      end = pointer[i+1];
      result =0;
      for (int l = start; l < end; l++)
	result +=Sim_Mat(j,index[l]);
      result /= kernel_center_norm[i];
      //cout<<"("<<j<<","<<i<<")="<<result<<endl;
      if (result >temp)
	{
	  temp_id = i;
	  temp =result;
	}
    }
  return temp_id;
}

/*
void Matrix::polynomial_kernel(int *cluster, int *cluster_size, int n_Clusters, float **result, float c, int d, float *kernel_center_norm, int flag)
//this function computes the polynomial kernel distance between all data with kernel centroids
{
  int i, *pointer = new int[n_Clusters+1], *index = new int[n_col];
  pointer[0] =0;
  for (i=1; i<n_Clusters; i++)
    pointer[i] = pointer[i-1] + cluster_size[i-1];
  for (i=0; i<n_col; i++)
    {
      index[pointer[cluster[i]]] = i;
      pointer[cluster[i]] ++;
    }

  pointer[0] =0;
  for (i=1; i<=n_Clusters; i++)
    pointer[i] = pointer[i-1] + cluster_size[i-1];
  
  if (flag <0)
    for (i = 0; i < n_Clusters; i++)
      polynomial_kernel(pointer, index, i, c, d, kernel_center_norm, result[i]);
  else
    polynomial_kernel(pointer, index, flag, c, d,kernel_center_norm, result[i]);
  delete [] pointer;
  delete [] index;
}
*/
float Matrix::phi_x_dot_phi_y(int i, int j, float c, int d)
{
  float result;

  result = i_j_dot_product(i,j)+c;
  for(int k=1; k<d; k++)
    result *= result;
  return result;
}

void Matrix::pair_wise_dot_product(float constant, int degree, int alg)
//this function computes pair-wise dot product between vectors in the kernel space
  //essentially it is A^T * A
{
  int i, j;
  float temp;

  cout<<"Computing pair-wise kernels..."<<endl; 
  Sim_Mat = new VECTOR_double[n_col];
  for (j = 0; j < n_col; j++)
    Sim_Mat[j] = new float[j+1]; 
  memory_used+=(n_col+1)*n_col/2*sizeof(float);
  
  switch (alg)
    {
    case KERNEL_SPHERICAL_K_MEANS:
      for (i = 0; i < n_col; i++)
	if((temp = phi_x_dot_phi_y(i, i, constant,degree))>0)
	  Sim_Mat[i][i] = 1.0/sqrt(temp);
	else
	  Sim_Mat[i][i] = 0;
      for(i = 1; i < n_col; i++)
	for(j = 0; j <i; j++)
	  Sim_Mat[i][j] = phi_x_dot_phi_y(i, j, constant,degree)*Sim_Mat[i][i]*Sim_Mat[j][j];
      
      for (i = 0; i < n_col; i++)
	if (Sim_Mat[i][i]>0)
	  Sim_Mat[i][i] = 1;
	else
	  Sim_Mat[i][i] = 0;
      break;
    case KERNEL_EUCLIDEAN_K_MEANS:
      for (i = 0; i < n_col; i++)
	for(j = 0; j <=i; j++)
	  Sim_Mat[i][j] = phi_x_dot_phi_y(i, j, constant,degree);
      break;
    }
      /*
	for (i = 0; i < n_col; i++)
	{
	for(j = 0; j <=i; j++)
	cout<<Sim_Mat[i][j]<<"\t";
	cout<<endl;
	}*/
}

float Matrix::kernel_euc_dis(int *pointer, int *index, int j, float *kernel_center_sum, int cluster_id)
  //this function computes the polynomial kernel Euclidean distance between jth data with the 'cluster_id'-th kernel centroid
  //return Euclidean distance between jth data the 'cluster_id'-th center
{
  int start, end;
  float temp, cluster_size_inv;
  
  start = pointer[cluster_id];
  end = pointer[cluster_id+1];
  cluster_size_inv = 1.0/(end-start);
  temp = 0;
  for (int l = start; l < end; l++)
    temp -=Sim_Mat(j,index[l]);
  temp *= 2*cluster_size_inv;
  temp += kernel_center_sum[cluster_id]*cluster_size_inv*cluster_size_inv + Sim_Mat[j][j];
  return temp;
}

int Matrix::kernel_euc_dis(int *pointer, int *index,int j, int n_Clusters, float *kernel_center_sum)
  //this function computes the polynomial kernel Euclidean distance between jth data with all kernel centroids
  //return the closest center index
{
  int temp_id=0;
  float result, temp;
  
  temp = kernel_euc_dis(pointer, index, j, kernel_center_sum, 0);
  for (int i=1; i<n_Clusters; i++)
    {
      result = kernel_euc_dis(pointer, index, j, kernel_center_sum, i);
      //cout<<"("<<j<<","<<i<<")="<<result<<endl;
      if (result < temp)
	{
	  temp_id = i;
	  temp =result;
	}
    }
  return temp_id;
}

float Matrix::pair_wise_kernel_euc_dis(int i, int j)
  // compute Euclidean distance between vector i and j in kernel space
{
  return Sim_Mat[i][i]-2*Sim_Mat[i][j]+Sim_Mat[j][j];
}
  
void Matrix::cluster_kernel_sum(int *cluster, int *cluster_size, int n_Clusters, float *result, int flag)
//this function computes the sum of kernels of clusters
{
  int i, *pointer = new int[n_Clusters+1], *index = new int[n_col];
  pointer[0] =0;
  for (i=1; i<n_Clusters; i++)
    pointer[i] = pointer[i-1] + cluster_size[i-1];
  for (i=0; i<n_col; i++)
    {
      index[pointer[cluster[i]]] = i;
      pointer[cluster[i]] ++;
    }

  pointer[0] =0;
  for (i=1; i<=n_Clusters; i++)
    pointer[i] = pointer[i-1] + cluster_size[i-1];

  if (flag <0)
    for (i = 0; i < n_Clusters; i++)
      result[i] = kernel_sum(pointer, index, i);
  else
    result[flag]= kernel_sum(pointer, index, flag);
  delete [] pointer;
  delete [] index;
}
