#
# Makefile of General-Means clustering program
# Copyright (c) 2002 Yuqiang Guan
#
# compiler: g++ version > 2.7.0

CCC		= g++
CC		= gcc
CCCFLAGS	= -Wall -g -O
CCFLAGS		= -Wall -g -O
LDFLAGS		=
INCLUDES	= -I/usr/include/g++

SRCS = gmeans.cc Gmeans.cc mat_vec.cc RandomGenerator.cc SparseMatrix.cc DenseMatrix.cc tools.cc Spherical_k_means.cc Diametric_k_means.cc Euclidean_k_means.cc Kullback_leibler_k_means.cc IB.cc Matrix.cc 
OBJS = gmeans.o Gmeans.o mat_vec.o RandomGenerator.o SparseMatrix.o  DenseMatrix.o tools.o Spherical_k_means.o Diametric_k_means.o Euclidean_k_means.o Kullback_leibler_k_means.o IB.o Matrix.o 
PM_OBJS = powermethod.o RandomGenerator.o

.SUFFIXES: .c .cc .o

.cc.o:
	$(CCC) $(CCCFLAGS) $(INCLUDES) -c $<
.c.o:
	$(CC) $(CCFLAGS) $(INCLUDES) -c $<
.f.o:
	$(F77) -c $<



gmeans-$(OSTYPE) : $(OBJS)
	$(CCC) -o gmeans-$(OSTYPE) $(OBJS) $(LDFLAGS)

powermethod : $(PM_OBJS)
	$(CCC) -o powermethod $(PM_OBJS) $(LDFLAGS)

prof : $(OBJS)
	$(CCC) -o pgmeans-$(OSTYPE) $(OBJS) $(LDFLAGS) -pg

clean :
	rm gmeans-$(OSTYPE) $(OBJS)





