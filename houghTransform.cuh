#ifndef HOUGH_H
#define HOUGH_H

void callLinearHough(int *edges, int *votes, int numVotes, int width,
	                 int height, int threadsPerBlock, int numBlocks);

void callFastHough(int *edge_coord, int *votes, int rhoSize, int width,
	               int height, int threadsPerBlock, int numBlocks);

void callPolyHough(int *edges, float *edge_dir, int *votes, int pMax, int width,
	               int height, int threadsPerBlock, int numBlocks);

void callGetMax(int *votes, int *maxIndex, int *maxVotes, int numVotes,
	            int threadsPerBlock, int numBlocks);

int houghTransform(int *dev_edges, float *dev_dir, int width, int height);

#endif
