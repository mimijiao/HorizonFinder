#ifndef CANNYEDGE_H
#define CANNYEDGE_H

void callGaussianBlur(float *blur_image, int width, int height,
	                  int threadsPerBlock, int numBlocks);

void callSobelMask(float *image, int *edge_dir, float *dir, float *gradient,
	               int width, int height, int threadsPerBlock, int numBlocks);

void callSuppressNonMax(int *edge_dir, float *gradient, int *edges, int *visit,
	                    float hi_thresh, float low_thresh, int width,
	                    int height, int threadsPerBlock, int numBlocks);

void callHysteresisTracing(int *edges, int *visit, bool *modified, bool done,
	                int width, int height, int threadsPerBlock, int numBlocks);

void callGetCounts(float *gradient, int *counts, int width, int height,
	               int threadsPerBlock, int numBlocks);

void cannyEdge(int *image, int *dev_edges, float *dev_dir, int width,
	           int height);

#endif
