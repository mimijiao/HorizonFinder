#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <string>
#include <cassert>
#include "houghTransform.cuh"
#include "cannyEdge.cuh"

#define MAX(a, b) ((a > b) ? a : b)
#define MIN(a, b) ((a < b) ? a : b)
#define PI 3.14159265
#define ANGLE_RES 720 /* equal to 180/resolution (in degrees) */

using namespace std;

#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code,
                      const char *file,
                      int line,
                      bool abort=true) {
  if (code != cudaSuccess) {
    fprintf(stderr,"GPUassert: %s %s %d\n",
            cudaGetErrorString(code), file, line);
    exit(code);
  }
}

// timing setup code
cudaEvent_t h_start;
cudaEvent_t h_stop;

#define START_TIMER() {                           \
      gpuErrChk(cudaEventCreate(&h_start));       \
      gpuErrChk(cudaEventCreate(&h_stop));        \
      gpuErrChk(cudaEventRecord(h_start));        \
}

#define STOP_RECORD_TIMER(name) {                               \
      gpuErrChk(cudaEventRecord(h_stop));                       \
      gpuErrChk(cudaEventSynchronize(h_stop));                  \
      gpuErrChk(cudaEventElapsedTime(&name, h_start, h_stop));  \
      gpuErrChk(cudaEventDestroy(h_start));                     \
      gpuErrChk(cudaEventDestroy(h_stop));                      \
}

__global__ void getEdges(int *edges, int *edge_coord, int *length, int width,
	                     int height) {
	int tid, gid, idx, sh_idx, old;

	extern __shared__ int sh_edge[];

	gid = blockIdx.x * blockDim.x + threadIdx.x;
	tid = threadIdx.x;
	idx = tid / 32;

	sh_edge[idx] = 0;

	while (gid < width * height) {
		if (edges[tid] == 255) {
			do {
				idx++;
				sh_idx = idx;
				sh_edge[idx] = tid;
			} while (sh_edge[idx] != tid);
		}
		idx = sh_idx;
		gid += blockDim.x * gridDim.x;
	}

	atomicMax(&sh_edge[0], idx);
	__syncthreads();

	old = atomicAdd(length, sh_edge[0]);

	if (tid < sh_edge[0]) {
		edge_coord[old+tid] = sh_edge[tid+1];
	}
}

/* Does hough transform for finding a line.
 * Parallelizes over the pixels and angles and for each pixel, get the line
 * going through it for every angle between 0 and 180 at some resolution defined
 * by ANGLE_RES.
 */
__global__ void linearHough(int *edges, int *votes, int numVotes, int width,
	                        int height) {
	int gid, x, y, a, idx, vote_index;
    float r, theta;

    gid = blockIdx.x * blockDim.x + threadIdx.x;

    while (gid < width * height * ANGLE_RES) {
    	idx = gid / ANGLE_RES;
    	x = idx % width;
		y = idx / width;
		a = gid % ANGLE_RES;

		/* only consider edge pixels */
    	if (edges[idx] == 255) {
    		theta = a * PI / ANGLE_RES;
    		/* Calculate rho based on angle and point */
    		r = x * cos(theta) + y * sin(theta);
    		vote_index = round(r + ((float) numVotes / 2)) * ANGLE_RES + a;
    		atomicAdd(&votes[vote_index], 1);
    	}
    	gid += blockDim.x * gridDim.x;
    }
}

/* Gets the two longest lines in the image based on the vote accumulator array.
 * The two longest lines have the most pixels voting for that line, so we find
 * the bins with the most votes.
 * It also suppresses bins that define very similar lines. If the rho value
 * is within 3 of the current bin's rho value, and the angle is within half a
 * degree (or 1 degree, if the resolution is 1 degree) of it, then we consider
 * these two lines to be the "same" line. And we only look at the maximum of
 * these "same" lines. This is because we want to find the two longest distinct
 * lines, and we do not want two "same" lines to be returned as the maximums.
 */
__global__ void getMax(int *votes, int *maxIndex, int *maxVotes, int numVotes) {
	int tid, idx, i, vote, vote2, curr, a, r, numThreads, aBound;
	bool isMax;

	extern __shared__ int sh_votes[];

	tid = threadIdx.x;
	idx = 2 * blockIdx.x * blockDim.x + tid;
	numThreads = blockDim.x;
	aBound = MAX(ANGLE_RES / 360, 1);

	sh_votes[tid] = 0;
	sh_votes[tid+2*numThreads] = 0;

	while (idx < numVotes) {
		a = idx % ANGLE_RES;
		r = idx / ANGLE_RES;
		vote = votes[idx];
		isMax = true;

		/* Make sure this line is a local maximum among lines that are similar
		 * to it. */
		for (int i = -3; i < 4; i++) {
			for (int j = -aBound; j < aBound + 1; j++) {
				if ((a + j > 0) && (a + j < ANGLE_RES) && (r + i > 0) &&
					(r + i < numVotes / ANGLE_RES)) {
					if (vote < votes[(r+i)*ANGLE_RES+(a+j)]) {
						isMax = false;
					}
				}
			}
		}

		/* Parallel reduction algorithm for finding the max. Does one
		 * comparison while loading the data so that threads aren't idle */
		if ((vote >= sh_votes[tid]) && isMax) {
			sh_votes[tid+2*numThreads] = sh_votes[tid];
			sh_votes[tid+3*numThreads] = sh_votes[tid+numThreads];
			sh_votes[tid] = vote;
			sh_votes[tid+numThreads] = idx;
		}
		else if ((vote > sh_votes[tid+2*numThreads]) && isMax) {
			sh_votes[tid+2*numThreads] = vote;
			sh_votes[tid+3*numThreads] = idx;
		}

		isMax = true;

		if (idx + blockDim.x < numVotes) {
			a = (idx + blockDim.x) % ANGLE_RES;
			r = (idx + blockDim.x) / ANGLE_RES;
			vote = votes[idx+blockDim.x];

			/* Again, we only want local maximums */
			for (int i = -3; i < 4; i++) {
				for (int j = -aBound; j < aBound + 1; j++) {
					if ((a + j > 0) && (a + j < ANGLE_RES) && (r + i > 0) &&
						(r + i < numVotes / ANGLE_RES)) {
						if (vote < votes[(r+i)*ANGLE_RES+(a+j)]) {
							isMax = false;
						}
					}
				}
			}

			if ((vote >= sh_votes[tid]) && isMax) {
				sh_votes[tid+2*numThreads] = sh_votes[tid];
				sh_votes[tid+3*numThreads] = sh_votes[tid+numThreads];
				sh_votes[tid] = vote;
				sh_votes[tid+numThreads] = idx + blockDim.x;
			}
			else if ((vote > sh_votes[tid+2*numThreads]) && isMax) {
				sh_votes[tid+2*numThreads] = vote;
				sh_votes[tid+3*numThreads] = idx + blockDim.x;
			}
		}

		idx += 2 * blockDim.x * gridDim.x;
	}

	__syncthreads();

	/* Main parallel reduction step. Uses sequential addressing */
	for (i = blockDim.x / 2; i > 0; i >>= 1) {
		if (tid < i) {
			vote = sh_votes[tid+i];
			vote2 = sh_votes[tid+2*numThreads+i];
			curr = sh_votes[tid];
			if (vote >= curr) {
				if (vote2 >= curr) {
					sh_votes[tid+2*numThreads] = vote2;
					sh_votes[tid+3*numThreads] = sh_votes[tid+3*numThreads+i];
				}
				else {
					sh_votes[tid+2*numThreads] = curr;
					sh_votes[tid+3*numThreads] = sh_votes[tid+numThreads];
				}
				sh_votes[tid] = vote;
				sh_votes[tid+numThreads] = sh_votes[tid+numThreads+i];
			}
			else if (vote > sh_votes[tid+2*numThreads]) {
				sh_votes[tid+2*numThreads] = vote;
				sh_votes[tid+3*numThreads] = sh_votes[tid+numThreads+i];
			}
		}

		__syncthreads();
	}

	if (tid == 0) {
		maxVotes[2*blockIdx.x] = sh_votes[0];
		maxVotes[2*blockIdx.x+1] = sh_votes[2*numThreads];
		maxIndex[2*blockIdx.x] = sh_votes[numThreads];
		maxIndex[2*blockIdx.x+1] = sh_votes[3*numThreads];
	}
}

// __global__ void polyHough(int *edges, float *edge_dir, int *votes, int width, 
// 	                      int height) {
// 	int tid, x, y, xv, yv, angle, voteIdx;
// 	float k, dir, theta, cos_t, sin_t, temp;

// 	tid = blockIdx.x * blockDim.x + threadIdx.x;

// 	while (tid < width * height) {
// 		x = tid % width;
// 		y = tid / width;
// 		if (edges[tid] == 255) {
// 			for (angle = 0; angle < 180; angle++) {
// 				theta = (float) angle * PI / 180;
// 				cos_t = cos(theta);
// 				sin_t = sin(theta);
// 				dir = tan(edge_dir[tid] - PI / 2);
// 				k = -sin_t + dir * cos_t;
// 				k /= (2 * (cos_t + dir * sin_t));

// 				for (yv = 0; yv < height; yv++) {
// 					temp = k * (x * cos_t + y * sin_t);
// 					temp += (x * sin_t - y * cos_t);
// 					temp /= (k * sin_t - cos_t);
// 					temp -= yv;
// 					xv = round(temp * (k * sin_t - cos_t) / (k * cos_t + sin_t));

// 					voteIdx = angle * width * height + yv * width + xv;
// 					atomicAdd(&votes[voteIdx], 1);
// 				}
// 			}
// 		}

// 		tid += blockDim.x * gridDim.x;
// 	}
// }

__global__ void polyHough(int *edges, int *votes, int pMax, int width, 
	                      int height) {
	int tid, x, y, xv, yv, p, angle, voteIdx;
	float theta, cos_t, sin_t, temp;

	tid = blockIdx.x * blockDim.x + threadIdx.x;

	while (tid < width * height) {
		x = tid % width;
		y = tid / width;
		if (edges[tid] == 255) {
			for (angle = 0; angle < 180; angle++) {
				theta = (float) angle * PI / 180;
				cos_t = cos(theta);
				sin_t = sin(theta);

				for (yv = 0; yv < height; yv++) {
					for (xv = 0; xv < width; xv++) {
						temp = (y - yv) * sin_t + (x - xv) * cos_t;
						temp *= temp;
						temp /= (y - yv) * cos_t - (x - xv) * sin_t;
						p = round(temp);
						if (p < 0) {
							p *= -1;
							voteIdx = (angle + 180) * width * height * pMax;
						}
						else {
							voteIdx = angle * width * height * pMax;
						}
						voteIdx += (yv * width * pMax + xv * pMax + p);

						if ((voteIdx < ANGLE_RES * width * height * pMax) && (voteIdx > 0)) {
							atomicAdd(&votes[voteIdx], 1);
						}
					}
				}
			}
		}

		tid += blockDim.x * gridDim.x;
	}
}

__global__ void fastHough(int *edge_coord, int *votes, int rhoSize, int width,
	                      int height) {
	int gid, tid, a, coord, x, y, vote_index;
    float r;

    extern __shared__ int sh_line[];

    gid = blockIdx.x * blockDim.x + threadIdx.x;
    tid = threadIdx.x;

    while (tid < rhoSize) {
    	sh_line[tid] = 0;
    	tid += blockDim.x;
    }

    while (gid < blockDim.x * 180) {
    	tid = threadIdx.x;
    	while (tid < rhoSize) {
    		a = gid / blockDim.x;
    		coord = edge_coord[tid];
    		x = coord % width;
			y = coord / width;
    		r = (x - (float) width / 2) * cos(a * PI / 180) +
    			(y - (float) height / 2) * sin(a * PI / 180);
    		vote_index = round(r + ((float) rhoSize / 2));
    		atomicAdd(&sh_line[vote_index], 1);

    		tid += blockDim.x;
    	}
    	tid = threadIdx.x;
    	while (tid < rhoSize) {
    		votes[a*rhoSize+tid] = sh_line[tid];
    		sh_line[tid] = 0;
    		tid += blockDim.x;
    	}

    	gid += blockDim.x * gridDim.x;
    }
}


void callGetEdges(int *edges, int *edge_coord, int *length, int width,
	              int height, int threadsPerBlock, int numBlocks) {
	int sh_size = threadsPerBlock * sizeof(int);
	getEdges<<<numBlocks, threadsPerBlock, sh_size>>>(edges, edge_coord, length,
		       width, height);
}

void callLinearHough(int *edges, int *votes, int numVotes, int width,
	                 int height, int threadsPerBlock, int numBlocks) {
	linearHough<<<numBlocks, threadsPerBlock>>>(edges, votes, numVotes,
		          width, height);
}

void callFastHough(int *edge_coord, int *votes, int rhoSize, int width,
	               int height, int threadsPerBlock, int numBlocks) {
	int sh_size = rhoSize * sizeof(int);
	fastHough<<<numBlocks, threadsPerBlock, sh_size>>>(edge_coord, votes,
		        rhoSize, width, height);
}

void callPolyHough(int *edges, float *edge_dir, int *votes, int pMax, int width,
	               int height, int threadsPerBlock, int numBlocks) {
	polyHough<<<numBlocks, threadsPerBlock>>>(edges, votes, pMax, 
		        width, height);
}

void callGetMax(int *votes, int *maxIndex, int *maxVotes, int numVotes,
	            int threadsPerBlock, int numBlocks) {
	int sh_size = 4 * threadsPerBlock * sizeof(int);
	getMax<<<numBlocks, threadsPerBlock, sh_size>>>(votes, maxIndex, maxVotes,
		     numVotes);
}

/* Does a naive hough transform on the CPU for comparison. Finds the two
 * longest lines, but does not suppress "same" lines. This is used as a run time
 * comparison. */
int cpuHoughLinear(int *edges, int *votes, int numVotes, int width,
	               int height) {
	int x, y, a, i, vote_index, maxVote = 0, max2Vote = 0, maxIndex, max2Index;
	float r, theta;

	for (y = 0; y < height; y++) {
		for (x = 0; x < width; x++) {
			if (edges[y*width+x] == 255) {
    			for (a = 0; a < ANGLE_RES; a++) {
    				theta = (float) a * PI / ANGLE_RES;
    				r = x * cos(theta) + y * sin(theta);
    				vote_index = round(r + ((float) numVotes / 2)) *
    				             ANGLE_RES + a;
    				votes[vote_index]++;
    			}
    		}
		}
	}

	for (i = 0; i < numVotes * ANGLE_RES; i++) {
		if (votes[i] >= maxVote) {
			max2Vote = maxVote;
			max2Index = maxIndex;
			maxVote = votes[i];
			maxIndex = i;
		}
		else if (votes[i] > max2Vote) {
			max2Vote = votes[i];
			max2Index = i;
		}
	}

	return maxIndex + max2Index * numVotes * ANGLE_RES;
}

/* Runs the hough transform on both the CPU and GPU. It finds the two longest
 * lines in the image and compares them to see if they are close to each other.
 * this is because some pictures have both a shore and a horizon, which both
 * show up at long straight lines. We want to get the water/land line.
 */
int houghTransform(int *dev_edges, float *dev_dir, int width, int height) {

	int vote_size = 2 * ceil(sqrt(height * height + width * width)) * ANGLE_RES;
	int cpu_max_index, cpu_max2_index;
	int gpu_max_index, gpu_max2_index, gpu_max_vote = 0, gpu_max2_vote = 0;
    
    int *votes, *edges, *max_vote, *max_index;
    int *dev_votes, *dev_max_vote, *dev_max_index;

    float gpu_timer, cpu_timer;
    int threadsPerBlock = 1024;
    int numBlocks = ceil((float) (width * height) / threadsPerBlock);

    max_vote = new int[2*numBlocks];
    max_index = new int[2*numBlocks];
    votes = new int[vote_size];
    edges = new int[width*height];

    gpuErrChk(cudaMalloc((void **) &dev_votes, vote_size * sizeof(int)));
    gpuErrChk(cudaMalloc((void **) &dev_max_vote, 2 * numBlocks * sizeof(int)));
    gpuErrChk(cudaMalloc((void **) &dev_max_index,
    	                2 * numBlocks * sizeof(int)));

    memset((void *) votes, 0, vote_size * sizeof(int));
    gpuErrChk(cudaMemset((void *) dev_votes, 0, vote_size * sizeof(int)));

    START_TIMER();
    gpuErrChk(cudaMemcpy(edges, dev_edges, width * height * sizeof(int),
    	                cudaMemcpyDeviceToHost));
    cpu_max2_index = cpuHoughLinear(edges, votes, vote_size / ANGLE_RES, width,
    	                            height);
    cpu_max_index = cpu_max2_index % vote_size;
    cpu_max2_index = cpu_max2_index / vote_size;
    STOP_RECORD_TIMER(cpu_timer);


    START_TIMER();
    
    callLinearHough(dev_edges, dev_votes, vote_size / ANGLE_RES, width, height,
    	            threadsPerBlock, numBlocks);
    callGetMax(dev_votes, dev_max_index, dev_max_vote, vote_size,
	           threadsPerBlock, numBlocks);
    gpuErrChk(cudaMemcpy(max_vote, dev_max_vote, 2 * numBlocks * sizeof(int),
    	                cudaMemcpyDeviceToHost));
    gpuErrChk(cudaMemcpy(max_index, dev_max_index, 2 * numBlocks * sizeof(int),
    	                cudaMemcpyDeviceToHost));

    for (int i = 0; i < 2 * numBlocks; i++) {
    	if (max_vote[i] >= gpu_max_vote) {
    		gpu_max2_vote = gpu_max_vote;
    		gpu_max2_index = gpu_max_index;
    		gpu_max_vote = max_vote[i];
    		gpu_max_index = max_index[i];
    	}
    	else if (max_vote[i] > gpu_max2_vote){
    		gpu_max2_vote = max_vote[i];
    		gpu_max2_index = max_index[i];
    	}
    }

    STOP_RECORD_TIMER(gpu_timer);
    //assert(gpu_max_index == cpu_max_index);
    //assert(gpu_max2_index == cpu_max2_index);

    // fast_vote = new int[numBlocks];
    // fast_index = new int[numBlocks];
    // START_TIMER();
    // callGetEdges(dev_edges, dev_edge_coord, dev_length, width, height,
    // 	         threadsPerBlock, numBlocks);
    // callFastHough(dev_edge_coord, dev_fast_votes, vote_size / 180, width,
	   //            height, threadsPerBlock, numBlocks);
    // callGetMax(dev_fast_votes, dev_fast_index, dev_fast_vote, vote_size,
	   //         threadsPerBlock, numBlocks);
    // cudaMemcpy(fast_vote, dev_fast_vote, numBlocks * sizeof(int),
    // 	       cudaMemcpyDeviceToHost);
    // cudaMemcpy(fast_index, dev_fast_index, numBlocks * sizeof(int),
    // 	       cudaMemcpyDeviceToHost);

    // for (int i = 0; i < numBlocks; i++) {
    // 	if (fast_vote[i] > max_fast_vote) {
    // 		max_fast_vote = fast_vote[i];
    // 		max_fast_index = fast_index[i];
    // 	}
    // }

    // STOP_RECORD_TIMER(fast_timer);
    // assert(gpu_max_index == max_fast_index);

    // cudaMemset((void *) dev_poly_votes, 0, poly_vote_size * sizeof(int));
    // poly_vote = new int[numBlocks];
    // poly_index = new int[numBlocks];
    // START_TIMER();
    // callPolyHough(dev_edges, dev_dir, dev_poly_votes, pMax, width, height,
    // 	          threadsPerBlock, numBlocks);
    // numBlocks = ceil((float) poly_vote_size / threadsPerBlock);
    // callGetMax(dev_poly_votes, dev_poly_index, dev_poly_vote, poly_vote_size,
    // 	       threadsPerBlock, numBlocks);
    // cudaMemcpy(poly_vote, dev_poly_vote, numBlocks * sizeof(int),
    // 	       cudaMemcpyDeviceToHost);
    // cudaMemcpy(poly_index, dev_poly_index, numBlocks * sizeof(int),
    // 	       cudaMemcpyDeviceToHost);

    // for (int i = 0; i < numBlocks; i++) {
    // 	if (poly_vote[i] > max_poly_vote) {
    // 		max_poly_vote = poly_vote[i];
    // 		max_poly_index = poly_index[i];
    // 	}
    // }

    // STOP_RECORD_TIMER(poly_timer);

    // ofstream gpu;
    // ofstream cpu;

    // gpu.open("houghGPU.txt", ios::app);
    // gpu << gpu_timer << endl;
    // cpu.open("houghCPU.txt", ios::app);
    // cpu << cpu_timer << endl;

    // gpu.close();
    // cpu.close();

    cout << "CPU Hough Transform: " << cpu_timer << " ms" << endl;
    cout << "GPU Hough Transform: " << gpu_timer << " ms" << endl;
    // cout << "GPU Fast Hough Transform: " << fast_timer << " ms" << endl;
    // cout << "GPU Polynomial Hough Transform " << poly_timer << " ms" << endl;

    /* Of the two longest lines, we want to see if they are close to each other,
     * which means that one is the shore, and the other is the horizon. We
     * choose the shore line, which is always below the horizon line */
    int angle1, r1, angle2, r2, yl1, yr1, yl2, yr2, diffl, diffr, rho_size;
    float theta1, theta2;
    rho_size = ceil(sqrt(height * height + width * width));
    angle1 = gpu_max_index % ANGLE_RES;
    theta1 = angle1 * PI / ANGLE_RES;
    r1 = (gpu_max_index / ANGLE_RES) - rho_size;
    yl1 = round(r1 / sin(theta1));
    yr1 = round((r1 - (width - 1) * cos(theta1)) / sin(theta1));
    angle2 = gpu_max2_index % ANGLE_RES;
    theta2 = angle2 * PI / ANGLE_RES;
    r2 = (gpu_max2_index / ANGLE_RES) - rho_size;
    yl2 = round(r2 / sin(theta2));
    yr2 = round((r2 - (width - 1) * cos(theta2)) / sin(theta2));

    diffl = yl2 - yl1;
    diffr = yr2 - yr1;
    /* If the second line has endpoints less than 50 pixels below the first,
     * then it is the shore line that we want. */
    if ((diffl >= -3) && (diffl < 50) && (diffr >= -3) && (diffr < 50)) {
    	gpu_max_index = gpu_max2_index;
    }
    cout << "longest line is (0, " << yl1 << "), (1501, " << yr1 << ")"<< endl;
    cout << "with angle " << angle1 << " and rho " << r1 << endl;
    cout << "second long line is (0, " << yl2 << "), (1501, " << yr2 << ")"<< endl;
    cout << "with angle " << angle2 << " and rho " << r2 << endl;

    // int a, x, y, p, remain;
    // a = max_poly_index / (width * height * pMax);
    // remain = max_poly_index % (width * height * pMax);
    // y = remain / (width * pMax);
    // remain = remain % (width * pMax);
    // x = remain / pMax;
    // p = remain % pMax;

    // cout << p << "(y - " << y << ")cos(" << a << ") - (x - ";
    // cout << x << ")sin(" << a << ") = ((y - " << y << ")cos(" << a << ")";
    // cout << " + (x - " << x << ")cos(" << a << "))^2" << endl;

    gpuErrChk(cudaFree(dev_votes));
    gpuErrChk(cudaFree(dev_max_vote));
    gpuErrChk(cudaFree(dev_max_index));
    free(votes);
    free(edges);
    free(max_vote);
    free(max_index);

    return gpu_max_index;
}


int main(int argc, const char* argv[]) {
    if (argc != 4) {
        fprintf(stderr, "usage: ./houghTransform input width height");
    }

    ifstream r;
    ofstream app;
    string color, output;
    int width = atoi(argv[2]), height = atoi(argv[3]), count = 0, idx;
    int radius, angle, x1, x2;
    int *image = new int[width*height], *edges;
    float theta, m, y1, y2, *dir;

    gpuErrChk(cudaMalloc((void **) &dir, width * height * sizeof(float)));
    gpuErrChk(cudaMalloc((void **) &edges, width * height * sizeof(int)));

    r.open(argv[1]);

    /* Load the image */
    if (r.is_open()) {
        while (getline(r, color)) {
            image[count] = atoi(color.c_str());
            count++;
        }
    }

    /* Run the Canny edge detector on the image to get a binary edge image */
    cannyEdge(image, edges, dir, width, height);
    
    idx = houghTransform(edges, dir, width, height);

    angle = idx % ANGLE_RES;
    theta = angle * PI / ANGLE_RES;
    radius = (idx / ANGLE_RES) - ceil(sqrt(height * height + width * width));
    x1 = 0;
    x2 = width - 1;
    y1 = (radius - x1 * cos(theta)) / sin(theta);
    y2 = (radius - x2 * cos(theta)) / sin(theta);
    m = (y2 - y1) / (width - 1);

    cout << "First point is (" << x1 << ", " << round(y1) << ")" << endl;
    cout << "Second point is (" << x2 << ", " << round(y2) << ")" << endl;

    /* Append the angle and y intercept to a file. Used to compare it to the
     * actual horizon line. */
    app.open("houghLines.txt");
    app << m << " " << y1 << endl;

    r.close();
    app.close();

    free(image);
    gpuErrChk(cudaFree(edges));
	gpuErrChk(cudaFree(dir));

    return 0;
}
