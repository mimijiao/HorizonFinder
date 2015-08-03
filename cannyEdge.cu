#include <iostream>
#include <cstdlib>
#include <fstream>
#include <string>
#include <cassert>
#include <stack>
#include <cuda_runtime.h>
#include "cannyEdge.cuh"

#define PI 3.14159265

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
cudaEvent_t start;
cudaEvent_t stop;

#define START_TIMER() {                         \
      gpuErrChk(cudaEventCreate(&start));       \
      gpuErrChk(cudaEventCreate(&stop));        \
      gpuErrChk(cudaEventRecord(start));        \
    }

#define STOP_RECORD_TIMER(name) {                           \
      gpuErrChk(cudaEventRecord(stop));                     \
      gpuErrChk(cudaEventSynchronize(stop));                \
      gpuErrChk(cudaEventElapsedTime(&name, start, stop));  \
      gpuErrChk(cudaEventDestroy(start));                   \
      gpuErrChk(cudaEventDestroy(stop));                    \
  }

  texture<int, 2, cudaReadModeElementType> img_tex;

/* does a gaussian blur for each pixel in a 5x5 grid around the pixel. */
__global__ void gaussianBlur(float *out_image, int width, int height) {
    int gid, row, col;
    float blur_color;

    gid = blockIdx.x * blockDim.x + threadIdx.x;

    while (gid < width * height) {
        row = gid / width;
        col = gid % width;
        blur_color = 0;

        /* Could be done with for loop, but unrolling for loops leads to
         * faster compiler optimizations.
         * Convolutions are time consuming since it is loading many values from
         * global memory and a lot of them overlap with values other threads
         * want. It can be sped up by loading the values to shared memory, but
         * texture memory provides a significant speed up as well. It also
         * simplifies the code b/c out of bound values are clamped. */

        /* row 1 of gaussian filter */
        blur_color += tex2D(img_tex, col - 2, row - 2) * 2;
        blur_color += tex2D(img_tex, col - 1, row - 2) * 4;
        blur_color += tex2D(img_tex, col, row - 2) * 5;
        blur_color += tex2D(img_tex, col + 1, row - 2) * 4;
        blur_color += tex2D(img_tex, col + 2, row - 2) * 2;

        /* row 2 of gaussian filter */
        blur_color += tex2D(img_tex, col - 2, row - 1) * 4;
        blur_color += tex2D(img_tex, col - 1, row - 1) * 9;
        blur_color += tex2D(img_tex, col, row - 1) * 12;
        blur_color += tex2D(img_tex, col + 1, row - 1) * 9;
        blur_color += tex2D(img_tex, col + 2, row - 1) * 4;

        /* row 3 of gaussian filter */
        blur_color += tex2D(img_tex, col - 2, row) * 5;
        blur_color += tex2D(img_tex, col - 1, row) * 12;
        blur_color += tex2D(img_tex, col, row) * 15;
        blur_color += tex2D(img_tex, col + 1, row) * 12;
        blur_color += tex2D(img_tex, col + 2, row) * 5;

        /* row 4 of gaussian filter */
        blur_color += tex2D(img_tex, col - 2, row + 1) * 4;
        blur_color += tex2D(img_tex, col - 1, row + 1) * 9;
        blur_color += tex2D(img_tex, col, row + 1) * 12;
        blur_color += tex2D(img_tex, col + 1, row + 1) * 9;
        blur_color += tex2D(img_tex, col + 2, row + 1) * 4;

        /* row 5 of gaussian filter */
        blur_color += tex2D(img_tex, col - 2, row + 2) * 2;
        blur_color += tex2D(img_tex, col - 1, row + 2) * 4;
        blur_color += tex2D(img_tex, col, row + 2) * 5;
        blur_color += tex2D(img_tex, col + 1, row + 2) * 4;
        blur_color += tex2D(img_tex, col + 2, row + 2) * 2;

        out_image[gid] = blur_color / 159;

        gid += blockDim.x * gridDim.x;
    }
}

/* Does Sobel edge detection. Calculates the gradient and the direction in which
 * the color is changing the most. Rounds the direction to the nearest 45 degree
 * angle. */
__global__ void sobelMask(float *image, int *edge_dir, float *dir, 
                          float *gradient, int width, int height) {
    int gid, row, col, pixel_idx;
    float angle, grad_x, grad_y;

    gid = blockIdx.x * blockDim.x + threadIdx.x;

    while (gid < width * height) {
        row = gid / width;
        col = gid % width;
        grad_x = 0;
        grad_y = 0;

        if ((row == 0) || (col == 0) || (row == height - 1) ||
            (col == width - 1)) {
            gradient[gid] = round(4 * 255 * sqrt(2.0));
            edge_dir[gid] = 0;
        }
        else {
            /* does convolution by taking mean of the sobel detector dot
            * multiplied with the 3x3 square around a pixel. */
            pixel_idx = (row - 1) * width + (col - 1);
            grad_x -= image[pixel_idx];
            grad_y += image[pixel_idx];

            pixel_idx = (row - 1) * width + col;
            grad_y += 2 * image[pixel_idx];

            pixel_idx = (row - 1) * width + (col + 1);
            grad_x += image[pixel_idx];
            grad_y += image[pixel_idx];

            pixel_idx = row * width + (col - 1);
            grad_x -= 2 * image[pixel_idx];

            pixel_idx = row * width + (col + 1);
            grad_x += 2 * image[pixel_idx];

            pixel_idx = (row + 1) * width + (col - 1);
            grad_x -= image[pixel_idx];
            grad_y -= image[pixel_idx];

            pixel_idx = (row + 1) * width + col;
            grad_y -= 2 * image[pixel_idx];

            pixel_idx = (row + 1) * width + (col + 1);
            grad_x += image[pixel_idx];
            grad_y -= image[pixel_idx];

            /* Calculate gradient */
            gradient[gid] = sqrt(grad_x * grad_x + grad_y * grad_y);
            angle = atan2(grad_y, grad_x);
            dir[gid] = angle;
            angle = angle * 180 / PI;

            /* Round angles to nearest 45 degrees */
            if (((angle <= 22.5) && (angle > -22.5)) || ((angle > 157.5)
                  && (angle <= -157.5))) {
                edge_dir[gid] = 0;
            }
            else if (((angle <= 67.5) && (angle > 22.5)) || ((angle > -157.5)
                       && (angle <= -112.5))) {
                edge_dir[gid] = 45;
            }
            else if (((angle <= 112.5) && (angle > 67.5)) || ((angle > -112.5)
                       && (angle <= -67.5))) {
                edge_dir[gid] = 90;
            }
            else {
                edge_dir[gid] = 135;
            }
        }

        gid += blockDim.x * gridDim.x;
    }
}

/* Thin the edges by deleting edges that do not have the max gradient of
 * the edge it is a part of. It looks at the gradients of the pixels to
 * either side of a pixel, depending on the direction of the gradient,
 * and suppresses the pixel if its gradient is not the max in that direction.
 *
 * Also does double thresholding. The color is set to 255 if it is a strong
 * edge (above the high threshold) and 127 if it's between the two thresholds.
 * anything below the low threshold is suppressed. */
__global__ void suppressNonMax(int *edge_dir, float *gradient, int *edges,
                            int *visit, float hi_thresh, float low_thresh,
                            int width, int height) {
    int gid, row, col, x_off, y_off, dir;
    float grad, neighbor1, neighbor2;

    gid = blockIdx.x * blockDim.x + threadIdx.x;

    while (gid < width * height) {
        row = gid / width;
        col = gid % width;
        dir = edge_dir[gid];
        grad = gradient[gid];
        neighbor1 = 1e6;
        neighbor2 = 1e6;

        /* check if the pixel is on an edge */
        if (grad > low_thresh) {
            if (dir == 0) {
                x_off = 1;
                y_off = 0;
            }
            else if (dir == 45) {
                x_off = 1;
                y_off = -1;
            }
            else if (dir == 135) {
                x_off = 1;
                y_off = 1;
            }
            else {
                x_off = 0;
                y_off = 1;
            }
            /* get the two pixels on either side of the pixel, according
             * to the direction of the gradient. */
            if ((row + y_off < height) && (row + y_off >= 0)
                 && (col + x_off < width)) {
                neighbor1 = gradient[(row+y_off)*width+(col+x_off)];
            }
            if ((row - y_off >= 0) && (row - y_off < height)
                 && (col - x_off >= 0)) {
                neighbor2 = gradient[(row-y_off)*width+(col-x_off)];
            }

            /* if it's a local maximum, it can be an edge. */
            if ((grad > neighbor1) && (grad > neighbor2)) {
                /* determine if the edge is a strong edge or a weak edge */
                if (grad < hi_thresh) {
                    edges[gid] = 127;
                }
                else {
                    edges[gid] = 255;
                    visit[gid] = 1;
                }
            }
            else {
                edges[gid] = 0;
            }
        }
        /* suppress pixel if not on an edge */
        else {
            edges[gid] = 0;
        }

        gid += blockDim.x * gridDim.x;
    }
}

/* Determines which weak edges should be kept. If a weak
 * edge is connected to a strong edge through other edges, it is kept.
 * Otherwise, it is suppressed. */
__global__ void hysteresisTracing(int *edges, int *visit, bool *modified,
                                  bool done, int width, int height) {
    int gid, row, col, idx;
    bool t_modified = false;
    gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (done) {
        while (gid < width * height) {
            if (edges[gid] == 127) {
                edges[gid] = 0;
            }

            gid += blockDim.x * gridDim.x;
        }
    }
    else {
        while (gid < width * height) {
            row = gid / width;
            col = gid % width;

            if (visit[gid] == 1) {
                visit[gid] = 0;

                /* check 8 pixels around each pixel for a weak edge */
                /* upper left */
                idx = (row-1)*width+(col-1);
                if ((col > 0) && (row > 0) && (edges[idx] == 127)) {
                    edges[idx] = 255;
                    visit[idx] = 1;
                    t_modified = true;
                }
                /* above */
                idx = (row-1)*width+col;
                if ((row > 0) && (edges[idx] == 127)) {
                    edges[idx] = 255;
                    visit[idx] = 1;
                    t_modified = true;
                }
                /* upper right */
                idx = (row-1)*width+(col+1);
                if ((row > 0) && (col < width - 1) && (edges[idx] == 127)) {
                    edges[idx] = 255;
                    visit[idx] = 1;
                }
                /* right */
                idx = row*width+(col+1);
                if ((col < width - 1) && (edges[idx] == 127)) {
                    edges[idx] = 255;
                    visit[idx] = 1;
                    t_modified = true;
                }
                /* lower right */
                idx = (row+1)*width+(col+1);
                if ((col < width - 1) && (row < height - 1) &&
                    (edges[idx] == 127)) {
                    edges[idx] = 255;
                    visit[idx] = 1;
                    t_modified = true;
                }
                /* below */
                idx = (row+1)*width+col;
                if ((row < height - 1) && (edges[idx] == 127)) {
                    edges[idx] = 255;
                    visit[idx] = 1;
                    t_modified = true;
                }
                /* lower left */
                idx = (row+1)*width+(col-1);
                if ((col > 0) && (row < height - 1) &&
                    (edges[idx] == 127)) {
                    edges[idx] = 255;
                    visit[idx] = 1;
                    t_modified = true;
                }
                /* left */
                idx = row*width+(col-1);
                if ((col > 0) && (edges[idx] == 127)) {
                    edges[idx] = 255;
                    visit[idx] = 1;
                    t_modified = true;
                }
            }
            gid += blockDim.x * gridDim.x;
        }

        /* Don't need to worry about atomics. We don't care which thread
         * sets this value, only that it gets set if there are more more edges
         * to visit */
        if (t_modified) {
            *modified = true;
        }
    }
}

/* Fills in the gradient histogram bins by getting the gradient at each pixel
 * and incrementing the corresponding bin */
__global__ void getCounts(float *gradient, int *counts, int width, int height) {
    int gid, tid, grad, sh_size;

    sh_size = ceil(4 * 255 * sqrt(2.0)); /* max gradient value */

    extern __shared__ int sh_counts[];

    gid = blockIdx.x * blockDim.x + threadIdx.x;
    tid = threadIdx.x;

    while (tid < sh_size) {
        sh_counts[tid] = 0;
        tid += blockDim.x;
    }
    __syncthreads();

    while (gid < width * height) {
        grad = round(gradient[gid]);

        atomicAdd(&sh_counts[grad], 1);

        gid += blockDim.x * gridDim.x;
    }

    __syncthreads();
    tid = threadIdx.x;

    while (tid < sh_size) {
        if (sh_counts[tid] != 0) {
            atomicAdd(&counts[tid], sh_counts[tid]);
        }
        tid += blockDim.x;
    }
}


void callGaussianBlur(float *blur_image, int width, int height,
                      int threadsPerBlock, int numBlocks) {
    gaussianBlur<<<numBlocks, threadsPerBlock>>>(blur_image, width, height);
}

void callSobelMask(float *image, int *edge_dir, float *dir, float *gradient,
                   int width, int height, int threadsPerBlock, int numBlocks) {
    sobelMask<<<numBlocks, threadsPerBlock>>>(image, edge_dir, dir, gradient,
                width, height);
}

void callSuppressNonMax(int *edge_dir, float *gradient, int *edges, int *visit,
                        float hi_thresh, float low_thresh, int width,
                        int height, int threadsPerBlock, int numBlocks) {
    suppressNonMax<<<numBlocks, threadsPerBlock>>>(edge_dir, gradient, edges,
                     visit, hi_thresh, low_thresh, width, height);
}

void callHysteresisTracing(int *edges, int *visit, bool *modified, bool done,
                int width, int height, int threadsPerBlock, int numBlocks) {
    hysteresisTracing<<<numBlocks, threadsPerBlock>>>(edges, visit, modified, 
                        done, width, height);
}

void callGetCounts(float *gradient, int *counts, int width, int height,
                   int threadsPerBlock, int numBlocks) {
    int sh_size = ceil(4 * 255 * sqrt(2.0)) * sizeof(int);
    getCounts<<<numBlocks, threadsPerBlock, sh_size>>>(gradient, counts, width,
                height);
}

/* Calculates the double threshold by letting a certain percentage of pixels
 * be edge pixels. */
int getThreshold(int width, int height, float *gradient) {
    int sum, numBlocks, threadsPerBlock, bin, grad_size;
    int *counts, *dev_counts;

    grad_size = ceil(4 * 255 * sqrt(2));
    threadsPerBlock = 512;
    numBlocks = ceil((float) width * height / threadsPerBlock);

    counts = new int[grad_size];
    gpuErrChk(cudaMalloc((void **) &dev_counts, grad_size * sizeof(int)));
    gpuErrChk(cudaMemset(dev_counts, 0, grad_size * sizeof(int)));

    callGetCounts(gradient, dev_counts, width, height, threadsPerBlock,
                  numBlocks);
    gpuErrChk(cudaMemcpy(counts, dev_counts, grad_size * sizeof(int),
                        cudaMemcpyDeviceToHost));

    sum = 0;
    bin = -1;
    while ((sum < 0.95 * width * height) && (bin < grad_size)) {
        bin++;
        sum += counts[bin];
    }

    return bin;
}

void cannyCPU(int *image, int *edges, int width, int height) {
    int sum, bin, dir, x_off, y_off, idx, row, col, neighbor, grad_size;
    float hi_thresh, low_thresh, blur_color, grad_x, grad_y, angle, grad;
    float neighbor1, neighbor2;
    int *edge_dir = new int[width*height];
    int *counts;
    float *blur = new float[width*height], *gradient = new float[width*height];
    bool *visited;
    stack<int> visit;

    int gauss[5][5] = {{2, 4, 5, 4, 2},
                       {4, 9, 12, 9, 4},
                       {5, 12, 15, 12, 5},
                       {4, 9, 12, 9, 4},
                       {2, 4, 5, 4, 2}};
    int sobel_x[3][3] = {{-1, 0, 1},
                         {-2, 0, 2},
                         {-1, 0, 1}};
    int sobel_y[3][3] = {{1, 2, 1},
                         {0, 0, 0},
                         {-1, -2, -1}};

    // Gaussian Blur
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            blur_color = 0;
            for (int r = -2; r < 3; r++) {
                for (int c = -2; c < 3; c++) {
                    row = i + r;
                    col = j + c;
                    if (row < 0)  {
                        row = 0;
                    }
                    else if (row >= height) {
                        row = height - 1;
                    }
                    if (col < 0) {
                        col = 0;
                    }
                    else if (col >= width) {
                        col = width - 1;
                    }

                    blur_color += image[row*width+col] * gauss[r+2][c+2];
                }
            }
            blur[i*width+j] = blur_color / 159;
        }
    }

    // Sobel Edge
    grad_size = round(4 * 255 * sqrt(2.0));
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            grad_x = 0;
            grad_y = 0;

            if ((i == 0) || (j == 0) || (i == height - 1) ||
                (j == width - 1)) {
                gradient[i*width+j] = grad_size;
                edge_dir[i*width+j] = 0;
            }
            else {
                for (int r = -1; r < 2; r++) {
                    for (int c = -1; c < 2; c++) {
                        grad_x += blur[(i+r)*width+(j+c)] * sobel_x[r+1][c+1];
                        grad_y += blur[(i+r)*width+(j+c)] * sobel_y[r+1][c+1];
                    }
                }

                /* Calculate gradient */
                gradient[i*width+j] = sqrt(grad_x * grad_x + grad_y * grad_y);
                angle = atan2(grad_y, grad_x);
                angle = angle * 180 / PI;

                /* Round angles to nearest 45 degrees */
                if (((angle <= 22.5) && (angle > -22.5)) || ((angle > 157.5)
                    && (angle <= -157.5))) {
                    edge_dir[i*width+j] = 0;
                }
                else if (((angle <= 67.5) && (angle > 22.5)) || ((angle > -157.5)
                           && (angle <= -112.5))) {
                    edge_dir[i*width+j] = 45;
                }
                else if (((angle <= 112.5) && (angle > 67.5)) || ((angle > -112.5)
                           && (angle <= -67.5))) {
                    edge_dir[i*width+j] = 90;
                }
                else {
                    edge_dir[i*width+j] = 135;
                }
            }
        }
    }

    // Threshold
    counts = (int *) calloc(grad_size + 1, sizeof(int));

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            idx = round(gradient[i*width+j]);
            counts[idx]++;
        }
    }

    sum = 0;
    bin = -1;
    while ((sum < 0.95 * width * height) && (bin < grad_size)) {
        bin++;
        sum += counts[bin];
    }
    hi_thresh = bin;
    low_thresh = hi_thresh * 0.4;

    // Non Maximum Suppression
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            dir = edge_dir[i*width+j];
            grad = gradient[i*width+j];
            neighbor1 = 1e6;
            neighbor2 = 1e6;

            /* check if the pixel is on an edge */
            if (grad > low_thresh) {
                if (dir == 0) {
                    x_off = 1;
                    y_off = 0;
                }
                else if (dir == 45) {
                    x_off = 1;
                    y_off = -1;
                }
                else if (dir == 135) {
                    x_off = 1;
                    y_off = 1;
                }
                else {
                    x_off = 0;
                    y_off = 1;
                }
                /* get the two pixels on either side of the pixel, according
                 * to the direction of the gradient. */
                if ((i + y_off < height) && (i + y_off >= 0)
                     && (j + x_off < width)) {
                    neighbor1 = gradient[(i+y_off)*width+(j+x_off)];
                }
                if ((i - y_off >= 0) && (i - y_off < height)
                     && (j - x_off >= 0)) {
                    neighbor2 = gradient[(i-y_off)*width+(j-x_off)];
                }

                /* if it's a local maximum, it can be an edge. */
                if ((grad > neighbor1) && (grad > neighbor2)) {
                    /* determine if the edge is a strong edge or a weak edge */
                    if (grad < hi_thresh) {
                        edges[i*width+j] = 127;
                    }
                    else {
                        edges[i*width+j] = 255;
                        visit.push(i * width + j);
                    }
                }
                else {
                    edges[i*width+j] = 0;
                }
            }
            /* suppress pixel if not on an edge */
            else {
                edges[i*width+j] = 0;
            }
        }
    }  

    // Hysteresis
    visited = (bool *) calloc(width * height, sizeof(bool));
    while (!visit.empty()) {
        idx = visit.top();
        visit.pop();
        row = idx / width;
        col = idx % width;

        for (int r = -1; r < 2; r++) {
            for (int c = -1; c < 2; c++) {
                if ((row + r > 0) && (row + r < height) && (col + c > 0) &&
                    (col + c < width)) {
                    neighbor = (row + r) * width + (col + c);
                    if ((!visited[neighbor]) && (edges[neighbor] == 127)) {
                        edges[neighbor] = 255;
                        visit.push(neighbor);
                    }
                }
            }
        }
    }

    free(edge_dir);
    free(counts);
    free(blur);
    free(gradient);
    free(visited);
}

void cannyEdge(int *image, int *dev_edges, float *dev_dir, int width,
               int height) {
    float cpu_timer, gpu_timer;
    int *dev_edge_dir, *dev_visit;
    int *edges = new int[width*height];
    float *dev_blur, *dev_gradient;
    bool modified, *dev_modified;
    cudaArray* dev_image;
    cudaChannelFormatDesc img_channel;

    // CPU Canny Edge
    START_TIMER();
    cannyCPU(image, edges, width, height);
    STOP_RECORD_TIMER(cpu_timer);


    // GPU Canny Edge

    START_TIMER();

    int threadsPerBlock = 512;
    int numBlocks = ceil((float) width * height / threadsPerBlock);
    float hi_thresh, low_thresh;

    gpuErrChk(cudaMalloc((void **) &dev_blur,
                        width * height * sizeof(float)));
    gpuErrChk(cudaMalloc((void **) &dev_edge_dir,
                        width * height * sizeof(int)));
    gpuErrChk(cudaMalloc((void **) &dev_gradient,
                        width * height * sizeof(float)));
    gpuErrChk(cudaMalloc((void **) &dev_visit, width * height * sizeof(int)));
    gpuErrChk(cudaMalloc((void **) &dev_modified, sizeof(bool)));

    gpuErrChk(cudaMemset(dev_visit, 0, width * height * sizeof(int)));

    /* Set up texture memory for image, used in Gaussian blur */
    img_channel = cudaCreateChannelDesc<int>();
    gpuErrChk(cudaMallocArray(&dev_image, &img_channel, width, height));
    gpuErrChk(cudaMemcpyToArray(dev_image, 0, 0, image,
              width * height * sizeof(int), cudaMemcpyHostToDevice));
    img_tex.filterMode = cudaFilterModePoint;
    img_tex.addressMode[0] = cudaAddressModeClamp;
    img_tex.addressMode[1] = cudaAddressModeClamp;
    gpuErrChk(cudaBindTextureToArray(img_tex, dev_image));

    /* Do the canny edge detection */
    callGaussianBlur(dev_blur, width, height, threadsPerBlock, numBlocks);
    callSobelMask(dev_blur, dev_edge_dir, dev_dir, dev_gradient, width, height,
                  threadsPerBlock, numBlocks);

    /* get the thresholds using Matlab's method */
    hi_thresh = getThreshold(width, height, dev_gradient);
    low_thresh = hi_thresh * 0.4;

    cout << "thresholds " << hi_thresh << " " << low_thresh << endl;

    callSuppressNonMax(dev_edge_dir, dev_gradient, dev_edges, dev_visit,
            hi_thresh, low_thresh, width, height, threadsPerBlock, numBlocks);

    /* BFS from every strong edge pixel */
    modified = true;
    while (modified) {
        gpuErrChk(cudaMemset(dev_modified, 0, sizeof(bool)));
        callHysteresisTracing(dev_edges, dev_visit, dev_modified, false, width,
                              height, threadsPerBlock, numBlocks);
        gpuErrChk(cudaMemcpy(&modified, dev_modified, sizeof(bool),
                            cudaMemcpyDeviceToHost));
    }
    callHysteresisTracing(dev_edges, dev_visit, dev_modified, true, width,
                          height, threadsPerBlock, numBlocks);

    STOP_RECORD_TIMER(gpu_timer);

    // ofstream gpu;
    // ofstream cpu;

    // gpu.open("cannyGPU.txt", ios::app);
    // gpu << gpu_timer << endl;
    // cpu.open("cannyCPU.txt", ios::app);
    // cpu << cpu_timer << endl;

    // gpu.close();
    // cpu.close();

    cout << "CPU Canny Edge Detector: " << cpu_timer << " ms" << endl;
    cout << "GPU Canny Edge Detector: " << gpu_timer << " ms" << endl;

    gpuErrChk(cudaUnbindTexture(img_tex));

    gpuErrChk(cudaFreeArray(dev_image));
    gpuErrChk(cudaFree(dev_blur));
    gpuErrChk(cudaFree(dev_edge_dir));
    gpuErrChk(cudaFree(dev_gradient));
    gpuErrChk(cudaFree(dev_visit));
    gpuErrChk(cudaFree(dev_modified));

    free(edges);
}


// int main(int argc, const char* argv[]) {
//     if (argc != 5) {
//         fprintf(stderr, "usage: ./cannyEdge input output width height");
//     }

//     ofstream wr;
//     ifstream r;
//     string gray;
//     int width = atoi(argv[3]), height = atoi(argv[4]), count = 0;
//     int *image = new int[width*height];
//     int *out = new int[width*height];
//     int *edges;
//     float timer, *dir;
//     cudaMalloc((void **) &dir, width * height * sizeof(float));
//     cudaMalloc((void **) &edges, width * height * sizeof(int));

//     wr.open(argv[2]);
//     r.open(argv[1]);

//     if (r.is_open()) {
//         while (getline(r, gray)) {
//             image[count] = atoi(gray.c_str());
//             count++;
//         }
//     }

//     START_TIMER();
//     cannyEdge(image, edges, dir, width, height);
//     gpuErrChk(cudaMemcpy(out, edges, width * height * sizeof(int),
//               cudaMemcpyDeviceToHost));
//     STOP_RECORD_TIMER(timer);

//     if (wr.is_open()) {
//         for (int i = 0; i < width * height; i++) {
//             wr << out[i] << endl;
//         }
//     }

//     cout << "GPU time elapsed: " << timer << " ms" << endl;

//     r.close();
//     wr.close();

//     free(image);
//     free(out);
//     cudaFree(edges);
//     cudaFree(dir);

//     return 0;
// }
