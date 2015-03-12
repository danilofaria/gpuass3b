#include <sys/time.h>

#include <stdio.h>
#include <cuda.h>
#include <math.h>

// STUDENTS: be sure to set the single define at the top of this file, 
// depending on which machines you are running on.
#include "im1.h"
#define PI           3.14159265358979323846  /* pi */
#define RADIUS 2
#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 8


// handy error macro:
#define GPU_CHECKERROR( err ) (gpuCheckError( err, __FILE__, __LINE__ ))
static void gpuCheckError( cudaError_t err,
                          const char *file,
                          int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
               file, line );
        exit( EXIT_FAILURE );
    }
}

//
// your __global__ kernel can go here, if you want:
//

 __global__ void gaussianBlur (unsigned int w, unsigned int h, float * G, float *imageArray, float *blurredImageArray, int colorNumber)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    // do nothing if we are not in the useable space of
    // threads (see kernel launch call: you may be creating
    // more threads than you need)
    if (x >= w || y >= h ) return;
    int stride = colorNumber*(w*h);

    int idx = ((y * w) + x) + stride;

    float weight = 0.f;
    float color=0.f, g_ij;
    int g_idx;
    int idx2;
    int i,j;

    __shared__ float s_data [BLOCK_SIZE_Y + RADIUS*2][BLOCK_SIZE_X + RADIUS*2]; 

    s_data[RADIUS + threadIdx.y][RADIUS + threadIdx.x] = imageArray[idx];

    if(threadIdx.x < RADIUS){
        idx2 = ((y * w) + max(0,x - 2*threadIdx.x - 1)) + stride;
        s_data[RADIUS + threadIdx.y][RADIUS - threadIdx.x - 1] = imageArray[idx2];
    }
    if(threadIdx.y < RADIUS){
        idx2 = (((max(0, y - 2*threadIdx.y - 1))* w) + x) + stride;
        s_data[RADIUS - threadIdx.y - 1][RADIUS + threadIdx.x] = imageArray[idx2];
    }
    if(threadIdx.x >= BLOCK_SIZE_X - RADIUS){
        idx2 = ((y * w) + min(w-1, x + RADIUS)) + stride;
        s_data[RADIUS + threadIdx.y][threadIdx.x + 2*RADIUS] = imageArray[idx2];
    }
    if(threadIdx.y >= BLOCK_SIZE_Y - RADIUS){
        idx2 = ((min(h - 1, y + RADIUS)* w) + x) + stride;
        s_data[threadIdx.y + 2*RADIUS][RADIUS + threadIdx.x] = imageArray[idx2];
    }

    if(threadIdx.x < RADIUS && threadIdx.y < RADIUS){
        idx2 = (((max(0, y - 2*threadIdx.y - 1)) * w) + max(0,x - 2*threadIdx.x - 1)) + stride;
        s_data[RADIUS - threadIdx.y - 1][RADIUS - threadIdx.x - 1] = imageArray[idx2];
    }
    if(threadIdx.x >= BLOCK_SIZE_X - RADIUS && threadIdx.y >= BLOCK_SIZE_Y - RADIUS){
        idx2 = ((min(h - 1, y + RADIUS) * w) + min(w-1, x + RADIUS)) + stride;
        s_data[threadIdx.y + 2*RADIUS][threadIdx.x + 2*RADIUS] = imageArray[idx2];
    }
    if(threadIdx.x < RADIUS && threadIdx.y >= BLOCK_SIZE_Y - RADIUS){
        idx2 = ((min(h - 1, y + RADIUS) * w) + max(0,x - 2*threadIdx.x - 1)) + stride;
        s_data[threadIdx.y + 2*RADIUS][RADIUS - threadIdx.x - 1] = imageArray[idx2];
    }
    if(threadIdx.y < RADIUS && threadIdx.x >= BLOCK_SIZE_X - RADIUS){
        idx2 = (((max(0, y - 2*threadIdx.y - 1)) * w) + min(w-1, x + RADIUS)) + stride;
        s_data[RADIUS - threadIdx.y - 1][threadIdx.x + 2*RADIUS] = imageArray[idx2];
    }

    __syncthreads();

    for(i = max(0, y - RADIUS); i <= min(h - 1, y + RADIUS); i++){
        for(j = max(0, x - RADIUS); j <= min(w-1, x + RADIUS); j++){

            g_idx = (i-y+RADIUS)*(2*RADIUS+1) + (j-x+RADIUS);
            g_ij = G[g_idx];

            weight += g_ij;

            idx2 = ((i * w) + j) + stride;

            color += s_data[RADIUS + threadIdx.y + i-y][RADIUS + threadIdx.x+ j-x] * g_ij;
        }
    }

    blurredImageArray[idx] = color/weight;

}

int main (int argc, char *argv[])
{
 
    printf("reading openEXR file %s\n", argv[1]);
        
    int w, h;   // the width & height of the image, used frequently!
    struct timeval t0, t1, t2, t3;

    // First, convert the openEXR file into a form we can use on the CPU
    // and the GPU: a flat array of floats:
    // This makes an array h*w*sizeof(float)*3, with sequential r/g/b indices
    // don't forget to free it at the end

    float *h_imageArray;
    readOpenEXRFile (argv[1], &h_imageArray, w, h);
    // 
    // serial code: saves the image in "hw1_serial.exr"
    //

    // for every pixel in p, get it's Rgba structure, and convert the
    // red/green/blue values there to luminance L, effectively converting
    // it to greyscale:
    int maxTested = w*h*3;
    float * h_blurredImageArray;
    h_blurredImageArray = (float*) malloc (maxTested * sizeof(float));

    float sigma = (1.0f*RADIUS)/3.f;
    float twoSigma2 = 2*sigma*sigma;
    float twoPiSigma2 = PI*twoSigma2;
    float weight;
    float r, g, b;
    unsigned int idx2, g_idx;
    float g_ij;

    // Pre compute kernel matrix
    float m = (2*RADIUS+1);
    float h_G[(2*RADIUS+1)*(2*RADIUS+1)];
    for(int i = - RADIUS; i <= RADIUS; i++){
        for(int j = - RADIUS; j <= RADIUS; j++){
            g_ij = exp(- (j*j + i*i) / twoSigma2 );
            g_ij /= twoPiSigma2;
            int idx = (i + RADIUS)*m + (j + RADIUS);
            h_G[idx] = g_ij;
        }
    }

    // Serial gaussian blur
    gettimeofday (&t0, 0);
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            
            unsigned int idx = ((i * w) + j) * 3;
            
            weight = 0.f;
            r = 0.f, g = 0.f, b= 0.f;

            for(int y = max(0, i - RADIUS); y <= min(h - 1, i + RADIUS); y++){
                for(int x = max(0, j - RADIUS); x <= min(w-1, j + RADIUS); x++){
                    
                    g_idx = (y-i+RADIUS)*(2*RADIUS+1) + (x-j+RADIUS);
                    g_ij = h_G[g_idx];
                    weight += g_ij;

                    idx2 = ((y * w) + x) * 3;
                    r += h_imageArray[idx2]*g_ij;
                    g += h_imageArray[idx2+1]*g_ij;
                    b += h_imageArray[idx2+2]*g_ij;
                }
            }

            h_blurredImageArray[idx] = r/weight;

            h_blurredImageArray[idx+1] = g/weight;

            h_blurredImageArray[idx+2] = b/weight;

       }
    }
    gettimeofday (&t1, 0);
    printf("writing output image hw1_serial.exr\n");
    writeOpenEXRFile ("hw1_serial.exr", h_blurredImageArray, w, h);
    free(h_imageArray); // make sure you free it: if you use this variable
                        // again, readOpenEXRFile will allocate more memory
    free(h_blurredImageArray);


    //
    // Now the GPU version: it will save whatever is in h_imageArray
    // to the file "hw1_gpu.exr"
    //
    
    // read the file again - the file read allocates memory for h_imageArray:
    readOpenEXRFile (argv[1], &h_imageArray, w, h);



    // at this point, h_imageArray has sequenial floats for red, green , and
    // blue for each pixel: r,g,b,r,g,b,r,g,b,r,g,b. You need to copy
    // this array to GPU global memory, and have one thread per pixel compute
    // the luminance value, with which you will overwrite each r,g,b, triple.

    //
    // process it on the GPU: 1) copy it to device memory, 2) process
    // it with a 2d grid of 2d blocks, with each thread assigned to a 
    // pixel. then 3) copy it back.
    //


    //
    // Your memory copy, & kernel launch code goes here:
    //

    gettimeofday (&t2, 0);
    float * h_imageArray2;

    GPU_CHECKERROR(
    cudaHostAlloc ((void **) &h_imageArray2,
                maxTested * sizeof (float),
                cudaHostAllocDefault)
    );

    // re-arrange data to get coalesced memory access
    for(int c = 0; c < h*w; c++){
        h_imageArray2[c] = h_imageArray[3*c];        
        h_imageArray2[h*w + c] = h_imageArray[3*c+1];        
        h_imageArray2[h*w*2 + c] = h_imageArray[3*c+2];        
    }

    float *d_imageArray;
 
    GPU_CHECKERROR(
    cudaMalloc ((void **) &d_imageArray, maxTested * sizeof (float))
    );

    GPU_CHECKERROR(
    cudaMemcpy ((void *) d_imageArray,
                (void *) h_imageArray2,
                maxTested * sizeof (float),
                cudaMemcpyHostToDevice)
    );

    float *d_blurredImageArray;
    GPU_CHECKERROR(
    cudaMalloc ((void **) &d_blurredImageArray, maxTested * sizeof (float))
    );
 
    float *d_G;

    GPU_CHECKERROR(
    cudaMalloc ((void **) &d_G, m*m * sizeof (float))
    );
 
    GPU_CHECKERROR(
    cudaMemcpy ((void *) d_G,
                (void *) h_G,
                m*m * sizeof (float),
                cudaMemcpyHostToDevice)
    );

    int block_w = min(BLOCK_SIZE_X,w);
    int block_h = min(BLOCK_SIZE_Y,h);
    unsigned int num_blocks_x = ceil (1.0*w / (1.0*block_w) );
    unsigned int num_blocks_y = ceil (1.0*h / (1.0*block_h) );

    dim3 dimGrid(num_blocks_x, num_blocks_y, 1);
    dim3 dimBlock(block_w, block_h, 1);      
    // launch the kernel, once for each color channel:
    gaussianBlur<<<dimGrid, dimBlock>>>
                                        (w,h,d_G,
                                        d_imageArray,
                                        d_blurredImageArray,0);
    gaussianBlur<<<dimGrid, dimBlock>>>
                                        (w,h,d_G,
                                        d_imageArray,
                                        d_blurredImageArray,1);
    gaussianBlur<<<dimGrid, dimBlock>>>
                                        (w,h,d_G,
                                        d_imageArray,
                                        d_blurredImageArray,2);
 
    cudaMemcpy ((void *) h_imageArray2,
                (void *) d_blurredImageArray,
                maxTested * sizeof(float),
                cudaMemcpyDeviceToHost);

    // re-arrange data back to normal
    for(int c = 0; c < h*w; c++){
        h_imageArray[3*c] = h_imageArray2[c];        
        h_imageArray[3*c+1] = h_imageArray2[h*w + c];        
        h_imageArray[3*c+2] = h_imageArray2[h*w*2 + c];        
    }
    
    // make sure the GPU is finished doing everything!
    cudaDeviceSynchronize();
    gettimeofday (&t3, 0);
    // free up the memory:
    cudaFree (d_imageArray);
    cudaFree (d_blurredImageArray);
    cudaFree (d_G);

    // All your work is done. Here we assume that you have copied the 
    // processed image data back, frmm the device to the host, into the
    // original host array h_imageArray. You can do it some other way,
    // this is just a suggestion

       // complete the timing:
    float timdiff1 = (1000000.0*(t1.tv_sec - t0.tv_sec) + (t1.tv_usec - t0.tv_usec)) / 1000000.0;
    float timdiff2 = (1000000.0*(t3.tv_sec - t2.tv_sec) + (t3.tv_usec - t2.tv_usec)) / 1000000.0;
    
    printf("writing output image hw1_gpu.exr\n");
    writeOpenEXRFile ("hw1_gpu.exr", h_imageArray, w, h);
    free (h_imageArray);
    cudaFreeHost(h_imageArray2);

    printf ("Serial version: %.2f\n", timdiff1);
    printf ("Parallel version: %.2f\n", timdiff2);

    printf("done.\n");

    return 0;
}

