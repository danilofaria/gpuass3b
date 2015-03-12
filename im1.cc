#include "im1.h"

#ifdef HAS_OPENEXR

#include <ImfRgbaFile.h>
#include <ImfStringAttribute.h>
#include <ImfMatrixAttribute.h>
#include <ImfArray.h>

#include <iostream>
#include <stdlib.h>   

using namespace std;
using namespace Imf;
using namespace Imath;


void
readRgba (const char fileName[],
          Array2D<Rgba> &pixels,
          int &width,
          int &height)
{
    //
    // Read an RGBA image using class RgbaInputFile:
    //
    //	- open the file
    //	- allocate memory for the pixels
    //	- describe the memory layout of the pixels
    //	- read the pixels from the file
    //
    
    RgbaInputFile file (fileName);
    Box2i dw = file.dataWindow();
    
    width  = dw.max.x - dw.min.x + 1;
    height = dw.max.y - dw.min.y + 1;
    pixels.resizeErase (height, width);
    
    file.setFrameBuffer (&pixels[0][0] - dw.min.x - dw.min.y * width, 1, width);
    file.readPixels (dw.min.y, dw.max.y);
}


void
writeRgba (const char fileName[],
           const Rgba *pixels,
           int width,
           int height)
{
    //
    // Write an RGBA image using class RgbaOutputFile.
    //
    //	- open the file
    //	- describe the memory layout of the pixels
    //	- store the pixels in the file
    //
    
    
    RgbaOutputFile file (fileName, width, height, WRITE_RGBA);
    file.setFrameBuffer (pixels, 1, width);
    file.writePixels (height);
}


void
makeFloatImageArray (float **imageArray,
                     Array2D<Rgba> &p,
                     int w,
                     int h)
{
 
    // 1 float each for r, g, & b
    *imageArray = (float *) malloc (sizeof (float) * w * h * 3);
    
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            
            Rgba &px = p[y][x];  // get the pixel we are interested in
            
            (*imageArray)[(y * w + x) * 3] =   (float) px.r;
            (*imageArray)[(y * w + x) * 3+1] = (float) px.g;
            (*imageArray)[(y * w + x) * 3+2] = (float) px.b;
        }
    }
}

void
makeRgbaArray (Array2D<Rgba> &p,
               float *imageArray,
               int w,
               int h)
{
 
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            
            Rgba &px = p[y][x];  // get the pixel we are interested in

            px.r = imageArray[(y * w + x) * 3];
            px.g = imageArray[(y * w + x) * 3+1];
            px.b = imageArray[(y * w + x) * 3+2];
            px.a = 1;
        }
    }
}


// note: called has to free imageArray!
bool readOpenEXRFile (
    const char fileName[],
    float **imageArray,
    int &w,
    int &h)
{

   Array2D<Rgba> p;

    readRgba (fileName, p, w, h);

    // 1 float each for r, g, & b
    *imageArray = (float *) malloc (sizeof (float) * w * h * 3);
    
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            
            Rgba &px = p[y][x];  // get the pixel we are interested in
            
            (*imageArray)[(y * w + x) * 3] =   (float) px.r;
            (*imageArray)[(y * w + x) * 3+1] = (float) px.g;
            (*imageArray)[(y * w + x) * 3+2] = (float) px.b;
        }
    }

    return true;
}

bool writeOpenEXRFile (
    const char fileName[],
    float *imageArray,
    int w,
    int h)
{
 
    Array2D<Rgba> newImage(h, w);

    // Here we overwrite the newImage:
    makeRgbaArray (newImage, imageArray, w, h);
    
    writeRgba(fileName, &newImage[0][0], w, h);
    
    return true;

}


#endif

#ifdef HAS_OPENCV



#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

bool readOpenEXRFile(const char *name, float **imageArray, int &width, int &height)
{
    IplImage *image = cvLoadImage(name, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
    if (!image->imageData)
    {
        std::cout << "Could not open or find the image!" << std::endl;
        return false;
    }
    
    int w = image->width, h = image->height;
    *imageArray = (float *) malloc (sizeof(float) * w * h * 3);
    char *buffer = (char *) malloc (sizeof(char) * w * h * 4 * 3);
    
    // Copy the data from the IplImage byte by byte
    // tricks here:
    //   - EXR image's format is 3 channels of float, so 12 bytes total for one pixel
    //   - IplImage's storage is BGR, so need to rearrange the sequence to RGB
    for (int i = 0; i < h; i++){
        for (int j = 0; j < w; j++){
            int idx = image->widthStep * i + j * 12;
            buffer[idx] = image->imageData[idx+8];
            buffer[idx+1] = image->imageData[idx+9];
            buffer[idx+2] = image->imageData[idx+10];
            buffer[idx+3] = image->imageData[idx+11];
            buffer[idx+4] = image->imageData[idx+4];
            buffer[idx+5] = image->imageData[idx+5];
            buffer[idx+6] = image->imageData[idx+6];
            buffer[idx+7] = image->imageData[idx+7];
            buffer[idx+8] = image->imageData[idx];
            buffer[idx+9] = image->imageData[idx+1];
            buffer[idx+10] = image->imageData[idx+2];
            buffer[idx+11] = image->imageData[idx+3];
        }
    }
    
    memcpy(*imageArray, buffer, sizeof(float) * w * h * 3);
    width = w; height = h;
    cvReleaseImage(&image);
    free(buffer);
    return true;
}

bool writeOpenEXRFile(const char *name, float *imageArray, const int width, const int height)
{
    IplImage *image = cvCreateImage(cvSize(width, height), IPL_DEPTH_32F, 3);
    
    char *buffer = (char *) malloc (width * height * 4 * 3);
    memcpy(buffer, imageArray, width * height * sizeof(float) * 3);
    for (int i = 0; i < height; i++){
        for (int j = 0; j < width; j++){
            int idx = image->widthStep * i + j * 12;
    	    image->imageData[idx] = buffer[idx+8];
    	    image->imageData[idx+1] = buffer[idx+9];
            image->imageData[idx+2] = buffer[idx+10];
    	    image->imageData[idx+3] = buffer[idx+11];
            image->imageData[idx+4] = buffer[idx+4];
    	    image->imageData[idx+5] = buffer[idx+5];
            image->imageData[idx+6] = buffer[idx+6];
    	    image->imageData[idx+7] = buffer[idx+7];
            image->imageData[idx+8] = buffer[idx];
    	    image->imageData[idx+9] = buffer[idx+1];
            image->imageData[idx+10] = buffer[idx+2];
    	    image->imageData[idx+11] = buffer[idx+3];
        }
    }
    
    cvSaveImage(name, image);
    cvReleaseImage(&image);
    free(buffer);
    return true;
}

// Sample example program testing the function
// Read in an EXR file, clean its green and blue channel, write it back to another EXR file
int test_harness( int argc, char** argv )
{
    if( argc != 3)
    {
        std::cout <<" Usage: exrUtil inputImage outputImage" << std::endl;
        return -1;
    }
    
    float *h_imageArray;
    int w, h;
    readOpenEXRFile(argv[1], &h_imageArray, w, h);
    
    // clean the green and blue channel
    for (int i = 0; i < h; i++)
        for (int j = 0; j < w; j++){
    	    h_imageArray[i*w*3+j*3+2] = 0; h_imageArray[i*w*3+j*3+1] = 0;
        }
    
    writeOpenEXRFile(argv[2], h_imageArray, w, h);
    free(h_imageArray);
    return 0;
}


#endif

