
// STUDENTS: only allow ONE of these next two defines to be uncommented: the
// first one if you are compiling on CLIC (or your own machine with OpenEXR
// installed) OR the second one if you are compiling on the nvidia cluster.
#define HAS_OPENEXR 1
// #define HAS_OPENCV 1

bool readOpenEXRFile (
    const char fileName[],
    float **imageArray,
    int &w,
    int &h);

bool writeOpenEXRFile (
    const char fileName[],
    float *imageArray,
    int w,
    int h);
