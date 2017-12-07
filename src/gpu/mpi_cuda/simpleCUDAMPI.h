void initData(float * data, int dataSize);
void computeGPU(float * hostData, int blockSize, int gridSize);
float sum(float * data, int size);
void my_abort(int err);

