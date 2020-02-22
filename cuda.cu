#include <iostream>
#include <chrono>
using namespace std;

typedef struct Matrix
{
	Matrix(): ROW_X(4096), COL_X(4096), ROW_Y(4096), COL_Y(4096){}
	
	~Matrix(){	
		delete [] h_X;
		delete [] h_Y;
		delete [] h_Z;

		cudaFree(d_X);
		cudaFree(d_Y);
		cudaFree(d_Z);
	}

	float *d_X;
	float *d_Y;
	float *d_Z;
	float *h_X;
	float *h_Y; 
	float *h_Z;

	int ROW_X;
	int COL_X;
	int ROW_Y; 
	int COL_Y;
} Matrix;

__global__
void matrix_multiplication
(
	float * d_X,
 	float * d_Y,
 	float * d_Z,
 	unsigned int width_X,
 	unsigned int width_Z, 
 	unsigned int length_Z
 ){
	// thread id in thread block
	int local_th_id = threadIdx.x;
	int global_th_id = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(global_th_id >= length_Z)
		return;

	float temp = 0;
	int index_X = width_X * (global_th_id / width_Z);
	int index_Y = global_th_id % width_Z;

	for (int i = 0; i < width_X; i++){
		temp += d_X[index_X] * d_Y[index_Y];
		index_X++;
		index_Y += width_Z;
	}

	d_Z[global_th_id] = temp;
}

__global__
void tiled_matrix_multiplication
(
	float *d_X,
	float *d_Y,
	float *d_Z
){
	__shared__ float Xs[32][32];
	__shared__ float Ys[32][32];



}

void call_naive_matrix_muliplication_kernel(Matrix* matrix){
	std::chrono::milliseconds matrix_multiplication_kernel_execution_time {};
	const int THREADS_PER_BLOCK = 1024;
	const int THREAD_BLOCKS = (matrix->ROW_X * matrix->COL_Y + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

	std::chrono::time_point<std::chrono::system_clock> matrix_multiplication_kernel_start_time = std::chrono::system_clock::now();
	matrix_multiplication<<<THREAD_BLOCKS, THREADS_PER_BLOCK>>>(matrix->d_X, matrix->d_Y, matrix->d_Z, matrix->COL_X, matrix->COL_Y, matrix->ROW_X * matrix->COL_Y);
	cudaDeviceSynchronize();
	matrix_multiplication_kernel_execution_time += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - matrix_multiplication_kernel_start_time);

	cout << "Matrix multiplication kernel execution time : " << matrix_multiplication_kernel_execution_time.count() << "ms" << endl;
}

void call_tiled_matrix_multiplication_kernel(Matrix* matrix){
	std::chrono::milliseconds matrix_multiplication_kernel_execution_time {};
	// const int THREADS_PER_BLOCK = 1024;
	// const int THREAD_BLOCKS = (matrix->ROW_X * matrix->COL_Y + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

	dim3 blocks((matrix->ROW_X + 31) /32, (matrix->COL_Y + 31) /32);
	dim3 threads(32, 32);

	std::chrono::time_point<std::chrono::system_clock> matrix_multiplication_kernel_start_time = std::chrono::system_clock::now();
	tiled_matrix_multiplication<<<blocks, threads>>>(matrix->d_X, matrix->d_Y, matrix->d_Z);
	cudaDeviceSynchronize();
	matrix_multiplication_kernel_execution_time += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - matrix_multiplication_kernel_start_time);

	cout << "Matrix multiplication kernel execution time : " << matrix_multiplication_kernel_execution_time.count() << "ms" << endl;
}

int main (){
	Matrix *matrix = new Matrix();

	const int ROW_X = matrix->ROW_X;
	const int COL_X = matrix->COL_X;
	const int ROW_Y = matrix->ROW_Y;
	const int COL_Y = matrix->COL_Y;

	matrix->h_X = new float[ROW_X * COL_X];
	matrix->h_Y = new float[ROW_Y * COL_Y];
	matrix->h_Z = new float[COL_X * ROW_Y];
	
	// INITIALIZATION TWO MATRIX
	for (int i = 0; i < ROW_X * COL_X; i++){ matrix->h_X[i] = float(1);}
	for (int i = 0; i < ROW_Y * COL_Y; i++){ matrix->h_Y[i] = float(1);}
	for (int i = 0; i < ROW_X * COL_Y; i++){ matrix->h_Z[i] = float(0);}	

	// ALLOCATE MEMORY TO DEVICE
	cudaMalloc((void**) &(matrix->d_X), sizeof(float) * ROW_X * COL_X);
	cudaMalloc((void**) &(matrix->d_Y), sizeof(float) * ROW_Y * COL_Y);
	cudaMalloc((void**) &(matrix->d_Z), sizeof(float) * ROW_X * COL_Y);

	// COPY MEMORY FROM HOST TO DEVICE
	cudaMemcpy(matrix->d_X, matrix->h_X, sizeof(float) * ROW_X * COL_X, cudaMemcpyDefault);
	cudaMemcpy(matrix->d_Y, matrix->h_Y, sizeof(float) * ROW_Y * COL_Y, cudaMemcpyDefault);
	cudaMemcpy(matrix->d_Z, matrix->h_Z, sizeof(float) * ROW_X * COL_Y, cudaMemcpyDefault);
	
	//call_naive_matrix_muliplication_kernel(matrix);
	call_tiled_matrix_multiplication_kernel(matrix);
	// COPY RESULT MATRIX FROM DEVICE TO HOST
	cudaMemcpy(matrix->h_Z, matrix->d_Z, sizeof(float) * ROW_X * COL_Y, cudaMemcpyDefault);

	//for (int i = 0 ; i < ROW_X * COL_Y; i++){
	//	cout << h_Z[i] << " ";
	//}

	delete matrix;

	return 0;
}
