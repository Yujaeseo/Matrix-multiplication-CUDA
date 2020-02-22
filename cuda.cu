#include <iostream>
#include <chrono>
using namespace std;

typedef struct Matrix
{
	//Matrix(): ROW_X(4096), COL_X(4096), ROW_Y(4096), COL_Y(4096){}
	Matrix(): ROW_X(3999), COL_X(3999), ROW_Y(3999), COL_Y(3999){}
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
	float *d_Z,
	int width,
	int width_Z
){
	__shared__ float X_tile[32][32];
	__shared__ float Y_tile[32][32];
	
	int t_row = threadIdx.y; 
	int t_col = threadIdx.x;
	int width_tile = blockDim.y;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	float acc_result = 0;

	for (int i = 0; i < (width + width_tile - 1) / width_tile; i++){
		X_tile[t_row][t_col] = d_X[row * width + i * width_tile + t_col];
		Y_tile[t_row][t_col] = d_Y[(t_row + width_tile*i)*width + col];

		__syncthreads();
		
		for (int k = 0; k < width_tile; k++){
			acc_result += X_tile[t_row][k] * Y_tile[k][t_col];
		}
	}
	__syncthreads(); // 생략 가능하지 않을까
	d_Z[row * width_Z + col] = acc_result;
}

void call_naive_matrix_muliplication_kernel(Matrix* matrix){
	const int THREADS_PER_BLOCK = 1024;
	const int THREAD_BLOCKS = (matrix->ROW_X * matrix->COL_Y + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

	matrix_multiplication<<<THREAD_BLOCKS, THREADS_PER_BLOCK>>>(matrix->d_X, matrix->d_Y, matrix->d_Z, matrix->COL_X, matrix->COL_Y, matrix->ROW_X * matrix->COL_Y);
	cudaDeviceSynchronize();
}

void call_tiled_matrix_multiplication_kernel(Matrix* matrix){
	// const int THREADS_PER_BLOCK = 1024;
	// const int THREAD_BLOCKS = (matrix->ROW_X * matrix->COL_Y + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	dim3 blocks((matrix->ROW_X + 31) /32, (matrix->COL_Y + 31) /32);
	dim3 threads(32, 32);

	tiled_matrix_multiplication<<<blocks, threads>>>(matrix->d_X, matrix->d_Y, matrix->d_Z, matrix->COL_X, matrix->COL_Y);
	cudaDeviceSynchronize();
}

int main (){

	std::chrono::milliseconds naive_matrix_multiplication_kernel_execution_time {};
	std::chrono::milliseconds tiled_matrix_multiplication_kernel_execution_time {};

	std::chrono::time_point<std::chrono::system_clock> matrix_multiplication_kernel_start_time;
	
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
	

	// CALL NAIVE MATRIX MULTIPLICATION KERNEL
	matrix_multiplication_kernel_start_time = std::chrono::system_clock::now();
	call_naive_matrix_muliplication_kernel(matrix);
	naive_matrix_multiplication_kernel_execution_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - matrix_multiplication_kernel_start_time);
	cout << "naive Matrix multiplication kernel execution time : " << naive_matrix_multiplication_kernel_execution_time.count() << "ms" << endl;
	// COPY RESULT MATRIX FROM DEVICE TO HOST
	cudaMemcpy(matrix->h_Z, matrix->d_Z, sizeof(float) * ROW_X * COL_Y, cudaMemcpyDefault);
	for (int i = 0 ; i <100; i++){
		cout << matrix->h_Z[i] << " ";
	}
	// CALL TILED MATRIX MULIPLICATION KERNEL 
	matrix_multiplication_kernel_start_time = std::chrono::system_clock::now();
	call_tiled_matrix_multiplication_kernel(matrix);
	tiled_matrix_multiplication_kernel_execution_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - matrix_multiplication_kernel_start_time);
	cout << "tiled Matrix multiplication kernel execution time : " << tiled_matrix_multiplication_kernel_execution_time.count() << "ms" << endl;
	// COPY RESULT MATRIX FROM DEVICE TO HOST
	cudaMemcpy(matrix->h_Z, matrix->d_Z, sizeof(float) * ROW_X * COL_Y, cudaMemcpyDefault);

	// for (int i = 0 ; i <100; i++){
	// 	cout << matrix->h_Z[i] << " ";
	// }

	delete matrix;
	return 0;
}
