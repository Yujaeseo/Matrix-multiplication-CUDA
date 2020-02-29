#include <iostream>
#include <chrono>
#include <random>
#include <eigen/Sparse>
using namespace std;
using namespace Eigen;

#define DATA_TYPE int
#define WIDTH_TILE 32

typedef struct Matrix_multiplication
{
	Matrix_multiplication(): ROW_X(1024), COL_X(1024), ROW_Y(1024), COL_Y(1024) {initialize();}
	Matrix_multiplication(int input_ROW_X, int input_COL_X, int input_ROW_Y, int input_COL_Y): ROW_X(input_ROW_X), COL_X(input_COL_X), ROW_Y(input_ROW_Y), COL_Y(input_COL_Y){initialize();}
	Matrix_multiplication(const Matrix_multiplication &copied_matrix_mult){
		ROW_X = copied_matrix_mult.ROW_X;
		COL_X = copied_matrix_mult.COL_X;
		ROW_Y = copied_matrix_mult.ROW_Y;
		COL_Y = copied_matrix_mult.COL_Y;

		h_X = new DATA_TYPE[ROW_X * COL_X];
		h_Y = new DATA_TYPE[ROW_Y * COL_Y];
		h_Z = new DATA_TYPE[ROW_X * COL_Y];

		for (int i = 0; i < ROW_X * COL_X; i++){h_X[i] = copied_matrix_mult.h_X[i];}
		for (int i = 0; i < ROW_Y * COL_Y; i++){h_Y[i] = copied_matrix_mult.h_Y[i];}
		for (int i = 0; i < ROW_X * COL_Y; i++){h_Z[i] = copied_matrix_mult.h_Z[i];}
	}
	~Matrix_multiplication(){	
		delete [] h_X;
		delete [] h_Y;
		delete [] h_Z;

		cudaFree(d_X);
		cudaFree(d_Y);
		cudaFree(d_Z);
	}

	void initialize(){
		h_X = new DATA_TYPE[ROW_X * COL_X];
		h_Y = new DATA_TYPE[ROW_Y * COL_Y];
		h_Z = new DATA_TYPE[ROW_X * COL_Y];
	}

	DATA_TYPE *d_X;
	DATA_TYPE *d_Y;
	DATA_TYPE *d_Z;
	DATA_TYPE *h_X;
	DATA_TYPE *h_Y; 
	DATA_TYPE *h_Z;

	int ROW_X;
	int COL_X;
	int ROW_Y; 
	int COL_Y;
} Matrix_multiplication;

// GPU KERNEL NAIVE KERNEL
__global__
void naive_matrix_multiplication
(
	DATA_TYPE * d_X,
 	DATA_TYPE * d_Y,
 	DATA_TYPE * d_Z,
 	unsigned int width_X,
 	unsigned int width_Z, 
 	unsigned int length_Z
 ){
	// thread id in thread block
	int local_th_id = threadIdx.x;
	int global_th_id = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(global_th_id >= length_Z)
		return;

	DATA_TYPE temp = 0;
	int index_X = width_X * (global_th_id / width_Z);
	int index_Y = global_th_id % width_Z;

	for (int i = 0; i < width_X; i++){
		temp += d_X[index_X] * d_Y[index_Y];
		index_X++;
		index_Y += width_Z;
	}

	d_Z[global_th_id] = temp;
}

// GPU TILED KERNEL
__global__
void tiled_matrix_multiplication
(
	DATA_TYPE *d_X,
	DATA_TYPE *d_Y,
	DATA_TYPE *d_Z,
	int width_X,
	int width_Z,
	int height_Z
){
	__shared__ DATA_TYPE X_tile[WIDTH_TILE][WIDTH_TILE];
	__shared__ DATA_TYPE Y_tile[WIDTH_TILE][WIDTH_TILE];
	
	int t_row = threadIdx.y; int t_col = threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y; int col = blockIdx.x * blockDim.x + threadIdx.x;
	
	DATA_TYPE acc_result = 0;

	for (int i = 0; i < (width_X + WIDTH_TILE - 1) / WIDTH_TILE; i++){
		if (row < height_Z && (i*WIDTH_TILE + t_col) < width_X)
			X_tile[t_row][t_col] = d_X[row * width_X + i * WIDTH_TILE + t_col];
		else
			X_tile[t_row][t_col] = 0;

		if (col < width_Z && (i*WIDTH_TILE + t_row) < width_X)
			Y_tile[t_row][t_col] = d_Y[(t_row + WIDTH_TILE*i)*width_Z + col];
		else
			Y_tile[t_row][t_col] = 0;

		__syncthreads();
		
		for (int k = 0; k < WIDTH_TILE; k++){
			acc_result += X_tile[t_row][k] * Y_tile[k][t_col];
		}
		__syncthreads();
	}

	if (row < height_Z && col < width_Z)
		d_Z[row * width_Z + col] = acc_result;
}

// TRANSFER MATRIX DATA TO DEVICE
void matrix_data_transfer_to_device(Matrix_multiplication* matrix_mult){

	const int ROW_X = matrix_mult->ROW_X;
	const int COL_X = matrix_mult->COL_X;
	const int ROW_Y = matrix_mult->ROW_Y;
	const int COL_Y = matrix_mult->COL_Y;

	// ALLOCATE MEMORY TO DEVICE
	cudaMalloc((void**) &(matrix_mult->d_X), sizeof(DATA_TYPE) * ROW_X * COL_X);
	cudaMalloc((void**) &(matrix_mult->d_Y), sizeof(DATA_TYPE) * ROW_Y * COL_Y);
	cudaMalloc((void**) &(matrix_mult->d_Z), sizeof(DATA_TYPE) * ROW_X * COL_Y);

	// COPY MEMORY FROM HOST TO DEVICE
	cudaMemcpy(matrix_mult->d_X, matrix_mult->h_X, sizeof(DATA_TYPE) * ROW_X * COL_X, cudaMemcpyDefault);
	cudaMemcpy(matrix_mult->d_Y, matrix_mult->h_Y, sizeof(DATA_TYPE) * ROW_Y * COL_Y, cudaMemcpyDefault);
	cudaMemcpy(matrix_mult->d_Z, matrix_mult->h_Z, sizeof(DATA_TYPE) * ROW_X * COL_Y, cudaMemcpyDefault);
}

// EXECUTE NAIVE MATRIX MULTIPLICATION KERNEL 
void call_naive_matrix_multiplication_kernel(Matrix_multiplication* matrix_mult){
	const int THREADS_PER_BLOCK = 1024;
	const int THREAD_BLOCKS = (matrix_mult->ROW_X * matrix_mult->COL_Y + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

	naive_matrix_multiplication<<<THREAD_BLOCKS, THREADS_PER_BLOCK>>>(matrix_mult->d_X, matrix_mult->d_Y, matrix_mult->d_Z, matrix_mult->COL_X, matrix_mult->COL_Y, matrix_mult->ROW_X * matrix_mult->COL_Y);
	cudaDeviceSynchronize();
}

// EXECUTE TILED MATRIX MULTIPLICATION KERNEL 
void call_tiled_matrix_multiplication_kernel(Matrix_multiplication* matrix_mult){
	dim3 blocks((matrix_mult->COL_Y + WIDTH_TILE - 1) /WIDTH_TILE,(matrix_mult->ROW_X + WIDTH_TILE - 1) /WIDTH_TILE);
	dim3 threads(WIDTH_TILE, WIDTH_TILE);

	tiled_matrix_multiplication<<<blocks, threads>>>(matrix_mult->d_X, matrix_mult->d_Y, matrix_mult->d_Z, matrix_mult->COL_X, matrix_mult->COL_Y, matrix_mult->ROW_X);
	cudaDeviceSynchronize();
}

// COMPARE TWO MATRIX MULTIPLICATION RESULT 
bool compare_multiplication_result(DATA_TYPE matrix_result1 [], DATA_TYPE matrix_result2[], int result_length){

	for (int i = 0; i < result_length; i++){
		if (matrix_result1[i] != matrix_result2[i]){
			return false;
		}
	}

	return true;
}

// TEST MATRIX MULTIPLICATION KERNEL
void check_matrix_multiplication_result(const int trial_num){
	unsigned int seed = chrono::system_clock::now().time_since_epoch().count();
	mt19937 gen(seed);
	uniform_int_distribution<mt19937::result_type> dist_for_element(1,50);
	uniform_int_distribution<mt19937::result_type> dist_for_dim(100, 500);

	for(int trial_count = 0; trial_count < trial_num; trial_count++){
		int ROW_X, COL_X, ROW_Y, COL_Y;
		
		// INITIALIZE X, Y, Z MATRIX DIMENSION		
		ROW_X = dist_for_dim(gen);
		COL_X = ROW_Y = dist_for_dim(gen);
		COL_Y = dist_for_dim(gen);
		cout << "ROUND " << trial_count + 1 << ": TEST MATRIX MULTIPLICATION... X(" << ROW_X << "," << COL_X << ") and Y(" << ROW_Y << "," << COL_Y << ")" << endl;
		// MATRIX MULTIPLICATION STRUCT FOR NAIVE KERNEL
		Matrix_multiplication *test_matrix_mult = new Matrix_multiplication(ROW_X, COL_X, ROW_Y, COL_Y);

		// INITIALIZATION TWO MATRIX
		for (int i = 0; i < ROW_X * COL_X; i++){ test_matrix_mult->h_X[i] = DATA_TYPE(dist_for_element(gen));}
		for (int i = 0; i < ROW_Y * COL_Y; i++){ test_matrix_mult->h_Y[i] = DATA_TYPE(dist_for_element(gen));}
		
		// MATRIX MULTIPLICATION STRUCT FOR TILED KERNEL
		Matrix_multiplication *test_matrix_mult2 = new Matrix_multiplication(*test_matrix_mult);

		// DATA TRANSFER FROM HOST TO DEVICE
		matrix_data_transfer_to_device(test_matrix_mult);
		matrix_data_transfer_to_device(test_matrix_mult2);

		// KERNEL EXECUTION
		call_naive_matrix_multiplication_kernel(test_matrix_mult);
		call_tiled_matrix_multiplication_kernel(test_matrix_mult2);

		// DATA TRANSFER FROM DEIVE TO HOST 
		cudaMemcpy(test_matrix_mult->h_Z, test_matrix_mult->d_Z, sizeof(DATA_TYPE) * ROW_X * COL_Y, cudaMemcpyDefault);
		cudaMemcpy(test_matrix_mult2->h_Z, test_matrix_mult2->d_Z, sizeof(DATA_TYPE) * ROW_X * COL_Y, cudaMemcpyDefault);

		// EIGEN MATRIX MULTIPLICATION FOR TESTING NAIVE MATRIX MULTIPLICATION KERNEL  
		Matrix<int,-1,-1,RowMajor> matrix_dense_X, matrix_dense_Y, matrix_dense_Z; 
		// CONVERT 1D ARRAY TO EIGEN DENSE MATRIX  
		matrix_dense_X = Map<Matrix<int,-1,-1,RowMajor>, 0, OuterStride<>>(test_matrix_mult->h_X, ROW_X, COL_X, OuterStride<>(COL_X));
		matrix_dense_Y = Map<Matrix<int,-1,-1,RowMajor>, 0, OuterStride<>>(test_matrix_mult->h_Y, ROW_Y, COL_Y, OuterStride<>(COL_Y));
		// EIGEN MATRIX MULTIPLICATION AND GET 1D RESULT ARRAY
		matrix_dense_Z = matrix_dense_X * matrix_dense_Y;
		DATA_TYPE *eigen_matrix_result = matrix_dense_Z.data();

		// COMPARE MULTIPLICATION RESULT
		bool compare_result_naive_and_eigen;
		bool compare_result_tiled_and_naive;

		compare_result_naive_and_eigen = compare_multiplication_result(test_matrix_mult->h_Z, eigen_matrix_result, ROW_X*COL_Y);
		if (compare_result_tiled_and_naive)
			cout << "Eigen vs naive => " << "same matrix multiplication result." << endl;

		compare_result_tiled_and_naive = compare_multiplication_result(test_matrix_mult->h_Z, test_matrix_mult2->h_Z, ROW_X*COL_Y);
		if (compare_result_tiled_and_naive)
			cout << "Tiled vs naive => " << "same matrix multiplication result." << endl;

		// DELETE TESTED MATRIX
		delete test_matrix_mult;
		delete test_matrix_mult2;
	}
}

// PRINT 2D MATRIX 
void print_matrix(DATA_TYPE matrix [], int ROW, int COL){
	for (int i = 0; i < ROW * COL; i++){
		cout << matrix[i] << "  ";
		if((i+1)%COL==0)
			cout << "\n";
	}
}

int main (){

	std::chrono::milliseconds naive_matrix_multiplication_kernel_execution_time {};
	std::chrono::milliseconds tiled_matrix_multiplication_kernel_execution_time {};

	std::chrono::time_point<std::chrono::system_clock> matrix_multiplication_kernel_start_time;
	// MATRIX STRUCT
	Matrix_multiplication *matrix_mult = new Matrix_multiplication();

	const int ROW_X = matrix_mult->ROW_X;
	const int COL_X = matrix_mult->COL_X;
	const int ROW_Y = matrix_mult->ROW_Y;
	const int COL_Y = matrix_mult->COL_Y;
	
	// INITIALIZATION TWO MATRIX
	for (int i = 0; i < ROW_X * COL_X; i++){ matrix_mult->h_X[i] = DATA_TYPE(1);}
	for (int i = 0; i < ROW_Y * COL_Y; i++){ matrix_mult->h_Y[i] = DATA_TYPE(1);}

	// TRANSFER DATA IN MATRIX MULTIPLICATION TO DEVICE 
	matrix_data_transfer_to_device(matrix_mult);
	
	// CALL NAIVE MATRIX MULTIPLICATION KERNEL
	matrix_multiplication_kernel_start_time = std::chrono::system_clock::now();
	call_naive_matrix_multiplication_kernel(matrix_mult);
	naive_matrix_multiplication_kernel_execution_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - matrix_multiplication_kernel_start_time);
		
	// COPY RESULT MATRIX FROM DEVICE TO HOST
	cudaMemcpy(matrix_mult->h_Z, matrix_mult->d_Z, sizeof(DATA_TYPE) * ROW_X * COL_Y, cudaMemcpyDefault);

	// PRINT EXECUTION TIME AND MATRIX
	cout << "Naive Matrix multiplication kernel execution time : " << naive_matrix_multiplication_kernel_execution_time.count() << "ms" << endl;

	// CALL TILED MATRIX MULIPLICATION KERNEL 
	matrix_multiplication_kernel_start_time = std::chrono::system_clock::now();
	call_tiled_matrix_multiplication_kernel(matrix_mult);
	tiled_matrix_multiplication_kernel_execution_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - matrix_multiplication_kernel_start_time);

	// COPY RESULT MATRIX FROM DEVICE TO HOST
	cudaMemcpy(matrix_mult->h_Z, matrix_mult->d_Z, sizeof(DATA_TYPE) * ROW_X * COL_Y, cudaMemcpyDefault);

	// PRINT EXECUTION TIME AND MATRIX
	cout << "Tiled Matrix multiplication kernel execution time : " << tiled_matrix_multiplication_kernel_execution_time.count() << "ms" << endl;

	delete matrix_mult;
	return 0;
}
