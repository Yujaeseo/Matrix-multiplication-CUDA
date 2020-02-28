#include <iostream>
#include <chrono>
#include <random>
#include <eigen/Sparse>
using namespace std;
using namespace Eigen;

#define DATA_TYPE int
typedef struct Matrix_multiplication
{
	Matrix_multiplication(): ROW_X(4096), COL_X(4096), ROW_Y(4096), COL_Y(4096){initialize();}
	//Matrix_multiplication(): ROW_X(3999), COL_X(3999), ROW_Y(3999), COL_Y(3999){initialize();}
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
		h_Z = new DATA_TYPE[COL_X * ROW_Y];
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

__global__
void tiled_matrix_multiplication
(
	DATA_TYPE *d_X,
	DATA_TYPE *d_Y,
	DATA_TYPE *d_Z,
	int width,
	int width_Z
){
	__shared__ DATA_TYPE X_tile[32][32];
	__shared__ DATA_TYPE Y_tile[32][32];
	
	int t_row = threadIdx.y; 
	int t_col = threadIdx.x;
	int width_tile = blockDim.y;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	DATA_TYPE acc_result = 0;

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

void call_naive_matrix_multiplication_kernel(Matrix_multiplication* matrix_mult){
	const int THREADS_PER_BLOCK = 1024;
	const int THREAD_BLOCKS = (matrix_mult->ROW_X * matrix_mult->COL_Y + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

	naive_matrix_multiplication<<<THREAD_BLOCKS, THREADS_PER_BLOCK>>>(matrix_mult->d_X, matrix_mult->d_Y, matrix_mult->d_Z, matrix_mult->COL_X, matrix_mult->COL_Y, matrix_mult->ROW_X * matrix_mult->COL_Y);
	cudaDeviceSynchronize();
}

void call_tiled_matrix_multiplication_kernel(Matrix_multiplication* matrix_mult){
	// const int THREADS_PER_BLOCK = 1024;
	// const int THREAD_BLOCKS = (matrix->ROW_X * matrix->COL_Y + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	dim3 blocks((matrix_mult->ROW_X + 31) /32, (matrix_mult->COL_Y + 31) /32);
	dim3 threads(32, 32);

	tiled_matrix_multiplication<<<blocks, threads>>>(matrix_mult->d_X, matrix_mult->d_Y, matrix_mult->d_Z, matrix_mult->COL_X, matrix_mult->COL_Y);
	cudaDeviceSynchronize();
}

void check_matrix_multiplication_result(const int trial_num){
	unsigned int seed = chrono::system_clock::now().time_since_epoch().count();
	mt19937 gen(seed);
	uniform_int_distribution<mt19937::result_type> dist_for_element(1,50);
	uniform_int_distribution<mt19937::result_type> dist_for_dim(3000, 4000);

	for(int trial_count = 0; trial_count < trial_num; trial_count++){
		int ROW_X, COL_X, ROW_Y, COL_Y;
		
		// ROW_X = dist_for_dim(gen);
		// COL_X = ROW_Y = dist_for_dim(gen);
		// COL_Y = dist_for_dim(gen);

		ROW_X = ROW_Y = COL_X = COL_Y = 1024;
		// MATRIX MULTIPLICATION STRUCT FOR NAIVE KERNEL
		Matrix_multiplication *test_matrix_mult = new Matrix_multiplication(ROW_X, COL_X, ROW_Y, COL_Y);

		// INITIALIZATION TWO MATRIX
		for (int i = 0; i < ROW_X * COL_X; i++){ test_matrix_mult->h_X[i] = DATA_TYPE(dist_for_element(gen));}
		for (int i = 0; i < ROW_Y * COL_Y; i++){ test_matrix_mult->h_Y[i] = DATA_TYPE(dist_for_element(gen));}
		// MATRIX MULTIPLICATION STRUCT FOR TILED KERNEL
		Matrix_multiplication *test_matrix_mult2 = new Matrix_multiplication(*test_matrix_mult);

		matrix_data_transfer_to_device(test_matrix_mult);
		matrix_data_transfer_to_device(test_matrix_mult2);

		call_naive_matrix_multiplication_kernel(test_matrix_mult);
		call_tiled_matrix_multiplication_kernel(test_matrix_mult2);

		cudaMemcpy(test_matrix_mult->h_Z, test_matrix_mult->d_Z, sizeof(DATA_TYPE) * ROW_X * COL_Y, cudaMemcpyDefault);
		cudaMemcpy(test_matrix_mult2->h_Z, test_matrix_mult2->d_Z, sizeof(DATA_TYPE) * ROW_X * COL_Y, cudaMemcpyDefault);
		
		// cout << "=== Check pointer ===" << endl;
		// cout << test_matrix_mult->h_X << endl;
		// cout << test_matrix_mult->h_Y << endl;
		// cout << test_matrix_mult->h_Z << endl;
		// cout << test_matrix_mult->d_X << endl;
		// cout << test_matrix_mult->d_Y << endl;
		// cout << test_matrix_mult->d_Z << endl << endl << endl;

		// cout << test_matrix_mult2->h_X << endl;
		// cout << test_matrix_mult2->h_Y << endl;
		// cout << test_matrix_mult2->h_Z << endl;
		// cout << test_matrix_mult2->d_X << endl;
		// cout << test_matrix_mult2->d_Y << endl;
		// cout << test_matrix_mult2->d_Z << endl;

		Matrix<int,-1,-1,RowMajor> matrix_dense_X, matrix_dense_Y, matrix_dense_Z; 
		
		matrix_dense_X = Map<Matrix<int,-1,-1,RowMajor>, 0, OuterStride<>>(test_matrix_mult->h_X, ROW_X, COL_X, OuterStride<>(COL_X));
		matrix_dense_Y = Map<Matrix<int,-1,-1,RowMajor>, 0, OuterStride<>>(test_matrix_mult->h_Y, ROW_Y, COL_Y, OuterStride<>(COL_Y));

		matrix_dense_Z = matrix_dense_X * matrix_dense_Y;

		bool check_result = true;
		for (int i = 0; i < ROW_X * COL_Y; i++){
			//cout << test_matrix_mult->h_Z[i] << " " << matrix_dense_Z(i/COL_Y,i%COL_Y) << endl;
			if (matrix_dense_Z(i/COL_Y,i%COL_Y) != test_matrix_mult->h_Z[i]){
				check_result = false;
				break;
			}
		}

		if(check_result) cout << "Trial num " << trial_count + 1 << " : identical matrix multiplication result(naive, eigen)" << endl;
		else cout << "Trial num " << trial_count + 1 << " : not identical matrix multiplication result(naive, eigen)" << endl;
		
		check_result = true;

		for (int i = 0; i < ROW_X * COL_Y; i++){
			if (test_matrix_mult->h_Z[i] != test_matrix_mult2->h_Z[i]){
				check_result = false;
				break;
			} 
		}

		if (check_result) cout << "Trial num " << trial_count + 1  <<" : identical matrix result(naive, tiled)" << endl;
		else cout << "Trial num " << trial_count + 1 << " : not identical matrix multiplication result(naive, tiled)" << endl;

		delete test_matrix_mult;
		delete test_matrix_mult2;
	}
}

int main (){

	std::chrono::milliseconds naive_matrix_multiplication_kernel_execution_time {};
	std::chrono::milliseconds tiled_matrix_multiplication_kernel_execution_time {};

	std::chrono::time_point<std::chrono::system_clock> matrix_multiplication_kernel_start_time;
	
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
	cout << "naive Matrix multiplication kernel execution time : " << naive_matrix_multiplication_kernel_execution_time.count() << "ms" << endl;
	// COPY RESULT MATRIX FROM DEVICE TO HOST
	cudaMemcpy(matrix_mult->h_Z, matrix_mult->d_Z, sizeof(DATA_TYPE) * ROW_X * COL_Y, cudaMemcpyDefault);
	for (int i = 0 ; i <100; i++){
		cout << matrix_mult->h_Z[i] << " ";
	}
	// CALL TILED MATRIX MULIPLICATION KERNEL 
	matrix_multiplication_kernel_start_time = std::chrono::system_clock::now();
	call_tiled_matrix_multiplication_kernel(matrix_mult);
	tiled_matrix_multiplication_kernel_execution_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - matrix_multiplication_kernel_start_time);
	cout << "tiled Matrix multiplication kernel execution time : " << tiled_matrix_multiplication_kernel_execution_time.count() << "ms" << endl;
	// COPY RESULT MATRIX FROM DEVICE TO HOST
	cudaMemcpy(matrix_mult->h_Z, matrix_mult->d_Z, sizeof(DATA_TYPE) * ROW_X * COL_Y, cudaMemcpyDefault);

	for (int i = 0 ; i <100; i++){
		cout << matrix_mult->h_Z[i] << " ";
	}

	cout << "Matrix multiplication test :" << endl;
	check_matrix_multiplication_result(10);

	delete matrix_mult;
	return 0;
}
