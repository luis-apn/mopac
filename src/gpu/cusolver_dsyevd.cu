#include <cuda.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <stdio.h>
#include <stdlib.h>

#if CC12
#define REAL float
#else
#define REAL double
#endif

static void handleError(cudaError_t erro) {
  if (erro != cudaSuccess) {
    printf("%s\n", cudaGetErrorString(erro));
    exit(EXIT_FAILURE);
  }
}

static void handleCusolverError(cusolverStatus_t status, const char *msg) {
  if (status != CUSOLVER_STATUS_SUCCESS) {
    printf("cuSOLVER error (%s): %d\n", msg, (int)status);
  }
}

// ============================================================================
// Driver for cusolverDnDsyevd
// Replaces MagmaDsyevd_Driver1 + MagmaDsyevd_Driver2 in a single call.
//
// This function:
//   1. Creates a cuSOLVER handle
//   2. Copies eigenvecs (host) -> A_dev (device)
//   3. Queries workspace size via cusolverDnDsyevd_bufferSize
//   4. Allocates workspace on device
//   5. Calls cusolverDnDsyevd (compute eigenvectors + eigenvalues)
//   6. Copies results back: A_dev -> eigenvecs, W_dev -> eigvals
//   7. Retrieves devInfo -> info
//   8. Frees device memory and destroys handle
//
// Parameters (all host pointers):
//   n          - matrix dimension
//   eigenvecs  - n×n symmetric matrix (input), eigenvectors (output)
//   lda        - leading dimension (== n)
//   eigvals    - n eigenvalues (output, ascending order)
//   info       - 0 on success, -i if i-th param is wrong, >0 if not converged
// ============================================================================
extern "C" void CusolverDsyevd_Driver(int n, REAL *eigenvecs, int lda,
                                      REAL *eigvals, int *info) {
  cusolverDnHandle_t handle;
  cusolverStatus_t cusolver_status;

  // 1. Create cuSOLVER handle
  cusolver_status = cusolverDnCreate(&handle);
  if (cusolver_status != CUSOLVER_STATUS_SUCCESS) {
    printf("cusolverDnCreate failed\n");
    *info = -999;
    return;
  }

  // 2. Allocate device memory
  size_t size_matrix = sizeof(REAL) * lda * n;
  size_t size_eigvals = sizeof(REAL) * n;

  REAL *A_dev = NULL;
  REAL *W_dev = NULL;
  int *devInfo = NULL;

  handleError(cudaMalloc((void **)&A_dev, size_matrix));
  handleError(cudaMalloc((void **)&W_dev, size_eigvals));
  handleError(cudaMalloc((void **)&devInfo, sizeof(int)));

  // 3. Copy matrix from host to device
  handleError(
      cudaMemcpy(A_dev, eigenvecs, size_matrix, cudaMemcpyHostToDevice));

  // 4. Query workspace size
  int lwork = 0;
  cusolver_status = cusolverDnDsyevd_bufferSize(
      handle,
      CUSOLVER_EIG_MODE_VECTOR, // compute eigenvalues AND eigenvectors
      CUBLAS_FILL_MODE_UPPER,   // upper triangle (matches LAPACK 'u')
      n, A_dev, lda, W_dev, &lwork);
  handleCusolverError(cusolver_status, "cusolverDnDsyevd_bufferSize");

  // 5. Allocate workspace on device
  REAL *work_dev = NULL;
  handleError(cudaMalloc((void **)&work_dev, sizeof(REAL) * lwork));

  // 6. Execute eigenvalue decomposition
  cusolver_status =
      cusolverDnDsyevd(handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
                       n, A_dev, lda, W_dev, work_dev, lwork, devInfo);
  handleCusolverError(cusolver_status, "cusolverDnDsyevd");

  // 7. Copy results back to host
  handleError(
      cudaMemcpy(eigenvecs, A_dev, size_matrix, cudaMemcpyDeviceToHost));
  handleError(cudaMemcpy(eigvals, W_dev, size_eigvals, cudaMemcpyDeviceToHost));
  handleError(cudaMemcpy(info, devInfo, sizeof(int), cudaMemcpyDeviceToHost));

  // 8. Free device memory and destroy handle
  cudaFree(work_dev);
  cudaFree(devInfo);
  cudaFree(W_dev);
  cudaFree(A_dev);
  cusolverDnDestroy(handle);

  return;
}
