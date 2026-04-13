#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "utils.hpp"

inline static int MCD(const unsigned int A, const unsigned int B);
static int euclid(unsigned int A, unsigned int B);

void matmatikj(int ldA, int ldB, int ldC, double *A, double *B, double *C,
               int N1, int N2, int N3);
void matmatblock(int ldA, int ldB, int ldC, double *A, double *B, double *C,
                 int N1, int N2, int N3, int dbA, int dbB, int dbC);
void matmatthread(int ldA, int ldB, int ldC, double *A, double *B, double *C,
                  int N1, int N2, int N3, int dbA, int dbB, int dbC, int NTROW,
                  int NTCOL);


int main() {
  const unsigned int NTROW = 4;
  const unsigned int NTCOL = 4;
  const unsigned int DIMBLOCK = 256;

  const unsigned int N = 8192;
  const unsigned int M = 8192;
  const unsigned int K = 8192;
  const unsigned int LD = 8192;

  double start, end;

  double *h_A; // N x K
  double *h_B; // K x M
  double *h_C; // N x M

  h_A = (double *)malloc(N * LD * sizeof(double));
  h_B = (double *)malloc(K * LD * sizeof(double));
  h_C = (double *)calloc(N * LD, sizeof(double));

  init_mat(h_A, N, K, LD);
  init_mat(h_B, K, M, LD);

  start = get_cur_time();
  matmatthread(LD, LD, LD, h_A, h_B, h_C, N, K, M, DIMBLOCK, DIMBLOCK, DIMBLOCK, NTROW, NTCOL);
  end = get_cur_time();
  printf("Time: %f ms\n", end - start);

  print_mat(h_C, 5, 5, LD);
  free(h_C);
  free(h_B);
  free(h_A);
  return 0;
}

void matmatthread(int ldA, int ldB, int ldC, double *A, double *B, double *C,
                  int N1, int N2, int N3, int dbA, int dbB, int dbC, int NTROW,
                  int NTCOL) {

  const unsigned int NT = NTROW * NTCOL;
  const unsigned int myN1 = N1 / NTROW;
  const unsigned int myN3 = N3 / NTCOL;

  unsigned int myID, IDi, IDj;
  unsigned int start_row, start_col;

  omp_set_num_threads(NT);

  #pragma omp parallel private(myID, IDi, IDj, start_row, start_col)
  {
    myID = omp_get_thread_num();
    IDi = myID / NTCOL;
    IDj = myID % NTCOL;

    start_row = myN1 * IDi;
    start_col = myN3 * IDj;

    matmatblock(ldA, ldB, ldC, &A[start_row * ldA], &B[start_col],
                &C[start_row * ldC + start_col], myN1, N2, myN3, dbA, dbB, dbC);
  }
}

void matmatblock(int ldA, int ldB, int ldC, double *A, double *B, double *C,
                 int N1, int N2, int N3, int dbA, int dbB, int dbC) {
  const unsigned int num_submatrixes_A = N1 / dbA;
  const unsigned int num_submatrixes_B = N3 / dbC;
  const unsigned int num_subsubmatrixes = N2 / dbB;

  unsigned int row_A, col_B, curr_subsubmatrix;
  unsigned int idxA, idxB, idxC;
  unsigned int ii, jj, kk;
  for (ii = 0; ii < num_submatrixes_A; ii++) {
    row_A = ii * dbA;
    for (jj = 0; jj < num_submatrixes_B; jj++) {
      col_B = jj * dbC;
      idxC = row_A * ldC + col_B;
      for (kk = 0; kk < num_subsubmatrixes; kk++) {
        curr_subsubmatrix = kk * dbB;
        idxA = row_A * ldA + curr_subsubmatrix;
        idxB = curr_subsubmatrix * ldB + col_B;
        matmatikj(ldA, ldB, ldC, &A[idxA], &B[idxB], &C[idxC], dbA, dbB, dbC);
      }
    }
  }
}

void matmatikj(int ldA, int ldB, int ldC, double *A, double *B, double *C,
               int N1, int N2, int N3) {
  unsigned int i, j, k;
  for (i = 0; i < N1; i++) {
    for (k = 0; k < N2; k++) {
      for (j = 0; j < N3; j++) {
        C[i * ldC + j] += A[i * ldA + k] * B[k * ldB + j];
      }
    }
  }
}

inline static int MCD(const unsigned int A, const unsigned int B) {
  return euclid(A, B);
}

static int euclid(unsigned int A, unsigned int B) {
  int Q, R;
  while (B != 0) {
    Q = A / B;
    R = A % B;
    A = B;
    B = R;
  }
  return A;
}