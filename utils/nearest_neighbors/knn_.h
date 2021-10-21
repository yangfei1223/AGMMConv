#include <cstdlib>

// knn search
void cpp_knn(const float* points, const size_t npts, const size_t dim, const float* queries, const size_t nqueries, const size_t K, long* indices);
// knn search omp
void cpp_knn_omp(const float* points, const size_t npts, const size_t dim, const float* queries, const size_t nqueries, const size_t K, long* indices);
// radius search
void cpp_radius(const float* points, const size_t npts, const size_t dim, const float* queries, const size_t nqueries, const float radius, const size_t max_num, long* indices);
// radius search omp
void cpp_radius_omp(const float* points, const size_t npts, const size_t dim, const float* queries, const size_t nqueries, const float radius, const size_t max_num, long* indices);
// dense batch knn search
void cpp_knn_dense_batch(const float* batch_data, const size_t batch_size, const size_t npts, const size_t dim, const float* queries, const size_t nqueries, const size_t K, long* batch_indices);
// dense batch knn search omp
void cpp_knn_dense_batch_omp(const float* batch_data, const size_t batch_size, const size_t npts, const size_t dim, const float* queries, const size_t nqueries, const size_t K, long* batch_indices);
// dense batch radius search
void cpp_radius_dense_batch(const float* batch_data, const size_t batch_size, const size_t npts, const size_t dim, const float* queries, const size_t nqueries, const float radius, const size_t max_num, long* batch_indices);
// dense batch radius search omp
void cpp_radius_dense_batch_omp(const float* batch_data, const size_t batch_size, const size_t npts, const size_t dim, const float* queries, const size_t nqueries, const float radius, const size_t max_num, long* batch_indices);
// sparse batch knn search
void cpp_knn_sparse_batch(const float* supports, const int* n_supports, const float* queries, const int* n_queries, const size_t n_dims, const size_t batch_size, const size_t K, long* batch_indices);
// sparse batch knn search omp
void cpp_knn_sparse_batch_omp(const float* supports, const int* n_supports, const float* queries, const int* n_queries, const size_t n_dims, const size_t batch_size, const size_t K, long* batch_indices);
// sparse batch radius search
void cpp_radius_sparse_batch(const float* supports, const int* n_supports, const float* queries, const int* n_queries, const size_t n_dims, const size_t batch_size,  const float radius, const size_t max_num, long* batch_indices);
// sparse batch radius search omp
void cpp_radius_sparse_batch_omp(const float* supports, const int* n_supports, const float* queries, const int* n_queries, const size_t n_dims, const size_t batch_size,  const float radius, const size_t max_num, long* batch_indices);

// I don't know what the following two functions are used to do.
void cpp_knn_batch_distance_pick(const float* batch_data, const size_t batch_size, const size_t npts, const size_t dim,  float* queries, const size_t nqueries, const size_t K, long* batch_indices);

void cpp_knn_batch_distance_pick_omp(const float* batch_data, const size_t batch_size, const size_t npts, const size_t dim,  float* batch_queries, const size_t nqueries, const size_t K, long* batch_indices);