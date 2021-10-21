# distutils: language = c++
# distutils: sources = knn.cxx

import numpy as np
cimport numpy as np
import cython

cdef extern from "knn_.h":

    void cpp_knn(const float* points, const size_t npts, const size_t dim, const float* queries, const size_t nqueries, const size_t K, long* indices)

    void cpp_knn_omp(const float* points, const size_t npts, const size_t dim, const float* queries, const size_t nqueries, const size_t K, long* indices)

    void cpp_radius(const float* points, const size_t npts, const size_t dim, const float* queries, const size_t nqueries, const float radius, const size_t max_num, long* indices)

    void cpp_radius_omp(const float* points, const size_t npts, const size_t dim, const float* queries, const size_t nqueries, const float radius, const size_t max_num, long* indices)

    void cpp_knn_dense_batch(const float* batch_data, const size_t batch_size, const size_t npts, const size_t dim, const float* queries, const size_t nqueries, const size_t K, long* batch_indices)

    void cpp_knn_dense_batch_omp(const float* batch_data, const size_t batch_size, const size_t npts, const size_t dim, const float* queries, const size_t nqueries, const size_t K, long* batch_indices)

    void cpp_radius_dense_batch(const float* batch_data, const size_t batch_size, const size_t npts, const size_t dim, const float* queries, const size_t nqueries, const float radius, const size_t max_num, long* batch_indices)

    void cpp_radius_dense_batch_omp(const float* batch_data, const size_t batch_size, const size_t npts, const size_t dim, const float* queries, const size_t nqueries, const float radius, const size_t max_num, long* batch_indices)

    void cpp_knn_sparse_batch(const float* supports, const int* n_supports, const float* queries, const int* n_queries, const size_t n_dims, const size_t batch_size, const size_t K, long* batch_indices)

    void cpp_knn_sparse_batch_omp(const float* supports, const int* n_supports, const float* queries, const int* n_queries, const size_t n_dims, const size_t batch_size, const size_t K, long* batch_indices);

    void cpp_radius_sparse_batch(const float* supports, const int* n_supports, const float* queries, const int* n_queries, const size_t n_dims, const size_t batch_size,  const float radius, const size_t max_num, long* batch_indices)

    void cpp_radius_sparse_batch_omp(const float* supports, const int* n_supports, const float* queries, const int* n_queries, const size_t n_dims, const size_t batch_size,  const float radius, const size_t max_num, long* batch_indices)

    void cpp_knn_batch_distance_pick(const float* batch_data, const size_t batch_size, const size_t npts, const size_t dim, float* queries, const size_t nqueries, const size_t K, long* batch_indices)
        
    void cpp_knn_batch_distance_pick_omp(const float* batch_data, const size_t batch_size, const size_t npts, const size_t dim, float* batch_queries, const size_t nqueries, const size_t K, long* batch_indices)


def knn(pts, queries, K, omp=False):

    # define shape parameters
    cdef int npts
    cdef int dim
    cdef int K_cpp
    cdef int nqueries

    # define tables
    cdef np.ndarray[np.float32_t, ndim=2] pts_cpp
    cdef np.ndarray[np.float32_t, ndim=2] queries_cpp
    cdef np.ndarray[np.int64_t, ndim=2] indices_cpp

    # set shape values
    npts = pts.shape[0]
    nqueries = queries.shape[0]
    dim = pts.shape[1]
    K_cpp = K

    # create indices tensor
    indices = np.zeros((queries.shape[0], K), dtype=np.int64)

    pts_cpp = np.ascontiguousarray(pts, dtype=np.float32)
    queries_cpp = np.ascontiguousarray(queries, dtype=np.float32)
    indices_cpp = indices

    # normal estimation
    if omp:
        cpp_knn_omp(<float*> pts_cpp.data, npts, dim, <float*> queries_cpp.data, nqueries, K_cpp, <long*> indices_cpp.data)
    else:
        cpp_knn(<float*> pts_cpp.data, npts, dim, <float*> queries_cpp.data, nqueries, K_cpp, <long*> indices_cpp.data)

    return indices


def radius(pts, queries, radius, max_num, omp=False):

    # define shape parameters
    cdef int npts
    cdef int dim
    cdef int max_num_cpp
    cdef float radius_cpp
    cdef int nqueries

    # define tables
    cdef np.ndarray[np.float32_t, ndim=2] pts_cpp
    cdef np.ndarray[np.float32_t, ndim=2] queries_cpp
    cdef np.ndarray[np.int64_t, ndim=2] indices_cpp

    # set shape values
    npts = pts.shape[0]
    nqueries = queries.shape[0]
    dim = pts.shape[1]
    max_num_cpp = max_num
    radius_cpp = radius

    # create indices tensor
    indices = -np.ones((queries.shape[0], max_num), dtype=np.int64)

    pts_cpp = np.ascontiguousarray(pts, dtype=np.float32)
    queries_cpp = np.ascontiguousarray(queries, dtype=np.float32)
    indices_cpp = indices

    # normal estimation
    if omp:
        cpp_radius_omp(<float*> pts_cpp.data, npts, dim, <float*> queries_cpp.data, nqueries, radius_cpp, max_num_cpp, <long*> indices_cpp.data)
    else:
        cpp_radius(<float*> pts_cpp.data, npts, dim, <float*> queries_cpp.data, nqueries, radius_cpp, max_num_cpp, <long*> indices_cpp.data)

    return indices


def knn_dense_batch(pts, queries, K, omp=False):

    # define shape parameters
    cdef int batch_size
    cdef int npts
    cdef int nqueries
    cdef int K_cpp
    cdef int dim

    # define tables
    cdef np.ndarray[np.float32_t, ndim=3] pts_cpp
    cdef np.ndarray[np.float32_t, ndim=3] queries_cpp
    cdef np.ndarray[np.int64_t, ndim=3] indices_cpp

    # set shape values
    batch_size = pts.shape[0]
    npts = pts.shape[1]
    dim = pts.shape[2]
    nqueries = queries.shape[1]
    K_cpp = K

    # create indices tensor
    indices = np.zeros((pts.shape[0], queries.shape[1], K), dtype=np.int64)

    pts_cpp = np.ascontiguousarray(pts, dtype=np.float32)
    queries_cpp = np.ascontiguousarray(queries, dtype=np.float32)
    indices_cpp = indices

    # normal estimation
    if omp:
        cpp_knn_dense_batch_omp(<float*> pts_cpp.data, batch_size, npts, dim, <float*> queries_cpp.data, nqueries, K_cpp, <long*> indices_cpp.data)
    else:
        cpp_knn_dense_batch(<float*> pts_cpp.data, batch_size, npts, dim, <float*> queries_cpp.data, nqueries, K_cpp, <long*> indices_cpp.data)

    return indices


def radius_dense_batch(pts, queries, radius, max_num, omp=False):

    # define shape parameters
    cdef int batch_size
    cdef int npts
    cdef int nqueries
    cdef float radius_cpp
    cdef int max_num_cpp
    cdef int dim

    # define tables
    cdef np.ndarray[np.float32_t, ndim=3] pts_cpp
    cdef np.ndarray[np.float32_t, ndim=3] queries_cpp
    cdef np.ndarray[np.int64_t, ndim=3] indices_cpp

    # set shape values
    batch_size = pts.shape[0]
    npts = pts.shape[1]
    dim = pts.shape[2]
    nqueries = queries.shape[1]
    radius_cpp = radius
    max_num_cpp = max_num

    # create indices tensor
    indices = np.zeros((pts.shape[0], queries.shape[1], max_num), dtype=np.int64)

    pts_cpp = np.ascontiguousarray(pts, dtype=np.float32)
    queries_cpp = np.ascontiguousarray(queries, dtype=np.float32)
    indices_cpp = indices

    # normal estimation
    if omp:
        cpp_radius_dense_batch_omp(<float*> pts_cpp.data, batch_size, npts, dim, <float*> queries_cpp.data, nqueries, radius_cpp, max_num_cpp, <long*> indices_cpp.data)
    else:
        cpp_radius_dense_batch(<float*> pts_cpp.data, batch_size, npts, dim, <float*> queries_cpp.data, nqueries, radius_cpp, max_num_cpp, <long*> indices_cpp.data)

    return indices


def knn_sparse_batch(supports, queries, s_batch, q_batch, K, omp=False):

    # define shape parameters
    cdef int batch_size
    cdef int n_dims
    cdef int K_cpp

    # define tables
    cdef np.ndarray[np.float32_t, ndim=2] supports_cpp
    cdef np.ndarray[np.int32_t, ndim=1] n_supports_cpp
    cdef np.ndarray[np.float32_t, ndim=2] queries_cpp
    cdef np.ndarray[np.int32_t, ndim=1] n_queries_cpp
    cdef np.ndarray[np.int64_t, ndim=2] indices_cpp

    # set shape values
    n_dims = supports.shape[-1]
    K_cpp = K

    # create indices tensor
    _, n_supports = np.unique(s_batch, return_counts=True)
    _, n_queries = np.unique(q_batch, return_counts=True)
    indices = -np.ones((queries.shape[0], K), dtype=np.int64)
    batch_size = len(n_supports)

    supports_cpp = np.ascontiguousarray(supports, dtype=np.float32)
    n_supports_cpp = np.ascontiguousarray(n_supports, dtype=np.int32)
    queries_cpp = np.ascontiguousarray(queries, dtype=np.float32)
    n_queries_cpp = np.ascontiguousarray(n_queries, dtype=np.int32)
    indices_cpp = indices


    # normal estimation
    if omp:
        cpp_knn_sparse_batch_omp(<float*> supports_cpp.data, <int*> n_supports_cpp.data, <float*> queries_cpp.data, <int*> n_queries_cpp.data, n_dims, batch_size, K_cpp, <long*> indices_cpp.data)
    else:
        cpp_knn_sparse_batch(<float*> supports_cpp.data, <int*> n_supports_cpp.data, <float*> queries_cpp.data, <int*> n_queries_cpp.data, n_dims, batch_size, K_cpp, <long*> indices_cpp.data)

    return indices


def radius_sparse_batch(supports, queries, s_batch, q_batch, radius, max_num, omp=False):

    # define shape parameters
    cdef int batch_size
    cdef int n_dims
    cdef float radius_cpp
    cdef int max_num_cpp


    # define tables
    cdef np.ndarray[np.float32_t, ndim=2] supports_cpp
    cdef np.ndarray[np.int32_t, ndim=1] n_supports_cpp
    cdef np.ndarray[np.float32_t, ndim=2] queries_cpp
    cdef np.ndarray[np.int32_t, ndim=1] n_queries_cpp
    cdef np.ndarray[np.int64_t, ndim=2] indices_cpp

    # set shape values
    n_dims = supports.shape[-1]
    radius_cpp = radius
    max_num_cpp = max_num

    # create indices tensor
    _, n_supports = np.unique(s_batch, return_counts=True)
    _, n_queries = np.unique(q_batch, return_counts=True)
    indices = -np.ones((queries.shape[0], max_num), dtype=np.int64)
    batch_size = len(n_supports)

    supports_cpp = np.ascontiguousarray(supports, dtype=np.float32)
    n_supports_cpp = np.ascontiguousarray(n_supports, dtype=np.int32)
    queries_cpp = np.ascontiguousarray(queries, dtype=np.float32)
    n_queries_cpp = np.ascontiguousarray(n_queries, dtype=np.int32)
    indices_cpp = indices


    # normal estimation
    if omp:
        cpp_radius_sparse_batch_omp(<float*> supports_cpp.data, <int*> n_supports_cpp.data, <float*> queries_cpp.data, <int*> n_queries_cpp.data, n_dims, batch_size, radius_cpp, max_num_cpp, <long*> indices_cpp.data)
    else:
        cpp_radius_sparse_batch(<float*> supports_cpp.data, <int*> n_supports_cpp.data, <float*> queries_cpp.data, <int*> n_queries_cpp.data, n_dims, batch_size, radius_cpp, max_num_cpp, <long*> indices_cpp.data)

    return indices


def knn_batch_distance_pick(pts, nqueries, K, omp=False):

    # define shape parameters
    cdef int batch_size
    cdef int npts
    cdef int nqueries_cpp
    cdef int K_cpp
    cdef int dim

    # define tables
    cdef np.ndarray[np.float32_t, ndim=3] pts_cpp
    cdef np.ndarray[np.float32_t, ndim=3] queries_cpp
    cdef np.ndarray[np.int64_t, ndim=3] indices_cpp

    # set shape values
    batch_size = pts.shape[0]
    npts = pts.shape[1]
    dim = pts.shape[2]
    nqueries_cpp = nqueries
    K_cpp = K

    # create indices tensor
    indices = np.zeros((pts.shape[0], nqueries, K), dtype=np.long)
    queries = np.zeros((pts.shape[0], nqueries, dim), dtype=np.float32)

    pts_cpp = np.ascontiguousarray(pts, dtype=np.float32)
    queries_cpp = np.ascontiguousarray(queries, dtype=np.float32)
    indices_cpp = indices

    if omp:
        cpp_knn_batch_distance_pick_omp(<float*> pts_cpp.data, batch_size, npts, dim, <float*> queries_cpp.data, nqueries, K_cpp, <long*> indices_cpp.data)
    else:
        cpp_knn_batch_distance_pick(<float*> pts_cpp.data, batch_size, npts, dim, <float*> queries_cpp.data, nqueries, K_cpp, <long*> indices_cpp.data)

    return indices, queries