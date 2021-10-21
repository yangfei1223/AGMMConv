
#include "knn_.h"
#include "nanoflann.hpp"
using namespace nanoflann;

#include "KDTreeTableAdaptor.h"

#include <ctime>
#include <cstdlib>
#include <iostream>
#include <omp.h>

#include <vector>
#include <random>
#include <algorithm>
#include <iterator>

using namespace std;


void cpp_knn(const float* points, const size_t npts, const size_t dim,  const float* queries, const size_t nqueries, const size_t K, long* indices)
{
	// create the kdtree
	typedef KDTreeTableAdaptor< float, float> KDTree;
	KDTree mat_index(npts, dim, points, 10);
	mat_index.index->buildIndex();

	std::vector<float> out_dists_sqr(K);
	std::vector<size_t> out_ids(K);

	// iterate over the points
	for(size_t i=0; i<nqueries; i++)
	{
		nanoflann::KNNResultSet<float> resultSet(K);
		resultSet.init(&out_ids[0], &out_dists_sqr[0] );
		mat_index.index->findNeighbors(resultSet, &queries[i*dim], nanoflann::SearchParams(10));
		for(size_t j=0; j<K; j++)
		{
			indices[i*K+j] = long(out_ids[j]);
		}
	}
}


void cpp_knn_omp(const float* points, const size_t npts, const size_t dim, const float* queries, const size_t nqueries, const size_t K, long* indices)
{
	// create the kdtree
	typedef KDTreeTableAdaptor< float, float> KDTree;
	KDTree mat_index(npts, dim, points, 10);
	mat_index.index->buildIndex();

	// iterate over the points
    # pragma omp parallel for
	for(size_t i=0; i<nqueries; i++)
	{
		std::vector<size_t> out_ids(K);
		std::vector<float> out_dists_sqr(K);
		nanoflann::KNNResultSet<float> resultSet(K);
		resultSet.init(&out_ids[0], &out_dists_sqr[0] );
		mat_index.index->findNeighbors(resultSet, &queries[i*dim], nanoflann::SearchParams(10));
		for(size_t j=0; j<K; j++)
		{
			indices[i*K+j] = long(out_ids[j]);
		}
	}
}


void cpp_radius(const float* points, const size_t npts, const size_t dim, const float* queries, const size_t nqueries, const float radius, const size_t max_num, long* indices)
{
    // create the kdtree
	typedef KDTreeTableAdaptor< float, float> KDTree;
	KDTree mat_index(npts, dim, points, 10);
	mat_index.index->buildIndex();

    std::vector<std::pair<size_t, float> > indices_dists;

	// iterate over the points
	for(size_t i=0; i<nqueries; i++)
	{
		nanoflann::RadiusResultSet<float> resultSet(radius, indices_dists);
		mat_index.index->findNeighbors(resultSet, &queries[i*dim], nanoflann::SearchParams(10));
		for(size_t j=0; j<indices_dists.size() && j < max_num; j++)
		{
            indices[i*max_num+j] = long(indices_dists[j].first);
		}
	}
}


void cpp_radius_omp(const float* points, const size_t npts, const size_t dim, const float* queries, const size_t nqueries, const float radius, const size_t max_num, long* indices)
{
    // create the kdtree
	typedef KDTreeTableAdaptor< float, float> KDTree;
	KDTree mat_index(npts, dim, points, 10);
	mat_index.index->buildIndex();

	// iterate over the points
    # pragma omp parallel for
	for(size_t i=0; i<nqueries; i++)
	{
        std::vector<std::pair<size_t, float> > indices_dists;
		nanoflann::RadiusResultSet<float> resultSet(radius, indices_dists);
		mat_index.index->findNeighbors(resultSet, &queries[i*dim], nanoflann::SearchParams(10));
		for(size_t j=0; j<indices_dists.size() && j < max_num; j++)
		{
            indices[i*max_num+j] = long(indices_dists[j].first);
		}
	}
}


void cpp_knn_dense_batch(const float* batch_data, const size_t batch_size, const size_t npts, const size_t dim, const float* queries, const size_t nqueries, const size_t K, long* batch_indices)
{
	for(size_t bid=0; bid < batch_size; bid++)
	{
		const float* points = &batch_data[bid*npts*dim];
		long* indices = &batch_indices[bid*nqueries*K];

		// create the kdtree
		typedef KDTreeTableAdaptor< float, float> KDTree;
		KDTree mat_index(npts, dim, points, 10);
		
		mat_index.index->buildIndex();

		std::vector<float> out_dists_sqr(K);
		std::vector<size_t> out_ids(K);

		// iterate over the points
		for(size_t i=0; i<nqueries; i++)
		{
			nanoflann::KNNResultSet<float> resultSet(K);
			resultSet.init(&out_ids[0], &out_dists_sqr[0] );
			mat_index.index->findNeighbors(resultSet, &queries[bid*nqueries*dim + i*dim], nanoflann::SearchParams(10));
			for(size_t j=0; j<K; j++)
			{
				indices[i*K+j] = long(out_ids[j]);
			}
		}
	}
}


void cpp_knn_dense_batch_omp(const float* batch_data, const size_t batch_size, const size_t npts, const size_t dim, const float* queries, const size_t nqueries, const size_t K, long* batch_indices)
{
    # pragma omp parallel for
	for(size_t bid=0; bid < batch_size; bid++)
	{
		const float* points = &batch_data[bid*npts*dim];
		long* indices = &batch_indices[bid*nqueries*K];

		// create the kdtree
		typedef KDTreeTableAdaptor< float, float> KDTree;
		KDTree mat_index(npts, dim, points, 10);

		mat_index.index->buildIndex();

		std::vector<float> out_dists_sqr(K);
		std::vector<size_t> out_ids(K);

		// iterate over the points
		for(size_t i=0; i<nqueries; i++)
		{
			nanoflann::KNNResultSet<float> resultSet(K);
			resultSet.init(&out_ids[0], &out_dists_sqr[0] );
			mat_index.index->findNeighbors(resultSet, &queries[bid*nqueries*dim + i*dim], nanoflann::SearchParams(10));
			for(size_t j=0; j<K; j++)
			{
				indices[i*K+j] = long(out_ids[j]);
			}
		}
	}
}


void cpp_radius_dense_batch(const float* batch_data, const size_t batch_size, const size_t npts, const size_t dim, const float* queries, const size_t nqueries, const float radius, const size_t max_num, long* batch_indices)
{
	for(size_t bid=0; bid < batch_size; bid++)
	{
		const float* points = &batch_data[bid*npts*dim];
		long* indices = &batch_indices[bid*nqueries*max_num];

		// create the kdtree
		typedef KDTreeTableAdaptor< float, float> KDTree;
		KDTree mat_index(npts, dim, points, 10);

		mat_index.index->buildIndex();

        std::vector<std::pair<size_t, float> > indices_dists;

		// iterate over the points
		for(size_t i=0; i<nqueries; i++)
		{
            nanoflann::RadiusResultSet<float> resultSet(radius, indices_dists);
			mat_index.index->findNeighbors(resultSet, &queries[bid*nqueries*dim + i*dim], nanoflann::SearchParams(10));
			for(size_t j=0; j<indices_dists.size() && j<max_num; j++)
			{
				indices[i*max_num+j] = long(indices_dists[j].first);
			}
		}
	}
}


void cpp_radius_dense_batch_omp(const float* batch_data, const size_t batch_size, const size_t npts, const size_t dim, const float* queries, const size_t nqueries, const float radius, const size_t max_num, long* batch_indices)
{
    # pragma omp parallel for
	for(size_t bid=0; bid < batch_size; bid++)
	{
		const float* points = &batch_data[bid*npts*dim];
		long* indices = &batch_indices[bid*nqueries*max_num];

		// create the kdtree
		typedef KDTreeTableAdaptor< float, float> KDTree;
		KDTree mat_index(npts, dim, points, 10);

		mat_index.index->buildIndex();

        std::vector<std::pair<size_t, float> > indices_dists;

		// iterate over the points
		for(size_t i=0; i<nqueries; i++)
		{
            nanoflann::RadiusResultSet<float> resultSet(radius, indices_dists);
			mat_index.index->findNeighbors(resultSet, &queries[bid*nqueries*dim + i*dim], nanoflann::SearchParams(10));
			for(size_t j=0; j<indices_dists.size() && j<max_num; j++)
			{
				indices[i*max_num+j] = long(indices_dists[j].first);
			}
		}
	}
}


void cpp_knn_sparse_batch(const float* supports, const int* n_supports, const float* queries, const int* n_queries, const size_t n_dims, const size_t batch_size, const size_t K, long* batch_indices)
{
	for(size_t b=0; b < batch_size; b++)
	{
        size_t s_start = 0, q_start = 0;
        for (size_t i=0; i < b; i++)
        {
            s_start += n_supports[i];
            q_start += n_queries[i];
        }
		const float* support_points = &supports[s_start*n_dims];
		long* indices = &batch_indices[q_start*K];

		// create the kdtree
		typedef KDTreeTableAdaptor< float, float> KDTree;
		KDTree mat_index(n_supports[b], n_dims, support_points, 10);

		mat_index.index->buildIndex();

        std::vector<float> out_dists_sqr(K);
		std::vector<size_t> out_ids(K);

		// iterate over the points
		for(size_t i=0; i<n_queries[b]; i++)
		{
            nanoflann::KNNResultSet<float> resultSet(K);
            resultSet.init(&out_ids[0], &out_dists_sqr[0] );
			mat_index.index->findNeighbors(resultSet, &queries[q_start*n_dims + i*n_dims], nanoflann::SearchParams(10));
			for(size_t j=0; j<K; j++)
			{
				indices[i*K+j] = long(out_ids[j] + s_start);
			}
		}
	}
}


void cpp_knn_sparse_batch_omp(const float* supports, const int* n_supports, const float* queries, const int* n_queries, const size_t n_dims, const size_t batch_size, const size_t K, long* batch_indices)
{
    # pragma omp parallel for
	for(size_t b=0; b < batch_size; b++)
	{
        size_t s_start = 0, q_start = 0;
        for (size_t i=0; i < b; i++)
        {
            s_start += n_supports[i];
            q_start += n_queries[i];
        }
		const float* support_points = &supports[s_start*n_dims];
		long* indices = &batch_indices[q_start*K];

		// create the kdtree
		typedef KDTreeTableAdaptor< float, float> KDTree;
		KDTree mat_index(n_supports[b], n_dims, support_points, 10);

		mat_index.index->buildIndex();

        std::vector<float> out_dists_sqr(K);
		std::vector<size_t> out_ids(K);

		// iterate over the points
		for(size_t i=0; i<n_queries[b]; i++)
		{
            nanoflann::KNNResultSet<float> resultSet(K);
            resultSet.init(&out_ids[0], &out_dists_sqr[0] );
			mat_index.index->findNeighbors(resultSet, &queries[q_start*n_dims + i*n_dims], nanoflann::SearchParams(10));
			for(size_t j=0; j<K; j++)
			{
				indices[i*K+j] = long(out_ids[j] + s_start);
			}
		}
	}
}


void cpp_radius_sparse_batch(const float* supports, const int* n_supports, const float* queries, const int* n_queries, const size_t n_dims, const size_t batch_size,  const float radius, const size_t max_num, long* batch_indices)
{
	for(size_t b=0; b < batch_size; b++)
	{
        size_t s_start = 0, q_start = 0;
        for (size_t i=0; i < b; i++)
        {
            s_start += n_supports[i];
            q_start += n_queries[i];
        }
		const float* support_points = &supports[s_start*n_dims];
		long* indices = &batch_indices[q_start*max_num];

		// create the kdtree
		typedef KDTreeTableAdaptor< float, float> KDTree;
		KDTree mat_index(n_supports[b], n_dims, support_points, 10);

		mat_index.index->buildIndex();

        std::vector<std::pair<size_t, float> > indices_dists;

		// iterate over the points
		for(size_t i=0; i<n_queries[b]; i++)
		{
            nanoflann::RadiusResultSet<float> resultSet(radius, indices_dists);
			mat_index.index->findNeighbors(resultSet, &queries[q_start*n_dims + i*n_dims], nanoflann::SearchParams(10));
			for(size_t j=0; j<indices_dists.size() && j<max_num; j++)
			{
				indices[i*max_num+j] = long(indices_dists[j].first + s_start);
			}
		}
	}
}


void cpp_radius_sparse_batch_omp(const float* supports, const int* n_supports, const float* queries, const int* n_queries, const size_t n_dims, const size_t batch_size,  const float radius, const size_t max_num, long* batch_indices)
{
    # pragma omp parallel for
	for(size_t b=0; b < batch_size; b++)
	{
        size_t s_start = 0, q_start = 0;
        for (size_t i=0; i < b; i++)
        {
            s_start += n_supports[i];
            q_start += n_queries[i];
        }
		const float* support_points = &supports[s_start*n_dims];
		long* indices = &batch_indices[q_start*max_num];

		// create the kdtree
		typedef KDTreeTableAdaptor< float, float> KDTree;
		KDTree mat_index(n_supports[b], n_dims, support_points, 10);

		mat_index.index->buildIndex();

        std::vector<std::pair<size_t, float> > indices_dists;

		// iterate over the points
		for(size_t i=0; i<n_queries[b]; i++)
		{
            nanoflann::RadiusResultSet<float> resultSet(radius, indices_dists);
			mat_index.index->findNeighbors(resultSet, &queries[q_start*n_dims + i*n_dims], nanoflann::SearchParams(10));
			for(size_t j=0; j<indices_dists.size() && j<max_num; j++)
			{
				indices[i*max_num+j] = long(indices_dists[j].first + s_start);
			}
		}
	}
}



void cpp_knn_batch_distance_pick(const float* batch_data, const size_t batch_size, const size_t npts, const size_t dim, float* batch_queries, const size_t nqueries, const size_t K, long* batch_indices)
{
	mt19937 mt_rand(time(0));
	for(size_t bid=0; bid < batch_size; bid++)
	{
		const float* points = &batch_data[bid*npts*dim];
		// long* indices = &batch_indices[bid*nqueries*K];
		// float* queries = &batch_queries[bid*nqueries*dim];

		// create the kdtree
		typedef KDTreeTableAdaptor< float, float> KDTree;
		KDTree tree(npts, dim, points, 10);
		tree.index->buildIndex();

		vector<int> used(npts, 0);
		int current_id = 0;
		for(size_t ptid=0; ptid<nqueries; ptid++)
		{
			// get the possible points
			vector<size_t> possible_ids;
			while(possible_ids.size() == 0)
			{
				for(size_t i=0; i<npts; i++)
				{
					if(used[i] == current_id)
					{
						possible_ids.push_back(i);
					}
				}
				if(possible_ids.size() == 0)
				{
					current_id = *std::min_element(std::begin(used), std::end(used));
				}
			}

			// get an index
			size_t index = possible_ids[mt_rand()%possible_ids.size()];

			// create the query
			vector<float> query(3);
			for(size_t i=0; i<dim; i++)
			{
				query[i] = batch_data[bid*npts*dim+index*dim+i];
			}
			// get the indices
			std::vector<float> dists(K);
			std::vector<size_t> ids(K);
			nanoflann::KNNResultSet<float> resultSet(K);
			resultSet.init(&ids[0], &dists[0] );
			tree.index->findNeighbors(resultSet, &query[0], nanoflann::SearchParams(10));

			for(size_t i=0; i<K; i++)
			{
				used[ids[i]] ++;
			}
			used[index] += 100;

			// fill the queries and neighborhoods
			for(size_t i=0; i<K; i++)
			{
				batch_indices[bid*nqueries*K+ptid*K+i] = ids[i];
			}
			for(size_t i=0; i<dim; i++)
			{
				batch_queries[bid*nqueries*dim+ptid*dim+i] = query[i];
			}
		}
	}
}


void cpp_knn_batch_distance_pick_omp(const float* batch_data, const size_t batch_size, const size_t npts, const size_t dim, float* batch_queries, const size_t nqueries, const size_t K, long* batch_indices)
{
	mt19937 mt_rand(time(0));
	#pragma omp parallel for
	for(size_t bid=0; bid < batch_size; bid++)
	{
		const float* points = &batch_data[bid*npts*dim];
		// long* indices = &batch_indices[bid*nqueries*K];
		// float* queries = &batch_queries[bid*nqueries*dim];

		// create the kdtree
		typedef KDTreeTableAdaptor< float, float> KDTree;
		KDTree tree(npts, dim, points, 10);
		tree.index->buildIndex();

		vector<int> used(npts, 0);
		int current_id = 0;
		for(size_t ptid=0; ptid<nqueries; ptid++)
		{
			// get the possible points
			vector<size_t> possible_ids;
			while(possible_ids.size() == 0)
			{
				for(size_t i=0; i<npts; i++)
				{
					if(used[i] == current_id)
					{
						possible_ids.push_back(i);
					}
				}
				if(possible_ids.size() == 0)
				{
					current_id = *std::min_element(std::begin(used), std::end(used));
				}
			}

			// get an index
			size_t index = possible_ids[mt_rand()%possible_ids.size()];

			// create the query
			vector<float> query(3);
			for(size_t i=0; i<dim; i++)
			{
				query[i] = batch_data[bid*npts*dim+index*dim+i];
			}
			// get the indices
			std::vector<float> dists(K);
			std::vector<size_t> ids(K);
			nanoflann::KNNResultSet<float> resultSet(K);
			resultSet.init(&ids[0], &dists[0] );
			tree.index->findNeighbors(resultSet, &query[0], nanoflann::SearchParams(10));

			for(size_t i=0; i<K; i++)
			{
				used[ids[i]] ++;
			}
			used[index] += 100;

			// fill the queries and neighborhoods
			for(size_t i=0; i<K; i++)
			{
				batch_indices[bid*nqueries*K+ptid*K+i] = ids[i];
			}
			for(size_t i=0; i<dim; i++)
			{
				batch_queries[bid*nqueries*dim+ptid*dim+i] = query[i];
			}
		}
	}
}
