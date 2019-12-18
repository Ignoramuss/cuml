import cudf
from cuml.neighbors import NearestNeighbors
import numpy as np

np_float = np.array([
  [1,2,3], # Point 1
  [1,2,4], # Point 2
  [2,2,4]  # Point 3
]).astype('float32')

gdf_float = cudf.DataFrame()
gdf_float['dim_0'] = np.ascontiguousarray(np_float[:,0])
gdf_float['dim_1'] = np.ascontiguousarray(np_float[:,1])
gdf_float['dim_2'] = np.ascontiguousarray(np_float[:,2])

print('n_samples = 3, n_dims = 3')
print(gdf_float)

nn_float = NearestNeighbors(algorithm="sweet")
nn_float.fit(gdf_float)
# get 3 nearest neighbors
distances,indices = nn_float.kneighbors(gdf_float,n_neighbors=3)

print(indices)
print(distances)
