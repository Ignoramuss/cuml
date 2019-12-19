import cudf
import timeit
from cuml.neighbors import NearestNeighbors
import numpy as np

def create_test_set(size=1000):
  with open("Skin_NonSkin.txt") as ipf:
    ip_list = []
    for ip in ipf:
      if size == 0:
        break
      ip_list.append([int(i.strip()) for i in ip.split()])
      size -= 1
    np_float = np.array(ip_list).astype('float32')
    return np_float

test_size = 1000
np_float = create_test_set(size=test_size)
gdf_float = cudf.DataFrame()
gdf_float['dim_0'] = np.ascontiguousarray(np_float[:,0])
gdf_float['dim_1'] = np.ascontiguousarray(np_float[:,1])
gdf_float['dim_2'] = np.ascontiguousarray(np_float[:,2])
gdf_float['dim_3'] = np.ascontiguousarray(np_float[:,3])

print('n_samples = {}, n_dims = 4'.format(test_size))
print(gdf_float)

start = timeit.timeit()
nn_float = NearestNeighbors(algorithm="brute")
nn_float.fit(gdf_float)
# get 3 nearest neighbors
distances_b, indices_b = nn_float.kneighbors(gdf_float, n_neighbors=3)
end = timeit.timeit()

startS = timeit.timeit()
nn_float = NearestNeighbors(algorithm="sweet")
nn_float.fit(gdf_float)
# get 3 nearest neighbors
distances_s, indices_s = nn_float.kneighbors(gdf_float, n_neighbors=3)
endS = timeit.timeit()

print(indices_b)
print(distances_b)

print(indices_s)
print(distances_s)

indices_b.to_csv("bf_indices.csv")
distances_b.to_csv("bf_distances.csv")

indices_s.to_csv("sweet_indices.csv")
distances_s.to_csv("sweet_distances.csv")

print("brute", (end - start))
print("sweet", (endS - startS))
