import numpy as np

arr = [[1,2], [3,4], [5,6], [1,3]]

A = np.array(arr)
print(A)
ind = np.lexsort((A[:,1], A[:,0]))
A = A[ind]
print(A)
print([3,6] in A)
print(np.where(np.all(A == [3,4], axis=1))[0])
