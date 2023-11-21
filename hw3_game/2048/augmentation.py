import numpy as np

# (n, 16, 4, 4)
n = 1
A = [[1, 0, 0, 0],
     [0, 2, 0, 0],
     [0, 0, 3, 0],
     [0, 0, 0, 4]]
A = [[A for _ in range(16)] for _ in range(n)]
A = np.array(A)

print(np.rot90(A, k=4, axes=(2, 3)))
