import numpy as np

arr_1d = np.array([1, 2, 3])
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
arr_3d = np.array(
    [
        [[1, 2, 3], [4, 5, 6]],
        [[7, 8, 9], [10, 11, 12]],
    ]
)
print(arr_1d.ndim)
print(arr_2d.ndim)
print(arr_3d.ndim)

print(arr_1d.dtype)
print(arr_2d.dtype)
print(arr_3d.dtype)
print(arr_3d.dtype)
input("Press Enter to continue: ")

model = input(arr_2d.dtype)

