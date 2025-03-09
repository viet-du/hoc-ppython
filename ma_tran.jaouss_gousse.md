# hoc-ppython
import numpy as np

def gauss_jordan_inverse(matrix):
    """
    Tìm ma trận nghịch đảo bằng phương pháp Gauss-Jordan.
    :param matrix: Ma trận vuông (numpy array)
    :return: Ma trận nghịch đảo nếu tồn tại, ngược lại báo lỗi 
    """
    matrix = np.array(matrix, dtype=float)
    n = matrix.shape[0]

    # Tạo ma trận mở rộng [A | I]
    augmented_matrix = np.hstack((matrix, np.eye(n)))

    # Áp dụng phép biến đổi hàng
    for i in range(n):
        # Tìm phần tử chốt (pivot)
        pivot = augmented_matrix[i, i]
        if abs(pivot) < 1e-10:  # Kiểm tra nếu pivot quá nhỏ
            raise ValueError("Ma trận không khả nghịch")

        # Chia hàng i cho pivot để pivot trở thành 1
        augmented_matrix[i] = augmented_matrix[i] / pivot

        # Dùng hàng i để khử các phần tử khác trong cột
        for j in range(n):
            if i != j:
                factor = augmented_matrix[j, i]
                augmented_matrix[j] -= factor * augmented_matrix[i]

    # Phần bên phải là ma trận nghịch đảo
    inverse_matrix = augmented_matrix[:, n:]
    return inverse_matrix

def input_matrix():
    n = int(input("Nhập kích thước ma trận vuông n x n: "))
    matrix = []
    print("Nhập từng dòng của ma trận, cách nhau bởi dấu cách:")
    for i in range(n):
        row = list(map(float, input(f"Dòng {i + 1}: ").split()))
        if len(row) != n:
            raise ValueError("Số phần tử trong hàng không đúng!")
        matrix.append(row)
    return np.array(matrix)

# Chạy chương trình
try:
    A = input_matrix()
    A_inv = gauss_jordan_inverse(A)
    print("Ma trận nghịch đảo của A là:\n", A_inv)
except ValueError as e:
    print(e)
B = np.array([
    [1, 1, 1, 1],
    [2, 2, 3, 5],
    [3, 3, 4, 5],
    [1, 1, 1, 2]
])

det_B = np.linalg.det(B)
print("\nĐịnh thức của B:", det_B)

if not np.isclose(det_B, 0):
    B_inv = np.linalg.inv(B)
    print("Ma trận nghịch đảo của B:\n", B_inv)
else:
    print("B không khả nghịch")
try:
    A_T = A.T
    right_side = B + A @ A_T
    X = right_side @ np.linalg.inv(A_T)
    print("Ma trận X:\n", X)
except np.linalg.LinAlgError:
    print("Không tồn tại X do A không khả nghịch")
