import math
import numpy as np

def mean(data):
    """
    Tính giá trị trung bình (mean) của một tập dữ liệu.

    Input:
        - data: Danh sách hoặc mảng chứa các giá trị số.
    Output:
        - Giá trị trung bình của tập dữ liệu.
    """
    if len(data) == 0:
        raise ValueError("Dữ liệu không được để trống")
    
    # Tổng các giá trị chia cho số lượng phần tử
    return sum(data) / len(data)

def variance(data, is_population=False):
    """
    Tính phương sai (variance) của tập dữ liệu.

    Input:
        - data: Danh sách các giá trị số.
        - is_population: Boolean, True nếu là phương sai tổng thể, False nếu là phương sai mẫu (mặc định là False).
    Output:
        - Giá trị phương sai.
    """
    if len(data) == 0:
        raise ValueError("Dữ liệu không được để trống")
    
    if len(data) == 1:
        raise ValueError("Cần ít nhất hai điểm dữ liệu để tính phương sai")
    
    mu = mean(data)
    # Tính tổng bình phương độ lệch so với giá trị trung bình
    sum_squared_diff = sum((x - mu) ** 2 for x in data)
    
    if is_population:
        # Phương sai tổng thể chia cho N
        return sum_squared_diff / len(data)
    else:
        # Phương sai mẫu chia cho N-1 (hiệu chỉnh Bessel)
        return sum_squared_diff / (len(data) - 1)
    
def standard_deviation(data, is_population=False):
    """
    Tính độ lệch chuẩn (standard deviation).

    Input:
        - data: Danh sách các giá trị số.
        - is_population: Tính cho tổng thể hay mẫu.
    Output:
        - Giá trị độ lệch chuẩn.
    """
    # Độ lệch chuẩn là căn bậc hai của phương sai
    return math.sqrt(variance(data, is_population))

def covariance(data_x, data_y, is_population=False):
    """
    Tính hiệp phương sai (covariance) giữa hai biến X và Y.

    Input:
        - data_x: Danh sách giá trị của biến X.
        - data_y: Danh sách giá trị của biến Y.
        - is_population: Tính cho tổng thể hay mẫu.
    Output:
        - Giá trị hiệp phương sai.
    """
    if len(data_x) == 0 or len(data_y) == 0:
        raise ValueError("Dữ liệu không được để trống")
    
    if len(data_x) != len(data_y) or len(data_x) == 1:
        raise ValueError("Hai tập dữ liệu phải có cùng kích thước và lớn hơn 1 phần tử")
    
    n = len(data_x)
    mu_x = mean(data_x)
    mu_y = mean(data_y)
    
    # Tính tổng tích các độ lệch tương ứng của X và Y
    cov_sum = sum((x - mu_x) * (y - mu_y) for x, y in zip(data_x, data_y))
    
    if is_population:
        return cov_sum / n
    else:
        return cov_sum / (n - 1)
    
def correlation(data_x, data_y):
    """
    Tính hệ số tương quan Pearson (correlation coefficient).

    Input:
        - data_x, data_y: Hai tập dữ liệu cần tính tương quan.
    Output:
        - Giá trị hệ số tương quan trong khoảng [-1, 1].
    """
    if len(data_x) == 0 or len(data_y) == 0:
        raise ValueError("Dữ liệu không được để trống")
    
    if len(data_x) != len(data_y) or len(data_x) == 1:
        raise ValueError("Dữ liệu không hợp lệ để tính tương quan")
    
    std_x = standard_deviation(data_x)
    std_y = standard_deviation(data_y)
    
    # Tránh lỗi chia cho 0 nếu một biến không có sự biến thiên
    if std_x == 0 or std_y == 0:
        raise ValueError("Độ lệch chuẩn bằng 0, không thể tính tương quan")
    
    cov = covariance(data_x, data_y)
    
    # Hệ số tương quan = Hiệp phương sai / Tích hai độ lệch chuẩn
    return cov / (std_x * std_y)

def euclidean_distance(p, q):
    """
    Tính khoảng cách Euclid giữa hai điểm trong không gian n-chiều.

    Input:
        - p, q: Hai vectơ (danh sách) đại diện cho hai điểm.
    Output:
        - Khoảng cách Euclid (số thực).
    """
    if len(p) != len(q):
        raise ValueError("Hai điểm phải có cùng số chiều")
    
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(p, q)))

def statistical_distance(p, q, variance):
    """
    Tính khoảng cách thống kê (statistical distance), có tính đến phương sai của từng chiều.

    Input:
        - p, q: Hai vectơ đại diện cho hai điểm.
        - variance: Danh sách phương sai tương ứng của từng chiều.
    Output:
        - Giá trị khoảng cách thống kê.
    """
    if len(p) != len(q):
        raise ValueError("Hai điểm phải có cùng số chiều")
    
    if len(p) != len(variance):
        raise ValueError("Số lượng giá trị phương sai phải khớp với số chiều của điểm")
    
    # Mỗi chiều được chuẩn hóa bằng cách chia cho phương sai của chiều đó
    return math.sqrt(sum(((x - y) ** 2) / var for x, y, var in zip(p, q, variance)))

def mahalanobis_distance(p, q, covariance_matrix):
    if len(p) != len(q):
        raise ValueError("Points must be of the same length")
    
    if len(p) != len(covariance_matrix):
        raise ValueError("Covariance matrix must be of the same length as points")
    
    p = np.array(p, dtype=float)
    q = np.array(q, dtype=float)
    covariance_matrix = np.array(covariance_matrix, dtype=float)

    diff = p - q

    inv_cov_matrix = np.linalg.pinv(covariance_matrix)

    distance = diff @ inv_cov_matrix @ diff

    return math.sqrt(distance)

def mahalanobis_all_samples(X, mu=None, covariance_matrix=None):
    X = np.array(X, dtype=float)
 
    if mu is None:
        mu = X.mean(axis=0)
    mu = np.array(mu, dtype=float)
 
    if covariance_matrix is None:
        S = np.cov(X, rowvar=False)
    else:
        S = np.array(covariance_matrix, dtype=float)
 
    S_inv = np.linalg.pinv(S)          

    distances = []
    for x in X:
        diff = x - mu
        D2 = float(diff @ S_inv @ diff)
        distances.append(math.sqrt(D2))
 
    return distances

