import math
import numpy as np

def mean(data):
    """
    Tính giá tr? trung bình (mean) c?a m?t t?p d? li?u.

    Input:
        - data: Danh sách ho?c m?ng ch?a các giá tr? s?.
    Output:
        - Giá tr? trung bình c?a t?p d? li?u.
    """
    if len(data) == 0:
        raise ValueError("D? li?u không ???c ?? tr?ng")
    
    # T?ng các giá tr? chia cho s? l??ng ph?n t?
    return sum(data) / len(data)

def variance(data, is_population=False):
    """
    Tính ph??ng sai (variance) c?a t?p d? li?u.

    Input:
        - data: Danh sách các giá tr? s?.
        - is_population: Boolean, True n?u là ph??ng sai t?ng th?, False n?u là ph??ng sai m?u (m?c ??nh là False).
    Output:
        - Giá tr? ph??ng sai.
    """
    if len(data) == 0:
        raise ValueError("D? li?u không ???c ?? tr?ng")
    
    if len(data) == 1:
        raise ValueError("C?n ít nh?t hai ?i?m d? li?u ?? tính ph??ng sai")
    
    mu = mean(data)
    # Tính t?ng bình ph??ng ?? l?ch so v?i giá tr? trung bình
    sum_squared_diff = sum((x - mu) ** 2 for x in data)
    
    if is_population:
        # Ph??ng sai t?ng th? chia cho N
        return sum_squared_diff / len(data)
    else:
        # Ph??ng sai m?u chia cho N-1 (hi?u ch?nh Bessel)
        return sum_squared_diff / (len(data) - 1)
    
def standard_deviation(data, is_population=False):
    """
    Tính ?? l?ch chu?n (standard deviation).

    Input:
        - data: Danh sách các giá tr? s?.
        - is_population: Tính cho t?ng th? hay m?u.
    Output:
        - Giá tr? ?? l?ch chu?n.
    """
    # ?? l?ch chu?n là c?n b?c hai c?a ph??ng sai
    return math.sqrt(variance(data, is_population))

def covariance(data_x, data_y, is_population=False):
    """
    Tính hi?p ph??ng sai (covariance) gi?a hai bi?n X và Y.

    Input:
        - data_x: Danh sách giá tr? c?a bi?n X.
        - data_y: Danh sách giá tr? c?a bi?n Y.
        - is_population: Tính cho t?ng th? hay m?u.
    Output:
        - Giá tr? hi?p ph??ng sai.
    """
    if len(data_x) == 0 or len(data_y) == 0:
        raise ValueError("D? li?u không ???c ?? tr?ng")
    
    if len(data_x) != len(data_y) or len(data_x) == 1:
        raise ValueError("Hai t?p d? li?u ph?i có cùng kích th??c và l?n h?n 1 ph?n t?")
    
    n = len(data_x)
    mu_x = mean(data_x)
    mu_y = mean(data_y)
    
    # Tính t?ng tích các ?? l?ch t??ng ?ng c?a X và Y
    cov_sum = sum((x - mu_x) * (y - mu_y) for x, y in zip(data_x, data_y))
    
    if is_population:
        return cov_sum / n
    else:
        return cov_sum / (n - 1)
    
def correlation(data_x, data_y):
    """
    Tính h? s? t??ng quan Pearson (correlation coefficient).

    Input:
        - data_x, data_y: Hai t?p d? li?u c?n tính t??ng quan.
    Output:
        - Giá tr? h? s? t??ng quan trong kho?ng [-1, 1].
    """
    if len(data_x) == 0 or len(data_y) == 0:
        raise ValueError("D? li?u không ???c ?? tr?ng")
    
    if len(data_x) != len(data_y) or len(data_x) == 1:
        raise ValueError("D? li?u không h?p l? ?? tính t??ng quan")
    
    std_x = standard_deviation(data_x)
    std_y = standard_deviation(data_y)
    
    # Tránh l?i chia cho 0 n?u m?t bi?n không có s? bi?n thiên
    if std_x == 0 or std_y == 0:
        raise ValueError("?? l?ch chu?n b?ng 0, không th? tính t??ng quan")
    
    cov = covariance(data_x, data_y)
    
    # H? s? t??ng quan = Hi?p ph??ng sai / Tích hai ?? l?ch chu?n
    return cov / (std_x * std_y)

def euclidean_distance(p, q):
    """
    Tính kho?ng cách Euclid gi?a hai ?i?m trong không gian n-chi?u.

    Input:
        - p, q: Hai vector (danh sách) ??i di?n cho hai ?i?m.
    Output:
        - Kho?ng cách Euclid (s? th?c).
    """
    if len(p) != len(q):
        raise ValueError("Hai ?i?m ph?i có cùng s? chi?u")
    
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(p, q)))

def statistical_distance(p, q, variance):
    """
    Tính kho?ng cách th?ng kê (statistical distance), có tính ??n ph??ng sai c?a t?ng chi?u.

    Input:
        - p, q: Hai vector ??i di?n cho hai ?i?m.
        - variance: Danh sách ph??ng sai t??ng ?ng c?a t?ng chi?u.
    Output:
        - Giá tr? kho?ng cách th?ng kê.
    """
    if len(p) != len(q):
        raise ValueError("Hai ?i?m ph?i có cùng s? chi?u")
    
    if len(p) != len(variance):
        raise ValueError("S? l??ng giá tr? ph??ng sai ph?i kh?p v?i s? chi?u c?a ?i?m")
    
    # M?i chi?u ???c chu?n hóa b?ng cách chia cho ph??ng sai c?a chi?u ?ó
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

