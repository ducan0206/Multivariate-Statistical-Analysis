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
    Tính phương sai (variance) của một tập dữ liệu.

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
        - p, q: Hai vector (danh sách) đại diện cho hai điểm.
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
        - p, q: Hai vector đại diện cho hai điểm.
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
    """
    Tính khoảng cách Mahalanobis giữa hai điểm p và q dựa trên ma trận hiệp phương sai.

    Input:
        - p, q: Hai vector đại diện cho hai điểm.
        - covariance_matrix: Ma trận hiệp phương sai của quần thể.
    Output:
        - Khoảng cách Mahalanobis (số thực).
    """
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
    """
    Tính khoảng cách Mahalanobis cho toàn bộ các mẫu trong tập dữ liệu so với tâm (mu).

    Input:
        - X: Ma trận dữ liệu (mảng 2 chiều hoặc DataFrame).
        - mu: Vector trung bình (tùy chọn, mặc định là trung bình của X).
        - covariance_matrix: Ma trận hiệp phương sai (tùy chọn, mặc định là tính từ X).
    Output:
        - Danh sách (list) chứa khoảng cách Mahalanobis của từng mẫu.
    """
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

def simulate_lln(features, pop_means, n_trials=100, step=20):
    """
    Mô phỏng Luật số lớn (LLN) bằng cách tính sai số trung bình tích lũy qua nhiều lần chạy.

    Input:
        - features: DataFrame chứa các biến được trưng.
        - pop_means: Dictionary chứa trung bình quần thể của từng biến.
        - n_trials: Số lần lặp lại mô phỏng (mặc định 100).
        - step: Bước nhảy của kích thước mẫu n (mặc định 20).
    Output:
        - all_trials_errors: Dictionary chứa ma trận sai số của từng biến.
        - sizes: Mảng các kích thước mẫu n đã tính.
    """
    n_samples = len(features)
    sizes = np.append(np.arange(step, n_samples, step), n_samples)
    all_trials_errors = {col: [] for col in features.columns}

    for _ in range(n_trials):
        # Shuffle chỉ số
        shuffled_idx = np.random.permutation(n_samples)
        shuffled_data = features.iloc[shuffled_idx]
        
        for col in features.columns:
            # Tính tổng tích lũy
            cum_sum = np.cumsum(shuffled_data[col].values)
            # Lấy giá trị tới các mốc n
            running_means = cum_sum[sizes - 1] / sizes
            # Tính sai số tương đối
            rel_errors = np.abs(running_means - pop_means[col]) / pop_means[col]
            all_trials_errors[col].append(rel_errors)
            
    return all_trials_errors, sizes

def simulate_clt(features, sample_sizes=[5, 10, 30, 50, 100], n_trials=1000):
    """
    Mô phỏng định lý giới hạn trung tâm (CLT) bằng cách lấy mẫu nhiều lần.

    Input:
        - features: DataFrame chứa các biến được trưng.
        - sample_sizes: Danh sách các kích thước mẫu n cần khảo sát.
        - n_trials: Số lần lấy mẫu cho mỗi n (mặc định 1000).
    Output:
        - clt_results: Dictionary lồng nhau {biến: {n: [danh sách trung bình mẫu]}}.
    """
    clt_results = {col: {n: [] for n in sample_sizes} for col in features.columns}

    for n in sample_sizes:
        for _ in range(n_trials):
            # Lấy mẫu ngẫu nhiên có hoàn lại
            sample_idx = np.random.choice(len(features), size=n, replace=True)
            sample = features.iloc[sample_idx]
            for col in features.columns:
                clt_results[col][n].append(sample[col].mean())

def mle_normal_univariate(data):
    """
    Ước lượng Triển vọng Cực đại (MLE) cho phân phối chuẩn một biến N(μ, σ²).

    Nghiệm giải tích (bằng cách đặt đạo hàm log-likelihood = 0):
        μ̂_MLE   = (1/n) Σ xᵢ               = x̄
        σ̂²_MLE  = (1/n) Σ (xᵢ − x̄)²       (chia n, không hiệu chỉnh Bessel)

    Lưu ý về tính chệch (bias):
        - μ̂_MLE là ước lượng KHÔNG CHỆCH của μ.
        - σ̂²_MLE là ước lượng CHỆCH của σ²: E[σ̂²_MLE] = (n-1)/n · σ².
        - Ước lượng không chệch là S² = (1/(n-1)) Σ (xᵢ − x̄)²  (Bessel correction).
        - Khi n lớn (ví dụ n=1599), tỷ số n/(n-1) ≈ 1 nên chênh lệch rất nhỏ.

    Parameters
    ----------
    data : array-like, shape (n,)
        Chuỗi quan sát.

    Returns
    -------
    dict với các key:
        mu_hat      : float   – ước lượng μ (= x̄)
        sigma2_hat  : float   – ước lượng σ² (MLE, chia n)
        sigma_hat   : float   – ước lượng σ  (MLE, chia n)
        n           : int     – số quan sát
    """
    x = np.asarray(data, dtype=float)
    n = len(x)
    if n < 2:
        raise ValueError("Cần ít nhất 2 quan sát để ước lượng MLE.")

    mu_hat     = x.mean()                         # x̄  (không chệch)
    sigma2_hat = np.sum((x - mu_hat) ** 2) / n    # chia n  → MLE (chệch)
    sigma_hat  = np.sqrt(sigma2_hat)

    return {
        "mu_hat":     mu_hat,
        "sigma2_hat": sigma2_hat,
        "sigma_hat":  sigma_hat,
        "n":          n,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MLE đa biến
# ─────────────────────────────────────────────────────────────────────────────

def mle_normal_multivariate(X):
    """
    Ước lượng Triển vọng Cực đại (MLE) cho phân phối chuẩn đa biến N_p(μ, Σ).

    Nghiệm giải tích:
        μ̂_MLE     = (1/n) Σⱼ xⱼ                       = x̄      (không chệch)
        Σ̂_MLE     = (1/n) Σⱼ (xⱼ − x̄)(xⱼ − x̄)ᵀ               (chệch, chia n)
        S_unbiased = (1/(n-1)) Σⱼ (xⱼ − x̄)(xⱼ − x̄)ᵀ           (không chệch)

    Mối quan hệ:
        S_unbiased = (n/(n-1)) · Σ̂_MLE

    Parameters
    ----------
    X : array-like, shape (n, p)
        Ma trận dữ liệu: n quan sát, mỗi quan sát p chiều.

    Returns
    -------
    dict với các key:
        mu_hat         : ndarray (p,)    – vector trung bình MLE (= x̄)
        Sigma_hat      : ndarray (p, p)  – ma trận hiệp phương sai MLE (chia n)
        Sigma_unbiased : ndarray (p, p)  – ma trận hiệp phương sai không chệch (chia n-1)
        n              : int             – số quan sát
        p              : int             – số biến
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X phải là mảng 2 chiều (n, p).")
    n, p = X.shape
    if n <= p:
        raise ValueError(
            f"Cần n > p để ma trận Σ không suy biến. Hiện n={n}, p={p}."
        )

    mu_hat = X.mean(axis=0)             # (p,)  — không chệch
    diff   = X - mu_hat                 # (n, p)

    Sigma_hat      = (diff.T @ diff) / n        # chia n   → MLE (chệch)
    Sigma_unbiased = (diff.T @ diff) / (n - 1)  # chia n-1 → không chệch

    return {
        "mu_hat":         mu_hat,
        "Sigma_hat":      Sigma_hat,
        "Sigma_unbiased": Sigma_unbiased,
        "n":              n,
        "p":              p,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Log-likelihood đa biến
# ─────────────────────────────────────────────────────────────────────────────

def log_likelihood_normal(X, mu, Sigma):
    """
    Tính log-likelihood của dữ liệu X dưới phân phối chuẩn đa biến N_p(μ, Σ).

    Công thức:
        ℓ(μ, Σ) = −(np/2)·ln(2π)
                  − (n/2)·ln|Σ|
                  − (1/2) Σⱼ (xⱼ − μ)ᵀ Σ⁻¹ (xⱼ − μ)

    Dùng `np.linalg.slogdet` để tính ln|Σ| ổn định về mặt số học,
    và `np.einsum` để tính tổng dạng bậc hai một cách hiệu quả.

    Parameters
    ----------
    X     : array-like, shape (n, p)
    mu    : array-like, shape (p,)      – vector trung bình
    Sigma : array-like, shape (p, p)    – ma trận hiệp phương sai (phải xác định dương)

    Returns
    -------
    float – giá trị log-likelihood (càng lớn càng tốt).
            Trả về -∞ nếu Σ suy biến (|Σ| ≤ 0).
    """
    X   = np.asarray(X,     dtype=float)
    mu  = np.asarray(mu,    dtype=float).ravel()
    Sig = np.asarray(Sigma, dtype=float)

    n, p = X.shape
    sign, logdet = np.linalg.slogdet(Sig)
    if sign <= 0:
        return -np.inf

    Sig_inv = np.linalg.inv(Sig)
    diff    = X - mu                                           # (n, p)
    quad    = np.einsum('ni,ij,nj->', diff, Sig_inv, diff)    # Σⱼ (xⱼ-μ)ᵀΣ⁻¹(xⱼ-μ)

    ll = -0.5 * (n * p * np.log(2.0 * np.pi) + n * logdet + quad)
    return float(ll)


# ─────────────────────────────────────────────────────────────────────────────
# Log-likelihood đơn biến
# ─────────────────────────────────────────────────────────────────────────────

def log_likelihood_univariate(data, mu, sigma2):
    """
    Tính log-likelihood của dữ liệu dưới phân phối chuẩn một biến N(μ, σ²).

    Công thức (thu gọn từ log-likelihood đa biến khi p=1):
        ℓ(μ, σ²) = −(n/2)·ln(2π·σ²) − (1/(2σ²)) Σᵢ (xᵢ − μ)²

    Parameters
    ----------
    data   : array-like, shape (n,)
    mu     : float   – trung bình
    sigma2 : float   – phương sai (phải > 0)

    Returns
    -------
    float – giá trị log-likelihood (càng lớn càng tốt).
            Trả về -∞ nếu sigma2 ≤ 0.
    """
    x  = np.asarray(data, dtype=float)
    n  = len(x)
    if sigma2 <= 0:
        return -np.inf

    ll = (-n / 2.0) * np.log(2.0 * np.pi * sigma2) \
         - np.sum((x - mu) ** 2) / (2.0 * sigma2)
    return float(ll)
