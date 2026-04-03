import math

def mean(data):
    if len(data) == 0:
        raise ValueError("Data cannot be empty")
    
    return sum(data) / len(data)

def variance(data, is_population=False):
    if len(data) == 0:
        raise ValueError("Data cannot be empty")
    
    if len(data) == 1:
        raise ValueError("At least two data points are required to calculate variance")
    
    mu = mean(data)
    sum_squared_diff = sum((x - mu) ** 2 for x in data)
    if is_population:
        return sum_squared_diff / len(data)
    else:
        return sum_squared_diff / (len(data) - 1)
    
def standard_deviation(data, is_population=False):
    return math.sqrt(variance(data, is_population))

def covariance(data_x, data_y, is_population=False):
    if len(data_x) == 0 or len(data_y) == 0:
        raise ValueError("Data cannot be empty")
    
    if len(data_x) != len(data_y) or len(data_x) == 1:
        raise ValueError("Data must have the same number of points and at least two points to calculate covariance")
    
    n = len(data_x)
    mu_x = mean(data_x)
    mu_y = mean(data_y)
    
    cov_sum = sum((x - mu_x) * (y - mu_y) for x, y in zip(data_x, data_y))
    
    if is_population:
        return cov_sum / n
    else:
        return cov_sum / (n - 1)
    
def correlation(data_x, data_y):
    if len(data_x) == 0 or len(data_y) == 0:
        raise ValueError("Data cannot be empty")
    
    if len(data_x) != len(data_y) or len(data_x) == 1:
        raise ValueError("Data must have the same number of points and at least two points to calculate correlation")
    
    std_x = standard_deviation(data_x)
    std_y = standard_deviation(data_y)
    
    if std_x == 0 or std_y == 0:
        raise ValueError("The standard deviation of one of the variables is 0, cannot calculate correlation")
    
    cov = covariance(data_x, data_y)
    
    return cov / (std_x * std_y)

def euclidean_distance(p, q):
    if len(p) != len(q):
        raise ValueError("Points must be of the same length")
    
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(p, q)))

def statistical_distance(p, q, variance):
    if len(p) != len(q):
        raise ValueError("Points must be of the same length")
    
    if len(p) != len(variance):
        raise ValueError("Variance must be of the same length as points")
    
    return math.sqrt(sum(((x - y) ** 2) / var for x, y, var in zip(p, q, variance)))
