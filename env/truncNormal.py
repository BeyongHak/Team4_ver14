import numpy as np
from scipy import stats

def make_discrete_truncnorm(mean, std, lower=0, upper=np.inf):
    """
    연속형 Truncated Normal을 구간 적분(Binning)하여 
    Scipy의 이산 확률 변수(rv_discrete) 객체로 반환합니다.
    """
    
    # 1. 연속형 Truncated Normal 객체 생성 (Parent Distribution)
    # Scipy truncnorm은 (x - mean) / std 형태의 표준화된 범위를 받습니다.
    a = (lower - mean) / std
    b = (upper - mean) / std
    tn_continuous = stats.truncnorm(a, b, loc=mean, scale=std)

    # 2. 확률 질량 함수(PMF) 생성 (구간 적분법)
    # P(X=k) = CDF(k + 0.5) - CDF(k - 0.5)
    # 확률이 거의 0이 될 때까지 충분히 넓은 범위를 잡습니다. (예: 평균 + 8표준편차)
    max_k = int(mean + 8 * std)  
    x_values = np.arange(lower, max_k + 1) # 0, 1, 2, ..., max_k
    
    # CDF를 이용해 구간 확률 계산
    upper_bounds = x_values + 0.5
    lower_bounds = x_values - 0.5
    
    # 벡터 연산으로 한방에 계산
    probs = tn_continuous.cdf(upper_bounds) - tn_continuous.cdf(lower_bounds)
    
    # 3. 확률 합 1.0 맞추기 (Normalization)
    probs = probs / probs.sum()
    
    # 4. Scipy rv_discrete 객체 생성
    discrete_dist = stats.rv_discrete(name='discrete_tn', values=(x_values, probs))
    
    return discrete_dist

def get_continuous_truncnorm(mean, std, lower=0, upper=np.inf):
    """
    연속형 Truncated Normal 객체(scipy.stats.truncnorm)를 반환합니다.
    """
    a = (lower - mean) / std
    b = (upper - mean) / std
    return stats.truncnorm(a, b, loc=mean, scale=std)

class HybridTruncNorm:
    """
    Optimization용 Discrete Truncated Normal과
    Simulation용 Continuous Truncated Normal을 모두 포함하는 클래스.
    """
    def __init__(self, mean, std=None, lower=0, upper=None):
        # [User Request] Enforce supplychain.py logic
        # 1. std = sqrt(mean)
        # 2. upper = inf
        
        self.mean_val = mean
        self.std_val = mean**0.5
        self.lower = lower
        self.upper = np.inf
        
        # Optimization용 (Discrete)
        self.discrete_dist = make_discrete_truncnorm(self.mean_val, self.std_val, self.lower, self.upper)
        
        # Simulation용 (Continuous)
        self.continuous_dist = get_continuous_truncnorm(self.mean_val, self.std_val, self.lower, self.upper)
        
    def pmf(self, k):
        return self.discrete_dist.pmf(k)
        
    def cdf(self, k):
        return self.discrete_dist.cdf(k)
        
    def ppf(self, q):
        return self.discrete_dist.ppf(q)
        
    def mean(self):
        # Optimization 로직(delta 계산 등)에서 정수 평균을 기대할 수 있으므로 Discrete Mean 반환
        return self.discrete_dist.mean()
        
    def rvs(self, size=None, random_state=None):
        # Simulation은 Continuous 값 사용
        return self.continuous_dist.rvs(size=size, random_state=random_state)
