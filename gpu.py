import lightgbm as lgb
print(lgb.__version__)

# Create a simple dataset and try GPU
import numpy as np
import lightgbm as lgb

X = np.random.rand(1000, 10)
y = np.random.rand(1000)

dtrain = lgb.Dataset(X, label=y)
params = {'device': 'gpu', 'objective': 'regression'}
bst = lgb.train(params, dtrain, num_boost_round=10)
print("GPU training works!")
