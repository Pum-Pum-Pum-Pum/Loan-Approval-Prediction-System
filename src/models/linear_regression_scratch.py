import numpy as np


class LinearRegressionGD:
    def __init__(self, learning_rate: float = 0.01, n_iters: int = 1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias
            error = y_pred - y

            dw = (1 / n_samples) * np.dot(X.T, error)
            db = (1 / n_samples) * np.sum(error)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            mse = np.mean(error ** 2)
            self.loss_history.append(mse)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.weights) + self.bias


def train_test_split_numpy(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42):
    np.random.seed(random_state)
    indices = np.random.permutation(len(X))
    test_count = int(len(X) * test_size)
    test_idx = indices[:test_count]
    train_idx = indices[test_count:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def r2_score_numpy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


def main() -> None:
    np.random.seed(42)

    X = 2 * np.random.rand(500, 1)
    noise = np.random.randn(500) * 0.5
    y = 4 + 3 * X[:, 0] + noise

    X_train, X_test, y_train, y_test = train_test_split_numpy(X, y, test_size=0.2)

    model = LinearRegressionGD(learning_rate=0.05, n_iters=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = np.mean((y_test - y_pred) ** 2)
    r2 = r2_score_numpy(y_test, y_pred)

    print('LINEAR REGRESSION FROM SCRATCH')
    print('------------------------------')
    print(f'Learned weight: {model.weights[0]:.4f}')
    print(f'Learned bias  : {model.bias:.4f}')
    print(f'Test MSE      : {mse:.4f}')
    print(f'Test R2       : {r2:.4f}')
    print(f'Final Train Loss: {model.loss_history[-1]:.4f}')
    print('\nFirst 5 predictions:')
    for i in range(5):
        print(f'Actual: {y_test[i]:.4f} | Predicted: {y_pred[i]:.4f}')


if __name__ == '__main__':
    main()
