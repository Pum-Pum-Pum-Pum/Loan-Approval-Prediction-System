import numpy as np


class LogisticRegressionGD:
    def __init__(self, learning_rate: float = 0.01, n_iters: int = 1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.loss_history = []

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        for _ in range(self.n_iters):
            linear_pred = np.dot(X, self.weights) + self.bias
            y_prob = self._sigmoid(linear_pred)

            dw = (1 / n_samples) * np.dot(X.T, (y_prob - y))
            db = (1 / n_samples) * np.sum(y_prob - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            loss = -np.mean(y * np.log(y_prob + 1e-9) + (1 - y) * np.log(1 - y_prob + 1e-9))
            self.loss_history.append(loss)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        linear_pred = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_pred)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


def train_test_split_numpy(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42):
    np.random.seed(random_state)
    indices = np.random.permutation(len(X))
    test_count = int(len(X) * test_size)
    test_idx = indices[:test_count]
    train_idx = indices[test_count:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def accuracy_numpy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(y_true == y_pred)


def main() -> None:
    np.random.seed(42)

    n_samples = 500
    X = np.random.randn(n_samples, 2)
    linear_signal = 2.0 * X[:, 0] - 1.5 * X[:, 1] + 0.3
    probs = 1 / (1 + np.exp(-linear_signal))
    y = (probs >= 0.5).astype(int)

    X_train, X_test, y_train, y_test = train_test_split_numpy(X, y, test_size=0.2)

    model = LogisticRegressionGD(learning_rate=0.1, n_iters=2000)
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)
    y_pred = model.predict(X_test, threshold=0.5)
    acc = accuracy_numpy(y_test, y_pred)

    print('LOGISTIC REGRESSION FROM SCRATCH')
    print('--------------------------------')
    print(f'Learned weights: {model.weights}')
    print(f'Learned bias   : {model.bias:.4f}')
    print(f'Test Accuracy  : {acc:.4f}')
    print(f'Final Train Loss: {model.loss_history[-1]:.4f}')
    print('\nFirst 5 predictions:')
    for i in range(5):
        print(f'Actual: {y_test[i]} | Prob: {y_prob[i]:.4f} | Predicted: {y_pred[i]}')


if __name__ == '__main__':
    main()
