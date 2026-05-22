import numpy as np
import matplotlib.pyplot as plt

# 目標関数の定義　（別の関数で試す際はこれを変える）
def target_function(x, y):
    """f(x, y) = sin(x) + cos(y)"""
    return np.sin(x) + np.cos(y)

# データセット生成関数
def make_dataset(n_samples, x_range=(-np.pi, np.pi), seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(x_range[0], x_range[1], size=(n_samples, 2))
    y = target_function(X[:, 0], X[:, 1]).reshape(-1, 1)
    return X, y

# ニューラルネットワーク
class TwoLayerNN:
    def __init__(self, n_in=2, n_hidden=32, n_out=1, seed=0):
        rng = np.random.default_rng(seed)

        self.W1 = rng.normal(0, np.sqrt(1.0 / n_in), size=(n_in, n_hidden))
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.normal(0, np.sqrt(1.0 / n_hidden), size=(n_hidden, n_out))
        self.b2 = np.zeros(n_out)

    # 順伝播
    def forward(self, X):
        self.X = X
        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.tanh(self.z1)
        self.y_hat = self.a1 @ self.W2 + self.b2
        return self.y_hat
    
    # Backpropagation
    def backward(self, y_true):
        N = y_true.shape[0]

        dy_hat = (2.0 / N) * (self.y_hat - y_true)     # (N, 1)

        dW2 = self.a1.T @ dy_hat                       # (H, 1)
        db2 = dy_hat.sum(axis=0)                       # (1,)
        da1 = dy_hat @ self.W2.T                       # (N, H)

        dz1 = da1 * (1.0 - self.a1 ** 2)               # (N, H)

        dW1 = self.X.T @ dz1                           # (2, H)
        db1 = dz1.sum(axis=0)                          # (H,)

        return dW1, db1, dW2, db2
    
    # パラメータ更新
    def update(self, grads, lr):
        dW1, db1, dW2, db2 = grads
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

# 損失関数　平均二乗誤差
def mse(y_hat, y):
    return float(np.mean((y_hat - y) ** 2))

# 学習ループ
def train(net, X, y, epochs=2000, batch_size=64, lr=0.05, seed=0):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    losses = []

    for epoch in range(epochs):
        idx = rng.permutation(n)
        for start in range(0, n, batch_size):
            b = idx[start:start + batch_size]
            net.forward(X[b])
            grads = net.backward(y[b])
            net.update(grads, lr)

        losses.append(mse(net.forward(X), y))
    
    return losses

# 実行部分
if __name__ == "__main__":
    print("Training TwoLayerNN on target function f(x, y) = sin(x) + cos(y)")
    # データセットの生成
    X, y = make_dataset(1000, seed=42)
    # ネットの初期化
    net = TwoLayerNN(n_in=2, n_hidden=32, n_out=1, seed=42)
    # 学習の実行
    losses = train(net, X, y, epochs=2000, batch_size=64, lr=0.05, seed=0)
    print(f"学習前 loss: {losses[0]:.5f}")
    print(f"学習後 loss: {losses[-1]:.5f}")

    plt.figure(figsize=(8, 5))
    plt.plot(losses, color="#1D9E75", linewidth=1.5)
    plt.yscale("log")             
    plt.xlabel("Epoch")
    plt.ylabel("MSE (log scale)")
    plt.title("Training Loss Curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=120)

    X_eval, y_eval = make_dataset(5, seed=999)
    y_pred = net.forward(X_eval)
    print("\n--- 評価点5点での比較 ---")
    print(f"{'x1':>9} {'x2':>9} {'true':>10} {'pred':>10} {'|error|':>10}")
    print("-" * 52)
    for i in range(5):
        x1, x2 = X_eval[i]
        yt, yp = y_eval[i, 0], y_pred[i, 0]
        print(f"{x1:>9.4f} {x2:>9.4f} {yt:>10.5f} {yp:>10.5f} {abs(yt-yp):>10.5f}")

    