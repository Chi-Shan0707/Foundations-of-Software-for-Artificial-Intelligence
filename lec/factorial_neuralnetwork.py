import math
import random
def log_fac(x):
    return math.log(math.factorial(x))

N = 10

train_data = [(i,log_fac(i)) for i in range(1, N + 1)]
test_data = [(i,log_fac(i)) for i in range(N + 1, N + 6)]

print("Train data:", train_data)
print("Test data:", test_data)

class FacNN:
    def __init__(self , hidden_size = 10):
        self.hidden_size = hidden_size
        self.W1 = [random.uniform(-1,1) for _ in range(hidden_size)]
        self.b1 = [random.uniform(-1,1) for _ in range(hidden_size)]
        self.W2 = [random.uniform(-1,1) for _ in range(hidden_size)]
        self.b2 = random.uniform(-1,1)
    def forward(self, x):
        h = []

        z = [0.0] * self.hidden_size

        y = 0
        for i in range(self.hidden_size):
            z[i] = self.W1[i] * x + self.b1[i]
            h.append(max(0, z[i])) # relu
        for i in range(self.hidden_size):
            y += self.W2[i] * h[i]
        y += self.b2
        return y
    
    def train(self, x, target, lr= 0.1):
        # Forward pass
        h = []
        z = [0.0] * self.hidden_size

        for i in range(self.hidden_size):
            z[i] = self.W1[i] * x + self.b1[i]
            h.append(max(0, z[i]))  # ReLU

        y = 0
        for i in range(self.hidden_size):
            y += self.W2[i] * h[i]
        y += self.b2

        error = y - target

        # Backpropagation
        # 输出层误差 (MSE: L = 0.5*(y-target)^2, dL/dy = y-target = error)
        dL_dy = error

        # 输出层梯度
        dL_dW2 = [dL_dy * h[i] for i in range(self.hidden_size)]
        dL_db2 = dL_dy

        # 隐藏层误差: δ_i = (y-target) * W2_i * ReLU'(z_i)
        # ReLU'(z) = 1 if z > 0 else 0
        dL_dz = [0.0] * self.hidden_size
        for i in range(self.hidden_size):
            relu_derivative = 1.0 if z[i] > 0 else 0.0
            dL_dz[i] = dL_dy * self.W2[i] * relu_derivative

        # 隐藏层梯度
        dL_dW1 = [dL_dz[i] * x for i in range(self.hidden_size)]
        dL_db1 = dL_dz[:]  # 复制列表

        # 更新权重
        for i in range(self.hidden_size):
            self.W1[i] -= lr * dL_dW1[i]
            self.b1[i] -= lr * dL_db1[i]
            self.W2[i] -= lr * dL_dW2[i]
        self.b2 -= lr * dL_db2

        return error ** 2  # 返回MSE


nn = FacNN(20)

# 训练网络
epochs = 5000
lr = 0.01

print("\n=== Training ===")
for epoch in range(epochs):
    total_loss = 0
    for x, target in train_data:
        loss = nn.train(x, target, lr)
        total_loss += loss
    if (epoch + 1) % 500 == 0:
        avg_loss = total_loss / len(train_data)
        print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.6f}")

# 测试结果
print("\n=== Results ===")
print("\nTrain data:")
for x, target in train_data:
    pred = nn.forward(x)
    print(f"x={x}: pred={pred:.4f}, actual={target:.4f}, error={abs(pred-target):.4f}")

print("\nTest data (unseen):")
for x, target in test_data:
    pred = nn.forward(x)
    print(f"x={x}: pred={pred:.4f}, actual={target:.4f}, error={abs(pred-target):.4f}")


