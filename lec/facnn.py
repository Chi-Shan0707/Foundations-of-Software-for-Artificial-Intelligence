import math
import random 

def fac(x):
    return math.factorial(x)

def log_fac(x):
    return math.log(math.factorial(x))

x = log_fac(5)
#print(x)

N = 10

train_data = [(x, log_fac(x)) for x in range(N)]
test_data = [(x, fac(x)) for x in range(N+5)]

#print(train_data)
#print(test_data)

def relu(z):
    return max(0.0, z)

def relu_deriv(z):
    return 1 if z>0 else 0


def sigmoid(z):
    return 1.0/(1.0+math.exp(-z))

def sigmoid_deriv(h):
    return h * (1-h)

class FacNN:
    def __init__(self, hidden_size = 10):
        self.hidden_size = hidden_size
        self.w1 = [random.uniform(-1, 1) for _ in range(hidden_size)]
        self.b1 = [random.uniform(-1, 1) for _ in range(hidden_size)]
        self.w2 = [random.uniform(-1, 1) for _ in range(hidden_size)]
        self.b2 = random.uniform(-1,1)

    def forward(self, x):
        z = [0] * self.hidden_size
        h = [0] * self.hidden_size
        y = 0
        for i in range(self.hidden_size):
            z[i] = self.w1[i] * x + self.b1[i]
            h[i] = sigmoid(z[i])

        for i in range(self.hidden_size):
            y += self.w2[i] * h[i] 

        y += self.b2
        return y, h, z

    def train(self, x, target, lr = 0.1):
        y, h, z = self.forward(x)
        error = y - target
        mse = error ** 2
        dy = error


        dz = []
        for i in range(self.hidden_size):
            grad = dy * self.w2[i] * sigmoid_deriv(h[i])
            dz.append(grad)

        for i in range(self.hidden_size):
            grad = dy * h[i]
            self.w2[i] -= grad * lr
        self.b2 -= dy * lr

        for i in range(self.hidden_size):
            self.w1[i] -= dz[i] * x * lr
            self.b1[i] -= dz[i] * lr

        return mse
        

nn = FacNN(20)
y = nn.forward(5)
#print(y)

for epoch in range(10000):
    loss = 0.0
    for x,y in train_data: 
        loss += nn.train(x, y, 0.001)

    if epoch % 100 == 0:
        print(loss)

for x,y in test_data:
    predict, _, _ = nn.forward(x)
    fac = math.exp(predict)
    print("label:", y, "output:", fac, "error:", fac-y) 