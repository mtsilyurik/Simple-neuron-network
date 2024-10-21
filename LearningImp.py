import numpy as np
import matplotlib.pyplot as plt

# Функция активации
def sigmoid(x):
    return 1/(1+np.exp(-x))

# Производная функции активации
def deriv_sigmoid(x):
    fx = sigmoid(x)
    return fx*(1-fx)

# Расчет среднеквадратичной ошибки
def mse_loss(y_true: np.array, y_pred: np.array):
    return ((y_true - y_pred)**2).mean()


class LearningNetwork:



    def __init__(self):
        # Вес
        self.w1 = 1
        self.w2 = 1
        self.w3 = 1
        self.w4 = 1
        self.w5 = 1
        self.w6 = 1

        # Смещения
        self.b1 = 0
        self.b2 = 0
        self.b3 = 0

        self.L = list()

    def feedforward(self, x: np.array) -> float:
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = sigmoid(self.w2 * x[0] + self.w2 * x[1] + self.b2)
        o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
        return o1

    def train(self, data, all_y_trues):
        learn_rate = 0.1
        epochs = 1000

        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                h1 = sigmoid(sum_h1)

                sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                h2 = sigmoid(sum_h2)

                sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
                o1 = sigmoid(sum_o1)
                y_pred = o1

                # Подсчет частных производных
                d_L_d_ypred = -2 * (y_true - y_pred)

                # Нейрон o1
                d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1)
                d_ypred_d_w6 = h2 * deriv_sigmoid(sum_o1)
                d_ypred_d_b3 = deriv_sigmoid(sum_o1)

                d_ypred_d_h1 = self.w5 * deriv_sigmoid(sum_o1)
                d_ypred_d_h2 = self.w6 * deriv_sigmoid(sum_o1)

                # Нейрон h1
                d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
                d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
                d_h1_d_b1 = deriv_sigmoid(sum_h1)

                # Нейрон h2
                d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2)
                d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2)
                d_h2_d_b2 = deriv_sigmoid(sum_h2)

                # Обновляем вес и смещения
                # Нейрон h1
                self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

                # Нейрон h2
                self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
                self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

                # Нейрон o1
                self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5
                self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6
                self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3

                # Общие потери после каждой фазы
                if epoch % 10 == 0:
                    y_preds = np.apply_along_axis(self.feedforward, axis=1, arr=data)
                    loss = mse_loss(y_true=all_y_trues, y_pred=y_preds)
                    self.L.append(loss)
                    print(f'Epoch: {epoch}, Loss: {loss}')
                    print(f'w1 = {self.w1}')
                    print(f'w2 = {self.w2}')
                    print(f'w3 = {self.w3}')
                    print(f'w4 = {self.w4}')
                    print(f'w5 = {self.w5}')
                    print(f'w6 = {self.w6}')
                    print(f'b1 = {self.b1}')
                    print(f'b2 = {self.b2}')
                    print(f'b3 = {self.b3}')


# Выборка
data = np.array([
    [-2, -1], # Alice
    [25, 6], # Bob
    [17, 4], # Charlie
    [-15, -6], # Diana
])

all_y_trues = np.array([
    1, # Alice
    0, # Bob
    0, # Charlie
    1, # Diana
])

check_data = np.array([
    [-14, -7, 1],
    [30, -4, 1],
    [19, 1, 1],
    [43, 5, 0],
    [92, 9, 0]
])

if __name__ == '__main__':
    network = LearningNetwork()
    network.train(data, all_y_trues)

    # Предсказания
    emily = np.array([-7, -3])
    frank = np.array([20, 2])

    print(f'Emily: {network.feedforward(emily)}')
    print(f'Frank: {network.feedforward(frank)}')

    plt.plot(network.L)
    plt.grid(True)
    plt.show()

    for d in check_data:
        print(d[:-1])
        print(f"Checking {network.feedforward(d[:-1])} should be {d[-1]}")

