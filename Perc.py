import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap

class Perceptron(object):
    def __init__(self, eta=0.1, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    '''
    Выполнить подгонку модели под тренировочные данные.
    Параметры
    Х - тренировочные данные: массив,
    размерность Х[n_samples, n_features] где 
                    n_samples - число образцов
                    n_features - число признаков
                    
    Y - целевые значения: массив размерностью Y[n_samples]
    '''

    def fit(self, X, y):
        # w_ – одномерный массив: веса после обучения
        self.w_ = np.zeros(1 + X.shape[1])

        # errors_ – список ошибок классификации в каждой эпохе
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                flag = update != 0.0
                errors += int(flag)
                self.errors_.append(errors)

        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)

def plot_decision_regions(X, y, classifier, resolution=0.02):
    # Генератор маркеров и палитра
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'green', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # Поверхность решения
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl,)


if __name__ == '__main__':
    df = pd.read_csv('Iris.csv')
    # Выборка из df 100 строк 1 и 3 столбец
    X = df.iloc[0:100, [1, 3]].values

    # Выборка из df 100 строк 4 столбец
    y = df.iloc[0:100, 5].values
    # Преобразование названий
    y = np.where(y=='Iris-setosa', -1, 1)

    # Первые 50 элементов обучающей выборки
    # Строки 0-50 столбцы 0, 1
    plt.scatter(X[0:50, 0], X[0:50, 1], color='red', marker='o', label='Iris-setosa')

    # Следующие 50 элементов
    plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='Iris-versicolor')
    plt.xlabel('Длина чашелистника')
    plt.ylabel('Длина липестка')
    plt.legend(loc='best')
    plt.show()

    # Обучение
    ppn = Perceptron(eta=0.1, n_iter=100)
    ppn.fit(X, y)
    plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
    plt.xlabel('Эпохи')
    plt.ylabel('Число случаев ошибочной классификации')
    plt.show()

    # Классификация
    i1 = [5.5, 1.6]
    i2 = [6.4, 4.5]

    R1 = ppn.predict(i1)
    R2 = ppn.predict(i2)

    if(R1 == -1):
        print("R1 – Iris-setosa")
    else:
        print("R1 – Iris-versicolor")

    # Диаграмма классификации
    plot_decision_regions(X, y, classifier=ppn)
    plt.xlabel('Длина чашелистника')
    plt.ylabel('Длина липестка')
    plt.legend(loc='best')
    plt.show()
