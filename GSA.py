from pprint import pprint
from matplotlib import pyplot as plt

import numpy as np


class GSA:
    def __init__(self, pop_size=1, dimension=1, max_iter=100):
        self.cost_func = None

        self.alpha = 0.1
        self.G = 0.9
        self.max_iter = max_iter
        self.pop_size = pop_size
        self.dimension = dimension
        self.best_so_far = None

        self.X = np.random.rand(pop_size, dimension)
        self.V = np.random.rand(pop_size, dimension)
        self.f = np.full((pop_size, dimension), None)  # None  # Will become a list inside cal_f
        self.a = np.full((pop_size, dimension), None)
        self.q = np.full((pop_size, 1), None)
        self.M = np.full((pop_size, 1), None)
        self.cost_matrix = np.full((pop_size, 1), None)

    # Evaluate a single x (x_i)
    def evaluate(self, args):
        return self.cost_func(args)

    # Generate the cost of all particles
    def gen_cost_matrix(self):
        for i, x in enumerate(self.X):
            self.cost_matrix[i] = self.evaluate(x)

    def cal_q(self):

        best = np.min(self.cost_matrix)
        worst = np.max(self.cost_matrix)

        self.q = (self.cost_matrix - worst) / best - worst

    def cal_m(self):
        self.M = self.q / np.sum(self.q)

    def cal_f(self):
        costs = self.cost_matrix.copy()
        costs.sort(axis=0)
        costs = costs

        for i in range(self.pop_size):
            f = None
            for cost in costs:
                j = int(np.where(self.cost_matrix == cost)[0])

                dividend = float(self.M[i] * self.M[j])
                divisor = np.sqrt(np.sum((self.X[i] - self.X[j]) ** 2)) + np.finfo('float').eps
                if f is None:
                    f = self.G * (dividend / divisor) * (self.X[j] - self.X[i])
                else:
                    f = f + self.G * (dividend / divisor) * (self.X[j] - self.X[i])

            self.f[i] = np.random.uniform(0, 1) * f

    def cal_a(self):
        for i in range(self.pop_size):
            self.a[i] = self.f[i] / self.M[i]

    def cal_v(self):
        self.V = (np.random.uniform(0, 1) * self.V) + self.a

    def move(self):
        self.X = self.X + self.V

    def update_g(self, iteration):
        self.G = self.G * np.e ** (- self.alpha * (iteration / self.max_iter))

    def show_results(self):
        print('Best seen so far is located at:', self.best_so_far, 'Cost:', self.evaluate(self.best_so_far))

    def update_best_so_far(self):
        best = np.min(self.cost_matrix)
        index = int(np.where(self.cost_matrix == best)[0])
        if self.best_so_far is None or self.evaluate(self.best_so_far) > self.evaluate(self.X[index]):
            self.best_so_far = self.X[index]

    def start(self):
        _iter = 0
        avg = []
        bsf = []
        while _iter < self.max_iter:
            self.gen_cost_matrix()
            self.cal_q()
            self.cal_m()
            self.update_g(_iter)
            self.cal_f()
            self.cal_a()
            self.cal_v()
            self.move()
            self.update_best_so_far()

            avg.append(sum(self.cost_matrix) / len(self.cost_matrix))
            bsf.append(self.evaluate(self.best_so_far))

            _iter += 1
        else:
            self.show_results()
            plt.figure()
            avg_plt = plt.subplot(1, 2, 1)
            avg_plt.plot(avg, color='gold')
            bsf_plt = plt.subplot(1, 2, 2)
            bsf_plt.plot(bsf, color='darkviolet')
            plt.show()


# Minimize function
def rosenbrock(args):
    x, y = args
    a = 1
    b = 100
    return (a - x) ** 2 + b * (y - x ** 2) ** 2


gsa = GSA(pop_size=20, dimension=2, max_iter=100)
gsa.cost_func = rosenbrock
gsa.start()
