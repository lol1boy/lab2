import csv
import os
import matplotlib.pyplot as plt
import numpy as np

def read_data(filename):
    x = []
    y = []

    base_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(base_dir, filename)

    with open(filepath, 'r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            x.append(float(row['rps']))
            y.append(float(row['cpu']))

    return x, y

x1, y1 = read_data("data.csv")
 
print("x:", x1) 
print("y:", y1)

x = np.array(x1)
y = np.array(y1)


def divided_diff(x, y):
    n = len(y)
    coef = np.copy(y).astype(float)
    for j in range(1, n):
        print(coef[j-1:n])
        coef[j:n] = (coef[j:n] - coef[j-1:n-1]) / (x[j:n] - x[0:n-j])
    return coef

coef = divided_diff(x, y)

def newton_polynomial(coef, x_data, x):
    n = len(coef)
    p = coef[-1]

    for k in range(2, n+1):
        p = coef[-k] + (x - x_data[-k]) * p
    return p

x_vals = np.linspace(50, 800, 5)
y_vals = [newton_polynomial(coef, x, xi) for xi in x_vals]

x_new = 600
y_new = newton_polynomial(coef, x, x_new)
print("New point:")
print("x =", x_new)
print("y =", y_new)


plt.plot(x, y, 'ro', label="Вузли")
plt.plot(x_vals, y_vals, 'b-', label="Многочлен Ньютона")
plt.plot(x_new, y_new, 'go', label="Нова точка (600)")
plt.legend()
plt.grid()
plt.show()