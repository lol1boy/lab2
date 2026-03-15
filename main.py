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
    # for j in range(1, n):
    #     print(coef[j-1:n])
        # coef[j:n] = (coef[j:n] - coef[j-1:n-1]) / (x[j:n] - x[0:n-j])
    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):          # right → left
            coef[i] = (coef[i] - coef[i - 1]) / (x[i] - x[i - j])
    return coef

coef = divided_diff(x, y)
print("\nDivided-difference coefficients:")
for k, c in enumerate(coef):
    print(f"  c[{k}] = {c:.8f}")

def newton_polynomial(coef, x_data, x_val):
    n = len(coef)
    p = coef[n - 1]
    for k in range(n - 2, -1, -1):
        p = coef[k] + (x_val - x_data[k]) * p
    return p

x_new = 600
y_new = newton_polynomial(coef, x, x_new)
print(f"\nPrediction:  CPU({x_new} RPS) = {y_new:.4f} %")
print("New point:")
print("x =", x_new)
print("y =", y_new)

x_vals = np.linspace(50, 800, 50)
y_vals = [newton_polynomial(coef, x, xi) for xi in x_vals]

plt.figure(figsize=(9, 5))
plt.plot(x, y, 'ro', label="Вузли")
plt.plot(x_vals, y_vals, 'b-', label="Многочлен Ньютона")
plt.plot(x_new, y_new, 'go', label="Нова точка (600)")
plt.legend()
plt.grid()
plt.show()