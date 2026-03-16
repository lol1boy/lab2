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
        for i in range(n - 1, j - 1, -1):
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

def print_dd_table(x, coef):
    n = len(x)
    
    table = np.zeros((n, n))
    table[:, 0] = y
    
    temp = np.copy(y).astype(float)
    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            temp[i] = (temp[i] - temp[i - 1]) / (x[i] - x[i - j])
        table[j:, j] = temp[j:]
    
    print(f"{'x':>8}", end="")
    for k in range(n):
        print(f"  order {k}", end="")
    print()
    print("─" * (8 + 10 * n))
    
    for i in range(n):
        print(f"{x[i]:>8.0f}", end="")
        for j in range(n - i):
            print(f"  {table[i][j]:>8.5f}", end="")
        print()

print_dd_table(x, coef)

def finite_differences(y):
    n = len(y)
    delta = list(y)
    coeffs = [delta[0]]
    for k in range(1, n):
        delta = [delta[i+1] - delta[i] for i in range(len(delta)-1)]
        coeffs.append(delta[0])
    return coeffs

def factorial_poly(x_data, y_data, x_val):
    n = len(x_data)
    h = (x_data[-1] - x_data[0]) / (n - 1)
    t = (x_val - x_data[0]) / h

    coeffs = finite_differences(y_data)

    result = 0.0
    binom = 1.0
    factorial = 1.0
    for k in range(n):
        result += binom * coeffs[k] / factorial
        binom *= (t - k)
        factorial *= (k + 1)
    return result

print(f"Factorial:  CPU(600) = {factorial_poly(x, y, 600):.4f} %")


def run_node_study(x_all, y_all, target=600):
    print(f"\n{'Nodes':<8} {'Newton':>12} {'Factorial':>12} {'Error N':>12} {'Error F':>12}")
    print("─" * 58)

    dd_ref = divided_diff(x_all, y_all)
    ref = newton_polynomial(dd_ref, x_all, target)

    for n_nodes in [3, 4, 5]:
        indices = np.round(np.linspace(0, len(x_all)-1, n_nodes)).astype(int)
        xn = x_all[indices]
        yn = y_all[indices]

        dd = divided_diff(xn, yn)
        newton_pred   = newton_polynomial(dd, xn, target)
        factorial_pred = factorial_poly(list(xn), list(yn), target)

        err_n = abs(newton_pred   - ref)
        err_f = abs(factorial_pred - ref)

        print(f"{n_nodes:<8} {newton_pred:>12.4f} {factorial_pred:>12.4f} "
              f"{err_n:>12.4f} {err_f:>12.4f}")

run_node_study(x, y)


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