import csv
import os

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

x, y = read_data("data.csv") 
 
print("x:", x) 
print("y:", y)