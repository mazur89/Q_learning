import json
import os
import numpy as np
import csv
import matplotlib.pyplot as plt

path = "/home/przemek/my_tensorflow/cartpole/save/"
wins_path = path + "wins.csv"

rows = []

with open(wins_path, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        rows.append(row)
        
headers = rows[0]
data = rows[1:]

aggregate = []
for i in range(1,8):
    aggregate.append([])
    for j in sorted(set([d[i] for d in data])):
        idx = [k for k in range(len(data)) if data[k][i] == j]
        count = len(idx)
        print(j)
        mean = sum([int(data[k][0]) for k in idx]) / count
        stdev = (sum([(int(data[k][0]) - mean) ** 2 for k in idx]) / (count - 1)) ** 0.5
        aggregate[-1].append([j, count, mean, stdev])
        
print(aggregate)

with open(path + "stats.csv", 'w') as f:
    writer = csv.writer(f)
    for i in range(7):
        writer.writerow([headers[i + 1], 'count', 'mean', 'stdev'])
        for agg in aggregate[i]:
            writer.writerow(agg)
            
gap = len(data) / 3

idx = []
height = []
width = []
colors = []
stdev = []
next_idx = 0
for agg in aggregate:
    idx.append([])
    for i in range(len(agg)):
        idx[-1].append(next_idx + agg[i][1] / 2)
        height.append(agg[i][2])
        width.append(agg[i][1])
        colors.append(1 - (2 * i + 1) / (4 * len(agg)))
        stdev.append(agg[i][3])
        next_idx += agg[i][1]
    next_idx += gap

fig, ax = plt.subplots()
ax.barh(sum(idx, []), height, width, color=[str(c) for c in colors], xerr = stdev, ecolor = [str(c + 0.15 * np.sign(0.7 - c)) for c in colors])
ax.set_yticks([sum(i) / len(i) for i in idx])
ax.set_yticklabels(headers[1:8])
ax.invert_yaxis()
ax.set_title('Average winning time with different settings')
ax.set_xlabel('Timesteps until win')
for i in range(len(aggregate)):
    for j in range(len(aggregate[i])):
        ax.text(30000, idx[i][j], aggregate[i][j][0], va = 'center', ha = 'center')

fig.subplots_adjust(left=0.3)
plt.show()
