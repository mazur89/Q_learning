import json
import os
import matplotlib.pyplot as plt
import numpy as np
import csv

params = ['n_hid', 'lr', 'timesteps_per_update_target', 'prioritize', 'kappa', 'gamma', 'timesteps_per_action_taken']
def process(s):
    idx = [s.find(p) - 1 for p in params] + [None]
    res = [s[idx[i] + len(params[i]) + 2 : idx[i + 1]].split('_') for i in range(len(params))]
    return [r[min([len(r), 2]) - 1] for r in res]

path = "/home/przemek/my_tensorflow/acrobot/save/"
ld = sorted(os.listdir(path))

min_avg_100_runs = []
steps_until_win = []

for i, s in zip(range(len(ld)), ld):
    
    if s == 'wins.csv':
        continue
    
    print('%d / %d' % (i, len(ld)))

    progress_path = path + s + '/progress.json'

    with open(progress_path, 'r') as f:
        data = json.load(f)
    
    keys = list(data.keys())

    episode_length = data['episode_length']
    
    steps_until_win.append(min([sum(episode_length[:j]) for j in range(len(episode_length)) if j > 99 and sum(episode_length[j - 100 : j]) < 11000] + [np.inf]))
            
    min_avg_100_runs.append(0.01 * min([sum(episode_length[j : j + 100]) for j in range(len(episode_length) - 99)]))
    
with open(path + 'wins.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['timesteps_until_win'] + params)
    for t, s in sorted(zip(steps_until_win, ld)):
        if t < np.inf:
            writer.writerow([t] + process(s))

for t, s in sorted(zip(steps_until_win, ld)):
    if t < np.inf:
        print("%d %s" % (t, process(s)))
        
d = {j: len([m for m in min_avg_100_runs if m < j]) for j in range(100, 200)}
#print(d)
        
#plt.hist(min_avg_100_runs)
#plt.show()