import json

n_hids = [[64], [32, 16]]
lrs = [5e-4, 2e-4]
ts = [1000, 2000]
prioritize = [[False, 0, 0], [True, 0.3, 0.3], [True, 0.7, 0.7]]
kappas = [[0, 0, 0], [0.01, 0.01, 0], [0.1, 0.1, 0]]
gammas = [1, 0.95]
ts2 = [1, 2]

variables = [n_hids, lrs, ts, prioritize, kappas, gammas, ts2, [False, True]]
idxes = [0] * (len(variables))

data_to_save = []
while idxes[-1] == 0:
    data_to_save.append([variables[i][idxes[i]] for i in range(len(variables))])
    print(data_to_save[-1])
    
    idxes[0] += 1
    current = 0
    while idxes[current] == len(variables[current]):
        idxes[current] = 0
        current += 1
        idxes[current] += 1

with open('/home/przemek/my_tensorflow/cartpole/training_params.json', 'w') as f:
    json.dump(data_to_save, f)

exit()
