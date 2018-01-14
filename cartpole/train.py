import json
from utils import train_model_and_save_results

names = ['n_hid', 'lr', 'timesteps_per_update_target', 'prioritize', 'kappa', 'gamma', 'timesteps_per_action_taken']

while True:
    params_path = '/home/przemek/my_tensorflow/cartpole/training_params.json'
    with open(params_path, 'r') as f:
        data = json.load(f)
        f.close()
    
    current = 0
    while current < len(data) and data[current][-1]:
        current += 1
        
    print(current)
        
    if current == len(data):
        break
    
    data_current = data[current]
    
    path = '/home/przemek/my_tensorflow/cartpole/save/'
    
    for i in range(len(names)):
        
        path += names[i] + '_'
        
        if isinstance(data_current[i], list):
            for d in data_current[i]:
                path += str(d) + '_'
        else:
            path += str(data_current[i]) + '_'
            
    path = path[:-1] + '/'
                
    print(path)
    
    train_model_and_save_results(
        env_name = 'CartPole-v0', 
        n_hid = data_current[0], 
        lr = data_current[1], 
        eps_min = 0.02, 
        delta = 5e-5, 
        gamma = data_current[5], 
        kappa = data_current[4],
        prioritize = data_current[3][0],
        alpha = data_current[3][1],
        beta = data_current[3][2],
        timesteps_per_update_target = data_current[2],
        timesteps_per_action_taken = data_current[6],
        total_timesteps = 100000,
        perturb = False,
        folder_path = path)
         
    
    data[current][-1] = True
    
    print(data[current])
    
    with open(params_path, 'w') as f:
        json.dump(data, f)
        f.close()
    
    
