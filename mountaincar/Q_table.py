import gym
import numpy as np
import tensorflow as tf
import json
import cloudpickle
import os
from baselines.deepq.replay_buffer import PrioritizedReplayBuffer

def refine_Q_table(Q_table, N, n=2):
    
    tmp = np.zeros((N * n, N * n, 3))
    for i in range(N * n):
        for j in range(N * n):
            for k in range(3):
                tmp[i][j][k] = Q_table[(int)(i / n)][(int)(j / n)][k]

    return tmp, N * n

def fill_Q_table(initial_size = 4,
                 total_timesteps = 25600,
                 refinement_constant = 100,
                 gamma = 0.99,
                 lr = 0.1):
    
    env = gym.make('MountainCar-v0')
    high = env.observation_space.high
    low = env.observation_space.low
    
    def preprocess_obs(n):
        def f(obs):
            res = ((obs - low) / (high - low) * (n - 1e-10))
            return [[x] for x in res.astype('int32')]
        return f
    
    N = initial_size
    Q_table = Q_table = np.zeros((N, N, 3))
    memory = []
    
    episode_rew = 0
    obs = env.reset()
    
    for t in range(total_timesteps):
                
        if t == refinement_constant * N * N:
        
            Q_table, N = refine_Q_table(Q_table, N)
        
            print('updated N = %d' % N)
                
        action = Q_table[preprocess_obs(N)(obs)].argmax()
        next_obs, rew, done, _ = env.step(action)
        episode_rew += rew
    
        memory.append((obs, action, rew, next_obs, done))
        if len(memory) > 50000:
            del(memory[0])
    
        if done:
            obs = env.reset()
        
            print('episode reward = %d' % episode_rew)
            episode_rew = 0
                
        else:
            obs = next_obs
    
        if len(memory) > 0:        
            idxes = [np.random.randint(len(memory)) for _ in range(32)]        
            tuples = [memory[idx] for idx in idxes]
        
            for s in tuples:
                Q_table[preprocess_obs(N)(s[0]) + [s[1]]] += lr * (s[2] + (1 - s[4]) * gamma * Q_table[preprocess_obs(N)(s[3])].max() - Q_table[preprocess_obs(N)(s[0]) + [s[1]]])
    
        if t % 1000 == 0:
            print('t = %d' % t)
            
    return Q_table, memory, N, env, high, low

def Q_model(n_hid, activation):
    
    W_0 = tf.get_variable("W_0", [2, n_hid])
    W_1 = tf.get_variable("W_1", [n_hid, 3])
    W_state = tf.get_variable("W_state", [n_hid, 6])
    
    b_0 = tf.get_variable("b_0", [n_hid], initializer = tf.zeros_initializer())
    b_1 = tf.get_variable("b_1", [3], initializer = tf.zeros_initializer())
    b_state = tf.get_variable("b_state", [6], initializer = tf.zeros_initializer())
    
    def Q_function(inpt):
        hid = activation(tf.matmul(inpt, W_0) + b_0)
        out = tf.matmul(hid, W_1) + b_1
        state = tf.reshape(tf.matmul(hid, W_state) + b_state, [-1, 2, 3])
        
        return out, state
    
    return [W_0, W_1, W_state, b_0, b_1, b_state], Q_function

def Huber_loss(x, delta=1.0):
    return tf.where(tf.abs(x) < delta, 0.5 * tf.square(x), delta * (tf.abs(x) - 0.5 * delta))

def run_mountaincar_and_save_results(lr,
                                     kappa,
                                     timesteps_per_update_target,
                                     timesteps_per_action_taken,
                                     gamma,
                                     prioritize,
                                     alpha,
                                     beta,
                                     folder_path):
    
    episode_length = [0]
    Q_errors = []
    state_errors = []
    grad_sums_of_squares = []
    
    with tf.variable_scope("Q"):
        Q_params, Q_function = Q_model(256, tf.nn.softplus)
    
    with tf.variable_scope('Q_target'):
        Q_params_target, Q_function_target = Q_model(256, tf.nn.softplus)
    
    obses = tf.placeholder(tf.float32, shape = [None, 2])
    actions = tf.placeholder(tf.int32, shape = [None])
    rewards = tf.placeholder(tf.float32, shape = [None])
    next_obses = tf.placeholder(tf.float32, shape = [None, 2])
    dones = tf.placeholder(tf.float32, shape = [None])
    weights = tf.placeholder(tf.float32, shape = [None])
    Q_values_target = tf.placeholder(tf.float32, shape = [None])
    
    Q_function_obses = Q_function(obses)
    Q_values_per_action = Q_function_obses[0]
    Q_difference = tf.reduce_sum(Q_values_per_action * tf.one_hot(actions, 3), axis = 1) - Q_values_target
    state_prediction = Q_function_obses[1]

    if prioritize:
        Q_error = tf.reduce_mean(tf.square(Q_difference) * weights)
        state_error = tf.reduce_mean(tf.square(tf.reduce_sum(state_prediction * tf.expand_dims(tf.one_hot(actions, 3), 1), axis = 2) - next_obses) * tf.expand_dims(weights, 1))
    else:
        Q_error = tf.reduce_mean(tf.square(Q_difference))
        state_error = tf.reduce_mean(tf.square(tf.reduce_sum(state_prediction * tf.expand_dims(tf.one_hot(actions, 3), 1), axis = 2) - next_obses))
     
    total_error = Q_error
    if kappa > 0:
        total_error += kappa * state_error

    Q_actions = tf.argmax(Q_values_per_action, axis = 1)

    Q_values_target_Bellman = rewards + (1 - dones) * gamma * tf.reduce_sum(tf.one_hot(tf.argmax(Q_function(next_obses)[0], axis = 1), 3) * Q_function_target(next_obses)[0], axis = 1)

    update_target = tf.group(*[tf.assign(Q_param_target, Q_param) for Q_param, Q_param_target in zip(Q_params, Q_params_target)])

    lr_variable = tf.get_variable('lr', (), initializer = tf.constant_initializer(0.1))
    
    grads = tf.gradients(total_error, Q_params)
    grad_sum_of_squares = sum([tf.reduce_sum(x * x) for x in grads if x is not None])

    Q_Adam = tf.train.AdamOptimizer(learning_rate = lr_variable)
    Q_minimize = Q_Adam.minimize(Q_error)
    total_minimize = Q_Adam.minimize(total_error)

    sess = tf.Session()

    sess.run(tf.global_variables_initializer())
    
    Q_table, memory, N, env, high, low = fill_Q_table()

    obses_valid_0 = np.array(sum([[i] * N * 3 for i in range(N)], []))
    obses_valid_1 = np.array(sum([[i] * 3 for i in range(N)], []) * N)
    actions_valid = np.array([0, 1, 2] * N * N)
    obses_valid = (np.stack((obses_valid_0, obses_valid_1), axis = 1) + 0.5) / N * (high - low) + low
    Q_values_target_valid = Q_table[obses_valid_0, obses_valid_1, actions_valid]
    weights_valid = np.ones(N * N * 3)

    def valid_error():
        return sess.run(Q_error, feed_dict = {
            obses: obses_valid,
            actions: actions_valid,
            Q_values_target: Q_values_target_valid, 
            weights: weights_valid})

    valid_error_current = 1e20
    valid_error_new = valid_error()

    while valid_error_new < 0.999 * valid_error_current:
    
        valid_error_current = valid_error_new
        print('valid error = %.6f' % valid_error_current)    
        sess.run(tf.assign(lr_variable, valid_error_current / 1000))
    
        for _ in range(64):
        
            sess.run(Q_minimize, feed_dict = {
                obses: obses_valid,
                actions: actions_valid,
                Q_values_target: Q_values_target_valid,
                weights: weights_valid})
            
        valid_error_new = valid_error()
    
        print('valid error new = %.6f' % valid_error_new)
        
    sess.run(tf.assign(lr_variable, lr))
    
    obs = env.reset()
    
    if prioritize:
        replay_buffer = PrioritizedReplayBuffer(50000, alpha)
        for mem in memory:
            replay_buffer.add(*mem)
    
    episode_rew = 0
        
    for t in range(100000):
                    
        if t % timesteps_per_action_taken == 0:
            action = sess.run(Q_actions, feed_dict = {obses: obs[None]})[0]        
            next_obs, rew, done, _ = env.step(action)    
            episode_rew += rew    
            if prioritize:
                replay_buffer.add(obs, action, rew, next_obs, done)
            else:
                memory.append((obs, action, rew, next_obs, done))
                if len(memory) > 50000:
                    del memory[0]
            obs = next_obs
            episode_length[-1] += 1
                
            if done:
                obs = env.reset()        
                print('episode reward = %d' % episode_rew)
                episode_rew = 0
                episode_length.append(0)
        
        if prioritize:
            
            beta_current = (beta * (100000 - t) + t) / 100000
            obses_current, actions_current, rewards_current, next_obses_current, dones_current, weights_current, idxes_current = replay_buffer.sample(32, beta_current)
            
        else:
            idxes = [np.random.randint(len(memory)) for _ in range(32)]
            tuples = [memory[idx] for idx in idxes]
    
            obses_current = np.array([s[0] for s in tuples])
            actions_current = np.array([s[1] for s in tuples])
            rewards_current = np.array([s[2] for s in tuples])
            next_obses_current = np.array([s[3] for s in tuples])
            dones_current = np.array([float(s[4]) for s in tuples])
            weights_current = np.ones(32)
                
        Q_values_target_current = sess.run(Q_values_target_Bellman, feed_dict = {
            rewards: rewards_current,
            next_obses: next_obses_current,
            dones: dones_current})
        
        if prioritize:
            new_weights = np.abs(sess.run(Q_difference, feed_dict = {
                obses: obses_current,
                actions: actions_current,
                Q_values_target: Q_values_target_current,
                next_obses: next_obses_current})) + 1e-6
            replay_buffer.update_priorities(idxes_current, new_weights)
        
        Q_errors.append(sess.run(Q_error, feed_dict = {
            obses: obses_current,
            actions: actions_current,
            Q_values_target: Q_values_target_current,
            next_obses: next_obses_current,
            weights: weights_current}).astype(np.float64))
        
        state_errors.append(sess.run(state_error, feed_dict = {
            obses: obses_current,
            actions: actions_current,
            Q_values_target: Q_values_target_current,
            next_obses: next_obses_current,
            weights: weights_current}).astype(np.float64))
        
        grad_sums_of_squares.append(sess.run(grad_sum_of_squares, feed_dict = {
            obses: obses_current,
            actions: actions_current,
            Q_values_target: Q_values_target_current,
            next_obses: next_obses_current,
            weights: weights_current}).astype(np.float64))
        
        sess.run(total_minimize, feed_dict = {
            obses: obses_current,
            actions: actions_current,
            Q_values_target: Q_values_target_current,
            next_obses: next_obses_current,
            weights: weights_current})
            
        if t % timesteps_per_update_target == 0:
            sess.run(update_target)
    
        if t % 1000 == 0:
            print('t = %d' % t)
    
    print('saving progress and params...')
    
    if not os.path.exists(folder_path + 'params/'):
        os.makedirs(folder_path + 'params/')
        
    with open(folder_path + 'progress.json', 'w') as f:
        data = {'episode_length': episode_length,
                   'Q_errors': Q_errors,
                   'state_errors': state_errors,
                   'grad_sums_of_squares': grad_sums_of_squares}
        
        json.dump(data, f)
    
    saver = tf.train.Saver({v.name: v for v in Q_params})
    saver.save(sess, folder_path + 'params/params.ckpt')
    
    with open(folder_path + 'params/params.pkl', 'wb') as f:
        cloudpickle.dump([sess.run(param) for param in Q_params], f)
    
    print('saved...')
    
    # tidy up
    
    sess.close()
    tf.reset_default_graph()
            
if __name__ == '__main__':
    
    names = ['lr', 'kappa', 'timesteps_per_update_target', 'timesteps_per_action_taken', 'gamma', 'prioritize']

    while True:
        params_path = '/home/przemek/my_tensorflow/mountaincar/training_params.json'
        with open(params_path, 'r') as f:
            data = json.load(f)
            f.close()
    
        current = 0
        while current < len(data) and (data[current][-1] or False):
            current += 1
        
        print(current)
        
        if current == len(data):
            break
    
        data_current = data[current]
    
        path = '/home/przemek/my_tensorflow/mountaincar/save/'
    
        for i in range(len(names)):
        
            path += names[i] + '_'
        
            if isinstance(data_current[i], list):
                for d in data_current[i]:
                    path += str(d) + '_'
            else:
                path += str(data_current[i]) + '_'
            
        path = path[:-1] + '/'
                
        print(path)
    
        run_mountaincar_and_save_results(lr = data_current[0],
                                        kappa = data_current[1],
                                        timesteps_per_update_target = data_current[2],
                                        timesteps_per_action_taken = data_current[3],
                                        gamma = data_current[4],
                                        prioritize = data_current[5][0],
                                        alpha = data_current[5][1],
                                        beta = data_current[5][2],
                                        folder_path = path)
         
    
        data[current][-1] = True
    
        print(data[current])
    
        with open(params_path, 'w') as f:
            json.dump(data, f)
            f.close()