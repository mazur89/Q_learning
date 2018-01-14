import tensorflow as tf
import numpy as np
import gym
import os
import json
import cloudpickle
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from datetime import datetime


def Q_model(sizes):
    
    # sizes = list of sizes of consecutive layers
    
    weights = []
    biases = []
    
    for i in range(len(sizes) - 1):
        
        weights.append(tf.get_variable('weight_' + str(i + 1), shape = [sizes[i], sizes[i + 1]]))
        biases.append(tf.get_variable('bias_' + str(i + 1), shape = [sizes[i + 1]], initializer = tf.zeros_initializer()))
        
    weight_state_pred = tf.get_variable('weight_state_pred', shape = [sizes[i], sizes[0] * sizes[-1]])
    bias_state_pred = tf.get_variable('bias_state_pred', shape = [sizes[0] * sizes[-1]])
    weight_reward_pred = tf.get_variable('weight_reward_pred', shape = [sizes[i], sizes[-1]])
    bias_reward_pred = tf.get_variable('bias_reward_pred', shape = [sizes[-1]])
    weight_done_pred = tf.get_variable('weight_done_pred', shape = [sizes[i], sizes[-1]])
    bias_done_pred = tf.get_variable('bias_done_pred', shape = [sizes[-1]])
        
    def Q_function(inpt):
        
        # inpt = tf.placeholder for the observations
        
        out = inpt
        
        for i in range(len(weights)):
            
            out = tf.matmul(out, weights[i]) + biases[i]
            
            if i < len(weights) - 2:
                out = tf.nn.softplus(out)
            
            if i == len(weights) - 2:
                out = tf.nn.tanh(out)
                
                state_pred = tf.matmul(out, weight_state_pred) + bias_state_pred
                state_pred = tf.reshape(state_pred, [-1, sizes[-1], sizes[0]])
                
                reward_pred = tf.matmul(out, weight_reward_pred) + bias_reward_pred
                
                done_pred = tf.matmul(out, weight_done_pred) + bias_done_pred
                
        # returns predictions for Q values, next state, single step reward and whether the episode is finished
                
        return out, state_pred, reward_pred, done_pred
    
    # returns list of parameters and Q_function
    
    return weights + biases + [weight_state_pred, bias_state_pred, weight_reward_pred, bias_reward_pred, weight_done_pred, bias_done_pred], Q_function



def Huber_loss(x, delta=1.0):
    return tf.where(tf.abs(x) < delta, 0.5 * tf.square(x), delta * (tf.abs(x) - 0.5 * delta))



def train_model_and_save_results(
    env_name, 
    n_hid, 
    lr, 
    eps_min, 
    delta, 
    gamma, 
    kappa,
    prioritize,
    alpha,
    beta,
    timesteps_per_update_target, 
    timesteps_per_action_taken,
    total_timesteps,
    perturb,
    folder_path):
    
    # env_name: environment name
    # n_hid: list of numbers of units in hidden layers
    # eps_min: final exploration epsilon
    # delta: linear decrement of exploration epsilon per timestep
    # kappa: list of 3 constants for next state, reward and done predictions
    # prioritize: boolean, whether to use prioritized reply buffer or not
    # alpha: prioritization constant
    # beta: weight correction constant
    # perturb: boolean, whether to use parameter noise explotration
    # folder_path: to save results
    
    env = gym.make(env_name)
    
    eps = tf.get_variable('eps', (), initializer=tf.constant_initializer(1.0), dtype=tf.float32)
    update_eps = tf.assign(eps, tf.maximum(eps - delta, eps_min))
    
    n_in = 1
    for n in env.observation_space.shape:
        n_in *= n
        
    n_out = env.action_space.n
    
    sizes = [n_in] + n_hid + [n_out]
    
    # create networks
    
    with tf.variable_scope('Q'):
        Q_params, Q_function = Q_model(sizes)
        
    with tf.variable_scope('Q_target'):
        Q_params_target, Q_function_target = Q_model(sizes)
    
    if perturb:
        
        with tf.variable_scope('Q_perturbed'):
            Q_params_perturbed, Q_function_perturbed = Q_model(sizes)
            
        with tf.variable_scope('Q_adapt'):
            Q_params_adapt, Q_function_adapt = Q_model(sizes)
        
        perturbation_scale = tf.get_variable("perturbation_scale", (), initializer=tf.constant_initializer(0.01))
        larger_perturbation_scale = perturbation_scale * 1.01
        smaller_perturbation_scale = perturbation_scale / 1.01
    
    # create placeholders
    
    obses = tf.placeholder(tf.float32, shape = [None, n_in])
    actions = tf.placeholder(tf.int32, shape = [None])
    rewards = tf.placeholder(tf.float32, shape = [None])
    next_obses = tf.placeholder(tf.float32, shape = [None, n_in])
    dones = tf.placeholder(tf.float32, shape = [None])
    weights = tf.placeholder(tf.float32, shape = [None])
    
    # create outputs of Q functions
    
    Q_function_obses = Q_function(obses)

    Q_values_per_action = Q_function_obses[0]
    Q_actions = tf.argmax(Q_values_per_action, axis=1)
    
    if perturb:
        
        ops = []
        for i in range(len(Q_params) - 6):
            ops.append(tf.assign(Q_params_perturbed[i], Q_params[i] + tf.random_normal(shape=tf.shape(Q_params[i]), mean=0., stddev=perturbation_scale)))
        assign_perturbed = tf.group(*ops)
        
        ops = []
        for i in range(len(Q_params) - 6):
            ops.append(tf.assign(Q_params_adapt[i], Q_params[i] + tf.random_normal(shape=tf.shape(Q_params[i]), mean=0., stddev=perturbation_scale)))
        assign_adapt = tf.group(*ops)
        
        Q_values_per_action_perturbed = Q_function_perturbed(obses)[0]
        Q_actions_perturbed = tf.argmax(Q_values_per_action_perturbed, axis=1)
        Q_values_per_action_adapt = Q_function_adapt(obses)[0]
        
        kl = tf.reduce_sum(tf.nn.softmax(Q_values_per_action) * (tf.log(tf.nn.softmax(Q_values_per_action)) - tf.log(tf.nn.softmax(Q_values_per_action_adapt))), axis=-1)
        kl_mean = tf.reduce_mean(kl)
        kl_eps = -tf.log(1 - eps + eps / n_out)
        with tf.control_dependencies([assign_adapt]):
            update_perturbation_scale = tf.cond(kl_mean < kl_eps, lambda: perturbation_scale.assign(perturbation_scale * 1.01), lambda: perturbation_scale.assign(perturbation_scale / 1.01))

    Q_values = tf.reduce_sum(Q_values_per_action * tf.one_hot(actions, n_out), axis=1)
    Q_values_target = rewards + gamma * tf.reduce_sum(tf.one_hot(tf.argmax(Q_function(next_obses)[0], axis=1), n_out) * Q_function_target(next_obses)[0], axis=1) * (1.0 - dones)
    
    # create errors
    
    TD = Q_values - tf.stop_gradient(Q_values_target)
    TD_error = tf.reduce_mean(weights * Huber_loss(TD))

    state_difference = tf.reduce_sum(Q_function_obses[1] * tf.expand_dims(tf.one_hot(actions, n_out), 2), axis=1) - next_obses
    state_difference_error = tf.reduce_mean(tf.expand_dims(weights, 1) * Huber_loss(state_difference))

    reward_difference = tf.reduce_sum(Q_function_obses[2] * tf.one_hot(actions, n_out), axis=1) - rewards
    reward_difference_error = tf.reduce_mean(weights * Huber_loss(reward_difference))

    done_difference = tf.nn.sigmoid_cross_entropy_with_logits(labels = dones, logits = tf.reduce_sum(Q_function_obses[3] * tf.one_hot(actions, n_out), axis=1))
    done_difference_error = tf.reduce_mean(weights * done_difference)
    
    # compute total error
    
    total_error = TD_error
    
    if kappa[0] > 0:
        total_error += kappa[0] * state_difference_error
        
    if kappa[1] > 0:
        total_error += kappa[1] * reward_difference_error
        
    if kappa[2] > 0:
        total_error += kappa[2] * done_difference_error
    
    # create gradients to save
    
    grads = tf.gradients(total_error, Q_params)
    grad_sum_of_squares = sum([tf.reduce_sum(x * x) for x in grads if x is not None])
    
    # create optimizer and update rule
    
    Adam = tf.train.AdamOptimizer(learning_rate=lr)
    update = Adam.minimize(total_error, var_list=Q_params)
    
    # initialize session
    
    sess = tf.Session()
    
    # define update_target
    
    def update_target():
        for i in range(len(Q_params)):
            sess.run(Q_params_target[i].assign(Q_params[i]))
            
    if prioritize:
        replay_buffer = PrioritizedReplayBuffer(50000, alpha)
    else:
        replay_buffer = ReplayBuffer(50000)
        
    # initialize parameters to save
    
    episode_length = [0]

    TD_errors = []
    state_difference_errors = []
    reward_difference_errors = []
    done_difference_errors = []
    grad_sums_of_squares = []
    
    # initialize all variables
    
    sess.run(tf.global_variables_initializer())
    
    # start training
    
    obs = env.reset()
    
    for t in range(total_timesteps):
                
        # choose action
        if t % timesteps_per_action_taken == 0:
            
            if (not perturb) and np.random.uniform() < sess.run(eps) :
        
                # take random action
        
                action = np.random.randint(n_out)
        
            else:
                    
                if perturb:
                
                    # take randomly perturb action, then update scale
                    action = sess.run(Q_actions_perturbed, feed_dict={obses: obs[None]})[0]
                    sess.run(update_perturbation_scale, feed_dict={obses: obs[None]})
                    
                else:
                
                    # take optimal action

                    action = sess.run(Q_actions, feed_dict={obses: obs[None]})[0]
                    
            next_obs, rew, done, _ = env.step(action)
        
            episode_length[-1] += 1
        
            # add experience to buffer
        
            replay_buffer.add(obs, action, rew, next_obs, float(done))
    
            obs = next_obs
        
            if(done):
            
                # episode finished
            
                print("episode length = " + str(episode_length[-1]))
                obs = env.reset()
                episode_length.append(0)
                if perturb:
                    sess.run(assign_perturbed)
                    print(sess.run(perturbation_scale))
            
    
        if t % timesteps_per_update_target == 0:
            
            # update target
            
            print("t = " + str(t) + ", updating target...")
            update_target()
        
        if t > 1000:        
            
            # update primary network
            
            if prioritize:
                beta_current = (beta * (total_timesteps - t) + t) / total_timesteps
                obses_current, actions_current, rewards_current, next_obses_current, dones_current, weights_current, idxes_current = replay_buffer.sample(32, beta_current)
            else:
                obses_current, actions_current, rewards_current, next_obses_current, dones_current = replay_buffer.sample(32)
                weights_current = np.ones_like(rewards_current)
        
            feed_dict = {
                obses: obses_current, 
                actions: actions_current,
                rewards: rewards_current,
                next_obses: next_obses_current,
                dones: dones_current,
                weights: weights_current}
            
            if prioritize:
                new_weights = np.abs(sess.run(TD, feed_dict = feed_dict)) + 1e-6
                replay_buffer.update_priorities(idxes_current, new_weights)
        
            TD_errors.append(sess.run(TD_error, feed_dict = feed_dict).astype(np.float64))
            state_difference_errors.append(sess.run(state_difference_error, feed_dict = feed_dict).astype(np.float64))
            reward_difference_errors.append(sess.run(reward_difference_error, feed_dict = feed_dict).astype(np.float64))
            done_difference_errors.append(sess.run(done_difference_error, feed_dict = feed_dict).astype(np.float64))
            grad_sums_of_squares.append(sess.run(grad_sum_of_squares, feed_dict = feed_dict).astype(np.float64))
        
            sess.run(update, feed_dict = feed_dict)
        
        # update eps and beta
        
        sess.run(update_eps)
        
    # training finished, save progress
    
    print('saving progress and params...')
    
    if not os.path.exists(folder_path + 'params/'):
        os.makedirs(folder_path + 'params/')
        
    with open(folder_path + 'progress.json', 'w') as f:
        data = {'episode_length': episode_length,
                   'TD_errors': TD_errors,
                   'state_difference_errors': state_difference_errors,
                   'reward_difference_errors': reward_difference_errors,
                   'done_difference_errors': done_difference_errors,
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
    
