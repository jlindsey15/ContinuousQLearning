import tensorflow as tf
import gym


from tensorflow.contrib.framework import *
from tensorflow.contrib.layers import *

import numpy as np


batch_norm = False
noise_magnitude = 1
updates_per_step = 5
batch_size = 100
layer_sizes = [200, 200]
monitor = False
num_eps = 100000000
lr = 0.001
reward_discount = 0.99
noise_decay = 0.999
target_update_rate = 0.001
environment = "Pendulum-v0"

env = gym.make(environment)
terminate_after_steps = env.spec.timestep_limit

sess = tf.Session()


with tf.variable_scope("qnet"):
    qnet_act = tf.placeholder(tf.float32, tuple([None]) + tuple(env.action_space.shape))
    qnet_obs = tf.placeholder(tf.float32, tuple([None]) + tuple(env.observation_space.shape))
    qnet_goal_Q = tf.placeholder(tf.float32, [None])

                    

    qnet_curr = qnet_obs
                                        
    for i in range(len(layer_sizes)):
        with tf.variable_scope('hidden_' + str(i)):
            qnet_curr = fully_connected(qnet_curr, layer_sizes[i],activation_fn=tf.nn.relu, scope='hidden_' + str(i))

                                                        
    with tf.variable_scope('val'):
        qnet_val = fully_connected(qnet_curr, 1, activation_fn=None, scope='V')

    with tf.variable_scope('pre_L'):
        qnet_pre_L = fully_connected(qnet_curr, ((env.action_space.shape[0] + 1) * env.action_space.shape[0])/2, activation_fn=None, scope='l')
    with tf.variable_scope('mu'):
        qnet_mu = fully_connected(qnet_curr, env.action_space.shape[0], activation_fn=None, scope='mu')
############
    for i in range(env.action_space.shape[0]):
        
        qnet_col = tf.slice(qnet_pre_L, [0, i * env.action_space.shape[0] - (i * i - i)/2], [-1, env.action_space.shape[0] - i])
        qnet_col = tf.pad(qnet_col, [[0, 0], [i, 0]])
        if i == 0:
            qnet_L = tf.expand_dims(tf.transpose(qnet_col), 0)
        else:
            qnet_L = tf.concat(0, [qnet_L, tf.expand_dims(tf.transpose(qnet_col), 0)])
    
    qnet_L = tf.transpose(qnet_L)
    qnet_P = tf.batch_matmul(tf.transpose(qnet_L, [0, 2, 1]), qnet_L)
    
    qnet_u_minus_mu = qnet_act - qnet_mu
    qnet_temp = tf.batch_matmul(tf.expand_dims(qnet_u_minus_mu, 1), qnet_P)
    qnet_adv = -0.5 * tf.batch_matmul(qnet_temp, tf.expand_dims(qnet_u_minus_mu, 2))
    qnet_adv = tf.reshape(qnet_adv, [-1, 1])
    qnet_Q = qnet_adv + qnet_val
                                                                                                                                            
    qnet_loss = tf.reduce_mean(tf.square(qnet_Q - qnet_goal_Q))
###############

with tf.variable_scope("tar"):
    tar_act = tf.placeholder(tf.float32, tuple([None]) + tuple(env.action_space.shape))
    tar_obs = tf.placeholder(tf.float32, tuple([None]) + tuple(env.observation_space.shape))
    tar_goal_Q = tf.placeholder(tf.float32, [None])

    tar_curr = tar_obs
    
    for i in range(len(layer_sizes)):
        with tf.variable_scope('hidden_' + str(i)):
            tar_curr = fully_connected(tar_curr, layer_sizes[i],activation_fn=tf.nn.relu, scope='hidden_' + str(i))

    with tf.variable_scope('val'):
        tar_val = fully_connected(tar_curr, 1, activation_fn=None, scope='V')

    with tf.variable_scope('pre_L'):
        tar_pre_L = fully_connected(tar_curr, ((env.action_space.shape[0] + 1) * env.action_space.shape[0])/2, activation_fn=None, scope='l')
    with tf.variable_scope('mu'):
        tar_mu = fully_connected(tar_curr, env.action_space.shape[0], activation_fn=None, scope='mu')
##########
    for i in range(env.action_space.shape[0]):

        tar_col = tf.slice(tar_pre_L, [0, i * env.action_space.shape[0] - (i * i - i)/2], [-1, env.action_space.shape[0] - i])
        tar_col = tf.pad(tar_col, [[0, 0], [i, 0]])
        if i == 0:
            tar_L = tf.expand_dims(tf.transpose(tar_col), 0)
        else:
            tar_L = tf.concat(0, [tar_L, tf.expand_dims(tf.transpose(tar_col), 0)])

    tar_L = tf.transpose(tar_L)
    tar_P = tf.batch_matmul(tf.transpose(tar_L, [0, 2, 1]), tar_L)
    
    tar_u_minus_mu = tar_act - tar_mu
    tar_temp = tf.batch_matmul(tf.expand_dims(tar_u_minus_mu, 1), tar_P)
    tar_adv = -0.5 * tf.batch_matmul(tar_temp, tf.expand_dims(tar_u_minus_mu, 2))
    tar_adv = tf.reshape(tar_adv, [-1, 1])
    tar_Q = tar_adv + tar_val
    
    tar_loss = tf.reduce_mean(tf.square(tar_Q - tar_goal_Q))
###########
tar_ops = []

for k in range(len(list(get_variables("qnet")))):
    qnet_var = get_variables("qnet")[k]
    tar_var = get_variables("tar")[k]
    tar_ops.append(tar_var.assign(target_update_rate * qnet_var + (1-target_update_rate) * tar_var))


buffer = {'obs': [], 'act': [], 'res': [], 'rew': [] }

with tf.name_scope("ignore"):
    goal_Q = tf.placeholder(tf.float32, [None])
    loss = tf.reduce_mean(tf.square(tf.squeeze(qnet_Q) - goal_Q))
    train_op = tf.train.AdamOptimizer(lr).minimize(loss)

sess.run(tf.initialize_all_variables())
for k in range(len(list(get_variables("qnet")))):
    qnet_var = get_variables("qnet")[k]
    tar_var = get_variables("tar")[k]
    sess.run(tar_var.assign(qnet_var))
if monitor:
    env.monitor.start("/monitor/" + environment + "_" + str(time.clock()))

for i_episode in range(num_eps):
    total_reward = 0
    print(environment)
    state = env.reset()
                    
    for t in range(0, terminate_after_steps):
        skip = False
        #env.render()
        
        mean_action = sess.run(qnet_mu, {qnet_obs: [state]})
        action = mean_action[0] + noise_magnitude * np.random.randn(env.action_space.shape[0]) * np.power(noise_decay, i_episode)

        last_state = state
        state, reward, done, info = env.step(action)
        total_reward = total_reward + reward
        
        if t >= terminate_after_steps:
            done = True
        
        buffer['obs'].append(last_state)
        buffer['act'].append(action)
        buffer['res'].append(state)
        buffer['rew'].append(reward)
        
        for j in range(updates_per_step):
            if len(buffer['obs']) <= batch_size:
                print(len(buffer['obs']))
                skip = True
            else:
                rand_indices = np.random.choice(len(buffer['obs']), size=batch_size)
            if skip:
                break
        
            obs = np.array(buffer['obs'])[rand_indices]
            act = np.array(buffer['act'])[rand_indices]
            res = np.array(buffer['res'])[rand_indices]
            rew = np.array(buffer['rew'])[rand_indices]
            
            
            value = np.squeeze(sess.run(tar_val, {tar_obs: res, tar_act: act}))

            q_goal =  rew + reward_discount * value

                                
            sess.run(train_op, {qnet_obs: obs, qnet_act: act, goal_Q: q_goal})
            for op in tar_ops:
                sess.run(op)

    
        if done:
            break
    print(str(i_episode) + ", " + str(total_reward))
                
if monitor:
    env.monitor.close()
