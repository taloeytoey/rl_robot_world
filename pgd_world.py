""" Trains an agent with (stochastic) Policy Gradients. """
import tensorflow as tf
import numpy as np
import random

#from flat_game import carmunk
import requests, json

url = 'http://192.168.1.136:9000/api'

class PolicyNetwork():
    """
    Policy Function approximator. 
    """

    def __init__(self, learning_rate, scope="policy_network"):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(dtype=tf.float32, shape=[None, 3], name="state")
            self.action = tf.placeholder(dtype=tf.int32, shape=[None, ], name="action")
            self.reward = tf.placeholder(dtype=tf.float32, shape=[None, ], name="reward")

            # FC1
            fc1 = tf.layers.dense(
                inputs=self.state,
                units=16,
                activation=tf.nn.tanh,  # tanh activation
                name='FC1'
            )

            # FC2
            fc2 = tf.layers.dense(
                inputs=fc1,
                units=32,
                activation=tf.nn.tanh,  # tanh activation
                name='FC2'
            )

            # logits
            logits = tf.layers.dense(
                inputs=fc2,
                units=3,
                activation=None,
                name='FC3'
            )

            self.action_prob = tf.nn.softmax(logits)
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.action)

            self.loss = tf.reduce_mean(neg_log_prob * self.reward)
            # train op
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())

    def predict(self, state, sess):
        return sess.run(self.action_prob, {self.state: state})

    def update(self, state, reward, action, sess):
        feed_dict = {self.state: state, self.reward: reward, self.action: action}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss

# hyperparameters
learning_rate = 0.005
gamma = 0.99  # discount factor for reward
resume = True  # resume from previous checkpoint?
max_episode_number = 1000  # how many episode we want to run ?
model_path = "../rl_robot_sim/_models/reinforce/model.ckpt"  # path for saving the model
real_env_model_path = "_models/reinforce/model.ckpt"


def discount_rewards(r):
    """
    take 1D float array of rewards and compute discounted rewards (A_t)
    A_t = R_t + gamma^1 * R_t+1 + gamma^2 * R_t+2 + ... + gamma^(T-t)R_T;
    where T is the last time step of the episode

    :param r: float array of rewards (R_1, R_2, ..., R_T)
    :return: float array of discounted reward (A_1, A_2, ..., A_T)
    """

    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        # if r[t] != 0: running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r




if __name__ == '__main__':

    fhand = open('log/episode_reward_log.txt', 'w')
    fhand.write('EPISODE\tTIME_STEPS_TAKEN\tTOTAL_REWARD')
    fhand.write('\n')
    fhand.close()

    state_list, action_list, reward_list = [], [], []
    running_reward = None
    reward_sum = 0
    episode_number = 0

    policy_network = PolicyNetwork(learning_rate)

    # saver
    saver = tf.train.Saver()
    # session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    if resume:
        saver.restore(sess, model_path)

    # create a new game instance
    #env = carmunk.GameState()
    
    done = False
       
    # get initial state by doing nothing and getting the state
    #_, observation = env.frame_step(2)
    action = json.dumps({'action':2})
    r = requests.post(url,action)
  
    while episode_number < max_episode_number:

        current_state = np.array(r.json()['observation'])

        #print (current_state)

        # forward the policy network and sample an action from the returned probability
        action_prob = policy_network.predict(current_state[np.newaxis, :], sess)
        action = np.random.choice(a=3, p=action_prob.ravel())

        # record various intermediates
        action_list.append(action)
        state_list.append(current_state)  # observation


        # step the environment and get new measurements
        #reward, observation = env.frame_step(action)
        action = json.dumps({'action':action})
        r = requests.post(url,action)

        reward_sum += r.json()['reward']

        reward_list.append(r.json()['reward'])  # record reward (has to be done after we call step() to get reward for previous action)

        if r.json()['reward'] == -50:
            done = True
        else:
            done = False


        if done:  # an episode finished
            episode_number += 1

            # stack together all inputs, action and rewards for this episode
            state_batch = np.vstack(state_list)
            action_batch = np.array(action_list)
            reward_batch = np.array(reward_list)

            state_list, action_list, reward_list = [], [], []  # reset array memory


            # compute the discounted reward backwards through time
            discounted_epr = discount_rewards(reward_batch)
            # standardize the rewards to be unit normal (helps control the gradient estimator variance)
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)


            # update model variables with data obtained from this episode
            policy_network.update(state_batch, discounted_epr, action_batch, sess)

            # record running_reward to get overview of the improvement so far
            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            
           
            print('ep %d: game finished, reward: %.2f, running_reward: %.2f' % (
                episode_number, reward_sum, running_reward))

            # Save episode information
            fhand = open('log/episode_reward_log.txt', 'a')
            fhand.write(str(episode_number) + '\t' + str(reward_sum) + '\t' + str(running_reward))
            fhand.write('\n')
            fhand.close()

            # reset reward_sum
            reward_sum = 0

            # save the model every 30 episodes
            if episode_number % 5 == 0: saver.save(sess, real_env_model_path)







