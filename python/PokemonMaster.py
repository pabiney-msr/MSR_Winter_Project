### tensor flow ###
import tensorflow as tf
import tflearn

### random ###
import random

### numbers ###
import numpy as np

### screenshot ###
import pyscreenshot

### sleep ###
import time

### input ###
from pykeyboard import PyKeyboard
#Hack to make input work for both Python 2 and Python 3
try:
    input = raw_input
except NameError:
    pass

#function to send commands
def sendCommand(action, keyboard):
    #convert action to key
    if action == "A":
        keyboard.press_key('w')
        time.sleep(2)
        keyboard.release_key('w')
    elif action == "B":
        keyboard.press_key('q')
        time.sleep(2)
        keyboard.release_key('q')
    elif action == "SELECT":
        keyboard.press_key(keyboard.backspace_key)
        time.sleep(2)
        keyboard.release_key(keyboard.backspace_key)
    elif action == "START":
        keyboard.press_key(keyboard.enter_key)
        time.sleep(2)
        keyboard.release_key(keyboard.enter_key)
    elif action == "UP":
        keyboard.press_key(keyboard.up_key)
        time.sleep(2)
        keyboard.release_key(keyboard.up_key)
    elif action == "DOWN":
        keyboard.press_key(keyboard.down_key)
        time.sleep(2)
        keyboard.release_key(keyboard.down_key)
    elif action == "LEFT":
        keyboard.press_key(keyboard.left_key)
        time.sleep(2)
        keyboard.release_key(keyboard.left_key)
    elif action == "RIGHT":
        keyboard.press_key(keyboard.right_key)
        time.sleep(2)
        keyboard.release_key(keyboard.right_key)

def getScreen(bounding_box):
    #im = pyscreenshot.grab(bbox=(10,10,510,510)) # X1,Y1,X2,Y2
    im = pyscreenshot.grab(bounding_box) # X1,Y1,X2,Y2
    return im

def build_dqn(num_actions, action_repeat):
    """
    Building a DQN.
    """
    inputs = tf.placeholder(tf.float32, [None, action_repeat, 84, 84])

    # Inputs shape: [batch, channel, height, width] need to be changed into
    # shape [batch, height, width, channel]
    net = tf.transpose(inputs, [0, 2, 3, 1])

    ## 2d conv layers
    net = tflearn.conv_2d(net, 32, 8, strides=4, activation='relu')
    net = tflearn.conv_2d(net, 64, 4, strides=2, activation='relu')

    ## rnn lstm layers


    ## fully connected layers
    net = tflearn.fully_connected(net, 256, activation='relu')
    q_values = tflearn.fully_connected(net, num_actions)

    ## return commands as output
    return q_values

def sample_final_epsilon():
    """
    Sample a final epsilon value to anneal towards from a distribution.
    These values are specified in section 5.1 of http://arxiv.org/pdf/1602.01783v1.pdf
    """
    final_epsilons = np.array([.1, .01, .5])
    probabilities = np.array([0.4, 0.3, 0.3])
    return np.random.choice(final_epsilons, 1, p=list(probabilities))[0]


def learner_thread(thread_id, env, session, graph_ops, num_actions, summary_ops, saver):
    global TMAX, T

    # Unpack graph ops
    s = graph_ops["s"]
    q_values = graph_ops["q_values"]
    st = graph_ops["st"]
    target_q_values = graph_ops["target_q_values"]
    reset_target_network_params = graph_ops["reset_target_network_params"]
    a = graph_ops["a"]
    y = graph_ops["y"]
    grad_update = graph_ops["grad_update"]

    summary_placeholders, assign_ops, summary_op = summary_ops

    # Wrap env with AtariEnvironment helper class
    env = 

    # Initialize network gradients
    s_batch = []
    a_batch = []
    y_batch = []

    final_epsilon = sample_final_epsilon()
    initial_epsilon = 1.0
    epsilon = 1.0

    print("Thread " + str(thread_id) + " - Final epsilon: " + str(final_epsilon))

    time.sleep(3*thread_id)
    t = 0
    while T < TMAX:
        # Get initial game observation
        s_t = env.get_initial_state()
        terminal = False

        # Set up per-episode counters
        ep_reward = 0
        episode_ave_max_q = 0
        ep_t = 0

        while True:
            # Forward the deep q network, get Q(s,a) values
            readout_t = q_values.eval(session=session, feed_dict={s: [s_t]})

            # Choose next action based on e-greedy policy
            a_t = np.zeros([num_actions])
            action_index = random.randrange(num_actions) if random.random() <= epsilon else np.argmax(readout_t)
            a_t[action_index] = 1

            # Scale down epsilon
            if epsilon > final_epsilon:
                epsilon -= (initial_epsilon - final_epsilon) / anneal_epsilon_timesteps

            # Gym excecutes action in game environment on behalf of actor-learner
            s_t1, r_t, terminal, info = env.step(action_index)

            # Accumulate gradients
            readout_j1 = target_q_values.eval(session = session,
                                              feed_dict = {st : [s_t1]})
            clipped_r_t = np.clip(r_t, -1, 1)
            if terminal:
                y_batch.append(clipped_r_t)
            else:
                y_batch.append(clipped_r_t + gamma * np.max(readout_j1))

            a_batch.append(a_t)
            s_batch.append(s_t)

            # Update the state and counters
            s_t = s_t1
            T += 1
            t += 1

            ep_t += 1
            ep_reward += r_t
            episode_ave_max_q += np.max(readout_t)

            # Optionally update target network
            if T % I_target == 0:
                session.run(reset_target_network_params)

            # Save model progress
            if t % checkpoint_interval == 0:
                saver.save(session, "qlearning.ckpt", global_step=t)

            # Print end of episode stats
            if terminal:
                stats = [ep_reward, episode_ave_max_q/float(ep_t), epsilon]
                for i in range(len(stats)):
                    session.run(assign_ops[i], {summary_placeholders[i]: float(stats[i])})
                print("| Thread %.2i" % int(thread_id), "| Step", t,
                      "| Reward: %.2i" % int(ep_reward), " Qmax: %.4f" %
                      (episode_ave_max_q/float(ep_t)),
                      " Epsilon: %.5f" % epsilon, " Epsilon progress: %.6f" %
                      (t/float(anneal_epsilon_timesteps)))
                break


def train(tf_session, tf_saver, test_model_path, actions):
    """
    Train a model.
    """
    tf_saver.restore(tf_session, test_model_path)




def evaluate(tf_session, tf_saver, test_model_path, actions):
    """
    Evaluate a model.
    """
    tf_saver.restore(tf_session, test_model_path)



def build_graph(num_actions):
    # Create shared deep q network
    s, q_network = build_dqn(num_actions=num_actions, action_repeat=action_repeat)
    network_params = tf.trainable_variables()
    q_values = q_network

    # Create shared target network
    st, target_q_network = build_dqn(num_actions=num_actions, action_repeat=action_repeat)
    target_network_params = tf.trainable_variables()[len(network_params):]
    target_q_values = target_q_network

    # Define cost and gradient update op
    a = tf.placeholder("float", [None, num_actions])
    y = tf.placeholder("float", [None])
    action_q_values = tf.reduce_sum(tf.mul(q_values, a), reduction_indices=1)
    cost = tflearn.mean_square(action_q_values, y)
    optimizer = tf.train.RMSPropOptimizer(learning_rate)
    grad_update = optimizer.minimize(cost, var_list=network_params)

    graph_ops = {"s": s,
                 "q_values": q_values,
                 "st": st,
                 "target_q_values": target_q_values,
                 "reset_target_network_params": reset_target_network_params,
                 "a": a,
                 "y": y,
                 "grad_update": grad_update}

    return graph_ops

def main(_):
    with tf.Session() as tf_session:
        #launch VBA-M in full screen always on top


        #get keyboard instance
        keyboard = PyKeyboard()

		#configuration variables
		testing = True
    	test_model_path = ""
    	actions = {"A", "B", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT"}
        num_actions = actions.size
        graph_ops = build_graph(num_actions)

        #saver for model
    	tf_saver = tf.train.Saver(max_to_keep=5)

        if testing:
            evaluate(tf_session, tf_saver, test_model_path, actions, )
        else:
            train(tf_session, tf_saver, test_model_path, actions, )

if __name__ == "__main__":
    tf.app.run()