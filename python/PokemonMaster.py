"""
Author: Patrick Abiney
Last Revised: 02/07/2017
GitHub: 

Requirements:
    - 
References:
    - 
Links:
    - 
"""

##########################################################################
# Imports
##########################################################################
### tensor flow ###
import tensorflow as tf
### tflearn helper library ###
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
### Launch Enve ###
import subprocess
### Reset Env ###
import os
### Reset Env ###
import signal
##########################################################################

##########################################################################
# Compatability
##########################################################################
try:
    input = raw_input
except NameError:
    pass
try:
    writer_summary = tf.summary.FileWriter
    merge_all_summaries = tf.summary.merge_all
    histogram_summary = tf.summary.histogram
    scalar_summary = tf.summary.scalar
except Exception:
    writer_summary = tf.train.SummaryWriter
    merge_all_summaries = tf.merge_all_summaries
    histogram_summary = tf.histogram_summary
    scalar_summary = tf.scalar_summary

##########################################################################
# Globals
##########################################################################
# Max training steps
TMAX = 80000000
# Current training step
T = 0
# Async gradient update frequency of each learning thread
I_AsyncUpdate = 5
# Timestep to reset the target network
I_target = 40000
# Learning rate
learning_rate = 0.001
# Reward discount rate
gamma = 0.99
# Actions
actions = {"A", "B", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT"}
# Number of timesteps to anneal epsilon
anneal_epsilon_timesteps = 400000
# Durration to press a key for
durration = 0.50
# Screen Dimensions
screen_start_x = 955
screen_start_y = 50
frame_width = 480
frame_height = 435
#get keyboard instance
keyboard = PyKeyboard()
# Path to VBA-M
game_boy = "~/Desktop/visualboyadvance-m-master/build/visualboyadvance-m"
# Path to ROM
ROM = "~/Desktop/ROMs/Pokemon_Blue.gb"
# PID (begins as -1)
pid = -1
##########################################################################

##########################################################################
#Function to close if open, then launch VBA-M and load ROM
##########################################################################
def reset_env():
    global game_boy, ROM, pid
    close_env()
    pid = subprocess.call(game_boy + " " + ROM + " &", shell=True)
##########################################################################

##########################################################################
#Function to close VBA-M if open
##########################################################################
def close_env():
    global pid
    if pid != -1:
        os.kill(pid, signal.SIGUSR1)
##########################################################################

##########################################################################
#Function to send commands
##########################################################################
def sendCommands(action):
    global durration, keyboard
    key = ''
    #convert action to key
    if action == "A":
        key = 'w'
    elif action == "B":
        key = 'q'
    elif action == "SELECT":
        key = keyboard.backspace_key
    elif action == "START":
        key = keyboard.enter_key
    elif action == "UP":
        key = keyboard.up_key
    elif action == "DOWN":
        key = keyboard.down_key
    elif action == "LEFT":
        key = keyboard.left_key
    elif action == "RIGHT":
        key = keyboard.right_key
    keyboard.press_key(key)
    time.sleep(durration)
    keyboard.release_key(key)
    keyboard.release_key(key) #release second time incase it stuck
##########################################################################

##########################################################################
# Helper function to get the game screen
##########################################################################
def getScreen():
    global screen_start_x, frame_height, screen_start_y, frame_width
    return pyscreenshot.grab(bbox=(screen_start_x, screen_start_y, screen_start_x+frame_width, screen_start_y+frame_height))
##########################################################################

##########################################################################
#Build a DQN
##########################################################################
def build_dqn(num_actions):
    global frame_height, frame_width
    inputs = tf.placeholder(tf.float32, [None, 0, frame_height, frame_width])

    # Inputs shape: [batch, channel, height, width] need to be changed into
    # shape [batch, height, width, channel]
    net = tf.transpose(inputs, [0, 2, 3, 1])

    ## 2d conv layers
    net = tflearn.conv_2d(net, 32, 8, strides=4, activation='relu')
    net = tflearn.conv_2d(net, 64, 4, strides=2, activation='relu')

    ## Fully connected layer
    net = tflearn.fully_connected(net, 256, activation='relu')

    ## lstm layers go here
    net = tflearn.lstm(net, 256, dropout=0.8, dynamic=True)

    ## Output: fully connected layer
    q_values = tflearn.fully_connected(net, num_actions)

    ## return commands as output
    return q_values
##########################################################################

##########################################################################
# Sample a final epsilon value to anneal towards from a distribution.
# These values are specified in section 5.1 of http://arxiv.org/pdf/1602.01783v1.pdf
##########################################################################
def sample_final_epsilon():
    final_epsilons = np.array([.1, .01, .5])
    probabilities = np.array([0.4, 0.3, 0.3])
    return np.random.choice(final_epsilons, 1, p=list(probabilities))[0]
##########################################################################

##########################################################################
# Learner Function
##########################################################################
def learner(iter_count, session, graph_ops, actions, num_actions, summary_ops, saver):
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

    # Initialize network gradients
    s_batch = []
    a_batch = []
    y_batch = []

    final_epsilon = sample_final_epsilon()
    initial_epsilon = 1.0
    epsilon = 1.0

    print("Final epsilon: " + str(final_epsilon))

    time.sleep(3*thread_id)
    t = 0
    while T < TMAX:
        # Set up per-episode counters
        ep_reward = 0
        episode_ave_max_q = 0
        ep_t = 0

        # Get the screen
        s_t = getScreen()

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

            # Send generated commands

            # Update env variables

            # Accumulate gradients
            readout_j1 = target_q_values.eval(session=session, feed_dict={st: [s_t1]})
            clipped_r_t = np.clip(r_t, -1, 1)
            y_batch.append(clipped_r_t if terminal else clipped_r_t+gamma*np.max(readout_j1))

            a_batch.append(a_t)
            s_batch.append(s_t)

            # Update the counters
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
                print("| Iteration", iter_count, "| Step", t, "| Reward: %.2i" % int(ep_reward), " Qmax: %.4f" % (episode_ave_max_q/float(ep_t)), " Epsilon: %.5f" % epsilon, " Epsilon progress: %.6f" % (t/float(anneal_epsilon_timesteps)))
                break
##########################################################################

##########################################################################
# Train a model.
##########################################################################
def train(session, saver, test_model_path, actions):
    summary_ops = build_summaries()
    summary_op = summary_ops[-1]

    # Initialize variables
    session.run(tf.initialize_all_variables())
    writer = writer_summary(summary_dir + "/qlearning", session.graph)

    # Initialize target network weights
    session.run(graph_ops["reset_target_network_params"])

    for iter_count in range(0, 20):
        # lerner function
        learner(iter_count, session, graph_ops, actions, num_actions, summary_ops, saver)

        # reset env for next iteration
        env_reset()

        # wait a bit
        time.sleep(0.01)

    # Show the agents training and write summary statistics
    last_summary_time = 0
    while True:
        now = time.time()
        if now - last_summary_time > summary_interval:
            summary_str = session.run(summary_op)
            writer.add_summary(summary_str, float(T))
            last_summary_time = now
##########################################################################

##########################################################################
# Evaluate a model.
##########################################################################
def evaluate(session, saver, test_model_path, actions):
    saver.restore(session, test_model_path)
    print("Restored model weights from ", test_model_path)

    # reset env
    env_reset()

    # Unpack graph ops
    s = graph_ops["s"]
    q_values = graph_ops["q_values"]

    for i_episode in xrange(num_eval_episodes):
        s_t = getScreen()
        ep_reward = 0
        while True:
            # get commands from session
            readout_t = q_values.eval(session=session, feed_dict={s: [s_t]})
            action_index = np.argmax(readout_t)

            # send commands to game
            sendCommands(action_index)

            # update variables
            s_t1 = getScreen()
            r_t = 
            s_t = s_t1
            ep_reward += r_t

            #condition to exit infinite loop goes here

        print(ep_reward)
##########################################################################

##########################################################################
# Build Graph
##########################################################################
def build_graph(num_actions, frame_height, frame_width):
    # Create shared deep q network
    s, q_network = build_dqn(num_actions, frame_height, frame_width)
    network_params = tf.trainable_variables()
    q_values = q_network

    # Create shared target network
    st, target_q_network = build_dqn(num_actions, frame_height, frame_width)
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
##########################################################################

##########################################################################
# Main Function
##########################################################################
def main():
    with tf.Session() as tf_session:
        #configuration variables
        testing = True
        test_model_path = ""
        num_actions = actions.size
        graph_ops = build_graph(num_actions)

        #saver for model
        tf_saver = tf.train.Saver(max_to_keep=5)

        if testing:
            evaluate(tf_session, tf_saver, test_model_path, actions)
        else:
            train(   tf_session, tf_saver, test_model_path, actions)
    # exit application
    close_env()
##########################################################################

##########################################################################
# Entry Point
##########################################################################
if __name__ == "__main__":
    tf.app.run()
##########################################################################