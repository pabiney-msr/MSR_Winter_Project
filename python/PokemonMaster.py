"""
Author: Patrick Abiney
Last Revised: 02/28/2017
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
from PIL import Image
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
### plotting ###
import matplotlib.pyplot as plt
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
# Durration to press a key for
durration = 0.50
thresh = 0.50
# Screen Dimensions
screen_start_x = 955
screen_start_y = 50
frame_width = 480
frame_height = 435
#get keyboard instance
keyboard = PyKeyboard()
# Actions
actions = ['w', 'q', keyboard.backspace_key, keyboard.enter_key, keyboard.up_key, keyboard.down_key, keyboard.left_key, keyboard.right_key]
num_actions = len(actions)
# Path to VBA-M
game_boy = "~/Desktop/visualboyadvance-m-master/build/visualboyadvance-m"
# Path to ROM
ROM = "~/Desktop/ROMs/Pokemon_Blue.gb"
# PID (begins as -1)
pid = -1
# Path to test model
test_model_path = "../TestModelSaves/"
# model_number
model_number = 0
# training path
training_path = "../Samples/"
# stats path
stats_path = "../Stats/"
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
def sendCommands(weighted_actions):
    global durration, keyboard, actions, num_actions, thresh
    print(weighted_actions[0])
    all_zero = True
    # subtract thresh from weight of each, then clamp to >= 0.0
    for i in range(0, num_actions):
        weighted_actions[0][i] -= thresh
        if weighted_actions[0][i] <= 0.0:
            weighted_actions[0][i] = 0.0
        else:
            all_zero = False
    # return if all_zero is still true, that means we aren't doing anything this cycle
    if all_zero:
        print("Pressed Nothing")
        time.sleep(durration)
        return
    # check which has the highest weight
    highest_index = np.argmax(weighted_actions[0])
    # convert command
    key = actions[highest_index]
    # if a direction and if "B" was also pressed
    if highest_index > 3 and weighted_actions[0][1] > 0.0:
        #print 
        print("Pressed 1 and " + str(highest_index))
        # Press "B"
        keyboard.press_key(actions[1])
        # Press direction
        keyboard.press_key(key)
        # Sleep for durration
        time.sleep(durration)
        # Release direction
        keyboard.release_key(key)
        keyboard.release_key(key)
        # Release "B"
        keyboard.release_key(actions[1])
        keyboard.release_key(actions[1])
    else:
        print("Pressed " + str(highest_index))
        # press single key
        keyboard.press_key(key)
        # sleep for durration
        time.sleep(durration)
        # release key
        keyboard.release_key(key)
        #release second time incase it stuck
        keyboard.release_key(key)
##########################################################################

##########################################################################
# Helper function to get the game screen
##########################################################################
def getScreen():
    global screen_start_x, frame_height, screen_start_y, frame_width
    im = pyscreenshot.grab(bbox=(screen_start_x, screen_start_y, screen_start_x+frame_width, screen_start_y+frame_height)).convert("L")
    output = np.ones((1, frame_height), dtype=np.uint8)
    pix = np.asarray(im)
    return output[:, pix]
##########################################################################

##########################################################################
#Function to save graphs
##########################################################################
def save_graphs(error_list):
    global stats_path
    if not os.path.exists(os.path.dirname(stats_path)):
        try:
            os.makedirs(os.path.dirname(stats_path))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    #generate global bar graph of performance
    plt.figure()
    index = np.arange(len(error_list))
    bar_width = 0.35
    #plot error
    plt.bar(index, error_list, bar_width, color='b', label='Iteration')
    #legend
    plt.legend()
    #save plot
    plt.savefig(stats_path + "iteration_compare.png")

##########################################################################

##########################################################################
#Function to save statistics
##########################################################################
def save_statistics(stats):
    global stats_path
    if not os.path.exists(os.path.dirname(stats_path)):
        try:
            os.makedirs(os.path.dirname(stats_path))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    open(stats_path + "stats.text", 'a').close()
    with open(stats_path + "stats.text", 'a') as file:
        file.write(str(stats) + '\n')
    file.close()

##########################################################################

##########################################################################
#Function to load the training and testing sets
##########################################################################
def load_sets():
    global training_path, frame_height, num_actions
    path_list = []

    if not os.path.exists(os.path.dirname(training_path)):
        try:
            os.makedirs(os.path.dirname(training_path))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    #open training path
    #print("Opening Training Path: " + str(training_path) + "master_train.txt")
    open(training_path + "master_train.txt", 'a').close()
    with open(training_path + "master_train.txt", 'r+') as f:
        read_data = f.read()
        #parse master file for each dir to look in
        while read_data != "":
            path_list.append(read_data[0:read_data.find('\n')])
            read_data = read_data[read_data.find('\n')+1:]
    f.close()
    
    #loop through each dir in order 0-n
    #print("Loop through each dir in order 0-n")
    setX = []#images formatted to be input
    setY = [] #labels formatted to be output
    size = len(path_list)
    #print(path_list)
    for i in range(0, size):
        curr_path = path_list[i]
        with open(curr_path + "labels.txt", 'r+') as f:
            read_data = f.read()
            j = 0
            output = np.ones((1, frame_height), dtype=np.uint8)
            line_list = []
            #print("Making Line List")
            while read_data != "":
                line_list.append(read_data[0:read_data.find('\n')])
                read_data = read_data[read_data.find('\n')+1:]
            #print("Done making line list")
            tmpX = []
            tmpY = []

            for line in line_list:                #load labels
                #print(line)
                space_index = line.find(' ')
                label1 = line[:space_index]
                labelB = line[space_index+1:line.find('\n')-1]
                action_array = np.zeros(num_actions)
                if labelB == 'b':
                    action_array[1] = 1.0

                if label1 == 'a':
                    action_array[0] = 1.0
                elif label1 == 'select':
                    action_array[2] = 1.0
                elif label1 == 'start':
                    action_array[3] = 1.0
                elif label1 == 'up':
                    action_array[4] = 1.0
                elif label1 == 'down':
                    action_array[5] = 1.0
                elif label1 == 'left':
                    action_array[6] = 1.0
                elif label1 == 'right':
                    action_array[7] = 1.0

                tmpY.append(action_array)
                #load images
                im = Image.open(curr_path + str(j) + ".png")
                pix = np.asarray(im)
                tmpX.append(output[:, pix])
                j += 1
            setX.append(tmpX)
            setY.append(tmpY)
    #print("done loading")
    return setX, setY
##########################################################################

##########################################################################
#Function to calculate error
##########################################################################
def calculate_error(key, pred):
    global thresh
    size = len(key)
    if size > len(pred):
        size = len(pred)
    wrong_count = 0.0
    for i in range(0, size):
        # check B value
        key_B = key[i][1]
        pred_B = pred[i][1]
        if (key_B >= thresh and pred_B < thresh) or (key_B < thresh and pred_B >= thresh):
            # the value for B was wrong!
            wrong_count += 1.0
        else:
            #check other values
            key_highest = 0
            pred_highest = 0
            for j in range(2, 8):
                if key[i][j] > key[i][key_highest] and key[i][j] > thresh:
                    key_highest = j
                if pred[i][j] > pred[i][pred_highest] and pred[i][j] > thresh:
                    pred_highest = j
            if key_highest != pred_highest:
                # the key pressed was wrong!
                wrong_count += 1.0
    return (wrong_count/(size))*100.0 #percent wrong
##########################################################################

##########################################################################
#Build network
##########################################################################
def build_network():
    global num_actions, frame_height, frame_width
    net = tflearn.input_data(shape=[None, 1, frame_height, frame_width])

    # Inputs shape: [batch, channel, height, width] need to be changed into
    # shape [batch, height, width, channel]
    net = tf.transpose(net, [0, 2, 3, 1])

    ## 2d conv layers
    net = tflearn.conv_2d(net, 32, 8, strides=4, activation='relu')
    net = tflearn.conv_2d(net, 64, 4, strides=2, activation='relu')

    ## Fully connected layer
    net = tflearn.fully_connected(net, 256, activation='relu')

    ## lstm layers
    #net = tf.reshape([1, net], shape=[1, 2, 0])
    #net = tflearn.lstm(net, 256, dropout=0.8)

    net = tflearn.fully_connected(net, num_actions)

    net = tflearn.regression(net)

    ## return empty inputs and commands as output
    return net
##########################################################################

##########################################################################
# Train a model.
##########################################################################
def train():
    global actions, test_model_path
    # Initialize variables
    network = build_network()
    #load saved training info here
    setX, setY = load_sets()
    num_set = len(setX)
    #error percent list
    error_list = []

    for i in range(0, num_set):
        #print("Training " + str(i))
        model = tflearn.DNN(network, tensorboard_verbose=0)
        #print("Fit")
        #print(setY[i])
        #print(setY[num_set-i-1])
        trainX = []
        trainY = []
        testX  = []
        testY  = []

        length = len(setY[i])
        percent = .95
        portion = length * percent
        for j in range(0, length):
            #if over percent remainin
            if (length-j-1) <= portion and len(trainX) <= portion:
                #add to train
                trainX.append(setX[i][j])
                trainY.append(setY[i][j])
            else:
                if random.randrange(0, 100) <= 50 and len(trainX) <= portion:
                    #add to train
                    trainX.append(setX[i][j])
                    trainY.append(setY[i][j])
                else:
                    #add to test
                    testX.append(setX[i][j])
                    testY.append(setY[i][j])
        model.fit(trainX, trainY, validation_set=(testX, testY))
        # save model
        

        model.save(test_model_path + str(i) + ".tflearn")
        # save statistics
        #print("Predict")
        pred = model.predict(setX[i])
        #print("Calculate Error")
        error = calculate_error(setY[i], pred)
        error_list.append(error)
        #print("Save error stats")
        save_statistics(error)
        if i == 1:
            evaluate(model)

    #save graphs
    #print("Save Graphs")
    save_graphs(error_list)
##########################################################################

##########################################################################
# Evaluate a model.
##########################################################################
def evaluate(model):
    global test_model_path, model_number, actions
    #network = build_network()
    #model = tflearn.DNN(network, tensorboard_verbose=0)
    #restore model
    #model.load(test_model_path + str(model_number) + ".tflearn")
    reset_env()
    s_t = getScreen()
    while True:
        # get commands from session
        readout_t = model.predict([s_t])
        # send commands to game
        sendCommands(readout_t)
        # update screen
        s_t = getScreen()
        # sleep
        time.sleep(2)
    close_env()
##########################################################################

##########################################################################
# Main Function
##########################################################################
def main():
    with tf.Session() as tf_session:
        #are we training?
        training = True
        #are we testing?
        testing = False
        if training:
            train()
        #if testing:
        #evaluate()
##########################################################################

##########################################################################
# Entry Point
##########################################################################
if __name__ == "__main__":
    main()
##########################################################################