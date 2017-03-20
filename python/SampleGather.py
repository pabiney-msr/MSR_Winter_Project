"""
Author: Patrick Abiney
Last Revised: 03/8/2017
GitHub: 

Requirements:
    - 
References:
    - 
Links:
    - 
"""
import subprocess
import signal
import os
import time

#get screenshot
import pyscreenshot

import evdev

pid = -1

##########################################################################
#Function to close if open, then launch VBA-M and load ROM
##########################################################################
def reset_env():
    global pid
    game_boy = "~/Desktop/visualboyadvance-m-master/build/visualboyadvance-m"
    ROM = "~/Desktop/ROMs/Pokemon_Blue.gb"
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
# Helper function to get the game screen
##########################################################################
def getScreen():
    screen_start_x = 955
    screen_start_y = 50
    frame_width = 480
    frame_height = 435
    im = pyscreenshot.grab(bbox=(screen_start_x, screen_start_y, screen_start_x+frame_width, screen_start_y+frame_height)).convert("L")
    return im
##########################################################################

def main():
    global pid

    #total number of playthroughs
    G = 0

    num_samples = 0

    # rate of sampling
    R = 1
    # path to samples
    sample_path = "../Samples/"
    #save file
    save_file = "master_train.txt"
    new_dir = ""

    keyboard_location = '/dev/input/event4'  # get the correct one from HAL or so
    keyboard_device = evdev.InputDevice(keyboard_location)

    #Open save text file
    save_path = sample_path + save_file
    if not os.path.exists(os.path.dirname(save_path)):
        try:
            os.makedirs(os.path.dirname(save_path))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    open(save_path, 'a').close()
    with open(save_path, 'r+') as f:
        read_data = f.read()
        for line in read_data:
            if line == '\n':
                G += 1
        #mkdir for new images
        new_dir = sample_path + str(G) + "/"
        #Append meta data to save text file
        f.write(new_dir + '\n')
    f.close()
    #Start Game
    reset_env()
    labels_path = new_dir + "labels.txt"
    if not os.path.exists(os.path.dirname(labels_path)):
        try:
            os.makedirs(os.path.dirname(labels_path))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    
    while pid != -1:
        key_state = "nothing"
        key_stateB = "nothing"
        for event in keyboard_device.read_loop():
            if key_state == "nothing":
                if event.type == evdev.ecodes.EV_KEY and event.value == 1:
                    #Keyboard interupt goes here
                    if event.code == 57: # space
                        close_env()
                        break
                    #what was pressed?
                    elif event.code == 17: # a
                        key_state = "a"
                    elif event.code == 16: # b
                        key_stateB = "b"
                    elif event.code == 14: # select
                        key_state = "select"
                    elif event.code == 28: # start
                        key_state = "start"
                    elif event.code == 103: # up
                        key_state = "up"
                    elif event.code == 108: # down
                        key_state = "down"
                    elif event.code == 105: # left
                        key_state = "left"
                    elif event.code == 106: # right
                        key_state = "right"
                    else:
                        continue
                    #Save Screen image as num_samples.png
                    getScreen().save(new_dir + str(num_samples) + ".png")
                    #Append Key State to labels file
                    with open(labels_path, 'a') as labels:
                        labels.write(key_state + " " + key_stateB + '\n')
                    labels.close()
                    #increment num_samples
                    num_samples += 1
                    #something happened
                    nothing_happened = False

                    if key_state != "nothing":
                        break
            else:
                break
        if key_state == "nothing" and key_stateB == "nothing":
            #Save Screen image as num_samples.png
            getScreen().save(new_dir + str(num_samples) + ".png")
            #Append Key State to labels file
            labels.write(key_state + " " + key_stateB + '\n')
            #increment num_samples
            num_samples += 1
            #something happened
            nothing_happened = False
        #Sleep for R
        print("Sleep for " + str(R) + "...\n")
        time.sleep(R)

if __name__ == "__main__":
    main()