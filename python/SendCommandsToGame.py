#for random
import random
#for sleep
import time
#for input
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
        keyboard.tap_key('w')
    elif action == "B":
        keyboard.tap_key('q')
    elif action == "SELECT":
        keyboard.tap_key(keyboard.backspace_key)
    elif action == "START":
        keyboard.tap_key(keyboard.enter_key)
    elif action == "UP":
        keyboard.tap_key(keyboard.up_key)
    elif action == "DOWN":
        keyboard.tap_key(keyboard.down_key)
    elif action == "LEFT":
        keyboard.tap_key(keyboard.left_key)
    elif action == "RIGHT":
        keyboard.tap_key(keyboard.right_key)

def main():
    #list of total commands
    actions = ["A", "B", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT"]
    action_size = len(actions)
    #number of commands to generate
    num_command = 20
    #loop to send commands
    time.sleep(10)
    #get keyboard instance
    keyboard = PyKeyboard()
    for i in range(0, num_command):
        #generate random command
        random_index = random.randrange(action_size)
        command = actions[random_index]
        #print coutn and command
        print "|COUNT: " + str(i) + "\t|\tCOMMAND: " + command + "\t|"
        #send command function call
        sendCommand(command, keyboard)
        time.sleep(2)

if __name__ == "__main__":
    main()