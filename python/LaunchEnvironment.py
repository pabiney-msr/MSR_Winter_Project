import subprocess
import signal
import os
import time

game_boy = "~/Desktop/visualboyadvance-m-master/build/visualboyadvance-m"
ROM = "~/Desktop/ROMs/Pokemon_Blue.gb"
pid = -1

def launch():
    global game_boy, ROM, pid
    pid = subprocess.call(game_boy + " " + ROM + " &", shell=True)

def close():
    os.kill(pid, signal.SIGUSR1)

def main():
    global game_boy, ROM 
    print("Game Boy PATH:\t" + game_boy)
    print("ROM PATH:\t" + ROM)
    print("Launching...")
    launch()
    print("Launched!")
    print("Sleep 10")
    time.sleep(10)
    print("Closing...")
    close()
    print("Closed!")

if __name__ == "__main__":
    main()