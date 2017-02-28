#get screenshot
import pyscreenshot
#for sleep
import time
import numpy as np

def main():
    #make timer
    timer_original = 3
    timer = timer_original
    #loop until timer
    while timer >= 0:
        screen_start_x = 955
        screen_start_y = 50
        frame_width = 480
        frame_height = 435
        im = pyscreenshot.grab(bbox=(screen_start_x, screen_start_y, screen_start_x+frame_width, screen_start_y+frame_height)).convert("L")
        #im.show()
        print("Image " + str(timer))
        print(np.asarray(im)[0])#pixel((0,0)))
        print("\n\n")
        time.sleep(2)

        #decrement
        timer -= 1

if __name__ == "__main__":
    main()