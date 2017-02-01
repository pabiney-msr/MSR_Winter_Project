#get screenshot
import pyscreenshot
#for sleep
import time
def main():
    #make timer
    timer_original = 10
    timer = timer_original
    #loop until timer
    while timer >= 0:
        im = pyscreenshot.grab(bbox=(10,10,510,510)) # X1,Y1,X2,Y2
        im.show()
        time.sleep(2)

        #decrement
        timer -= 1

if __name__ == "__main__":
    main()