from directkeys import PressKey, ReleaseKey, accel, left, right, brake, recover, clutch
import time
from threading import Thread, RLock
import numpy as np

# function of actions + one "action" to recover the racing car on track
def accelerating(push):
    if push == 0:
        ReleaseKey(accel)
    else:
        ReleaseKey(brake)
        ReleaseKey(right)
        ReleaseKey(left)
        PressKey(accel)
        time.sleep(push)

def turn_left(push):
    if push == 0:
        ReleaseKey(turn_left)
        
    else:
        ReleaseKey(brake)
        ReleaseKey(accel)
        ReleaseKey(right)
        PressKey(left)
        time.sleep(push)
    
def turn_right(push):
    if push == 0:
        ReleaseKey(turn_right)
        
    else:
        ReleaseKey(brake)
        ReleaseKey(accel)
        ReleaseKey(left)
        PressKey(right)
        time.sleep(push)
    
def braking(push):
    if push == 0:
        ReleaseKey(accel)
        
    else:
        ReleaseKey(accel)
        ReleaseKey(left)
        ReleaseKey(right)
        PressKey(brake)
        time.sleep(push)
        ReleaseKey(brake)
        time.sleep(1)
    
def recovering():
    ReleaseKey(brake)
    ReleaseKey(accel)
    ReleaseKey(left)
    ReleaseKey(right)
    PressKey(recover)
    time.sleep(1)
    ReleaseKey(recover)
    time.sleep(1)


# for nn
def driving_nn(action):
    # list of driving functions
    drive = [accelerating, turn_left, turn_right, braking]
    for num, e in zip(range(4), action):
        drive[num](abs(e))


# for cnn
def driving_cnn(action):
    # list of driving functions
    drive = [accelerating, turn_left, turn_right, braking]
    # print(type(action))
    for num, e in zip(range(4), np.nditer(action)): 
        drive[num](abs(e))


# for cnn with threads
class Driving(Thread):
    def __init__(self, throttle, left, right, brakm):
        Thread.__init__(self)
        self.throttle = throttle
        self.left = left
        self.right = right
        self.brakm = brakm

    def longitude(self):
        # throttle or braking
        if self.throttle >self.brakm:
            accelerating(abs(self.throttle))
        elif self.brakm > self.throttle:
            braking(abs(self.brakm))
        
        

    def lateral(self):
        # turn left or right
        if self.left >self.right:
            turn_left(abs(self.left))
        elif self.right > self.left:
            turn_right(abs(self.right))
        

        
    def run(self):
        with RLock():
            t1 = Thread(target=self.longitude)
            t2 = Thread(target=self.lateral)
            t1.start()
            t2.start()
            t1.join()
            t2.join()
            print(t1, t2)


def driving_cnn_threads(piloting):
    for i in range(4):
        throttle, left, right, brakm = np.nditer(piloting)
    while True:
        racing_driving = Driving(throttle, left, right, brakm)
    # while True:
        racing_driving.run()
    

