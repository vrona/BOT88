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
        ReleaseKey(accel)
        time.sleep(0.1)

def turn_left(push):
    if push == 0:
        ReleaseKey(left)
        
    else:
        ReleaseKey(brake)
        ReleaseKey(accel)
        ReleaseKey(right)
        PressKey(left)
        time.sleep(push)
        ReleaseKey(left)
        time.sleep(0.1)
    
def turn_right(push):
    if push == 0:
        ReleaseKey(right)
        
    else:
        ReleaseKey(brake)
        ReleaseKey(accel)
        ReleaseKey(left)
        PressKey(right)
        time.sleep(push)
        ReleaseKey(right)
        time.sleep(0.1)
    
def braking(push):
    if push == 0:
        ReleaseKey(brake)
        
    else:
        ReleaseKey(accel)
        ReleaseKey(left)
        ReleaseKey(right)
        PressKey(brake)
        time.sleep(push)
        ReleaseKey(brake)
        time.sleep(0.7)
    
def recovering():
    ReleaseKey(brake)
    ReleaseKey(accel)
    ReleaseKey(left)
    ReleaseKey(right)
    PressKey(recover)
    time.sleep(1)
    ReleaseKey(recover)
    time.sleep(1)

    

