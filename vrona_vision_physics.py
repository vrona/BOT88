import numpy as np
import time
from vrona_track_compviz import grab2np
import statedata
from heartmap import live
from pilote import accelerating, braking, turn_left, turn_right, recovering, driving_nn, driving_cnn, driving_cnn_threads

livegame  = live()

# current lap distance (live metric position of car on track)
def currentlapdistance():
    # global currentlapdistance
    while True:
        for player in livegame.lapinfo():
            currentlapdistance = player['lap_distance']
        return currentlapdistance


class VisionPhysics():
    def __init__(self, vision=None, currentlapdistance=None, runtime=5.):
  
        # capturing pcars2 screen window
        self.vision = grab2np()
        
        self.currentlapdistance = currentlapdistance # current lap distance

        self.crashstate = livegame.mCrashState # current crash status
        
        self.runtime = runtime
        self.dt = 1/50  # Timestep

        self.reset()

    # when reseting for the new step, observation = the current image from the front camera, the current crash statures
    # NB: Self.vision and Self.allvisphy are redundant here as at the beginning and during development and test, I have tried to merge Vision and low level motion.
    def reset(self):
        self.time = 0
        self.vision = np.zeros((89, 120,3)) if self.vision is None else np.copy(self.vision)
        self.currentlapdistance = np.array(currentlapdistance())
        # self.crashstate = np.array(livegame.mCrashState)
        self.allvisphy = np.zeros((89, 120,3)) if self.vision is None else np.copy(self.vision)
        # self.braking = braking(0.2)   # useful hard code braking when for some run of the car, if it has too much speed outside the track, it stops the car
        self.recovering = recovering()  # reintroduce the car on the track - hard code
        self.done = False


    def next_timestep(self, piloting):      

        # the direct input controller press the key of braking to stop the car when going off-track, too much propulsion, spinning or rolling.
        if livegame.mCrashState > 0:
            self.braking = braking(0.2)
            self.done = True

        # if the velocity in 3rd position is over 1 (meaning if it go backward and reach this velocity) the step is done.
        if livegame.mLocalVelocity[2] > 1:
            self.done = True

        # overwise it proceed the list of 4 vector actions [acceleration, left, right, braque]
        else:
            driving_cnn(piloting)

        self.allvisphy = self.vision # then vision is set as observation of the result of the next_step done
        self.time += self.dt
        if self.time > self.runtime:
            self.done = True
        return self.done
