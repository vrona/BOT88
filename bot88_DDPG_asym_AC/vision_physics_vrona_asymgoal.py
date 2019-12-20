import numpy as np
import time
import timeit
from trackvision_vrona_light import grab2np
import statedata
from heartmap import live
from directkeys import ReleaseKey
from pilote import accelerating, braking, turn_left, turn_right, recovering, accel, turn_left, turn_right
from racing_linepose import racing_line
from racing_line_delta import racing_line_delta

pCars  = live()

# current lap distance (live metric position of car on track)
def cld():
    # global clapdist
    while True:
        for player in pCars.lapinfo():
            clapdist = player['lap_distance']
        return clapdist

def worldpose():
    wopo=[]
    while True:
        for player in pCars.pose():
            wopo.append(player['position'])
        return wopo


class VisionPhysics():
    def __init__(self, vision=None, pose=None, racing_line=None, euler=None, lVelocity=None, aVelocity=None, lAccel=None,runtime=5.):

        self.vision = grab2np()                         # vision: capturing pcars2 screen window
        self.pose = pose                                # world position
        self.racing_line = racing_line                  # delta position racing car to racing_line
        self.euler = euler                              # eular eulers
        self.lVelocity = lVelocity                      # local velocities
        self.aVelocity = aVelocity                      # angular velocities
        self.lAccel = lAccel                            # local acceleration
        self.runtime = runtime
        self.dt = 1/50                                  # Timestep
        self.reset()

    # when reseting for the new step, observation = the current image from the front camera, the current crash statures
    # NB: Self.vision and Self.visnext are redundant here as at the beginning and during development and test, I have tried to merge Vision and low level motion.
    def reset(self):       
        self.start_time = timeit.default_timer() # time.time()
        self.vision = grab2np()
        self.pose = np.array(worldpose())
        self.racing_line = racing_line()
        self.euler = np.array([pCars.mOrientation[0], pCars.mOrientation[1], pCars.mOrientation[2]])
        self.lVelocity = np.array([0.0, 0.0, 0.0])
        self.aVelocity = np.array([0.0, 0.0, 0.0])
        self.lAccel = np.array([0.0, 0.0, 0.0])
        
        # self.braking = braking(0.2)   # useful hard code braking when for some run of the car, if it has too much longivelo outside the track, it stops the car
        recovering()  # reintroduce the car on the track - hard code
        self.done = False

    def next_timestep(self, piloting):
        # visionT=[]
        # poseG = []

        throttle, left, right, brakm = np.nditer(piloting)

        while self.runtime:
            
            crashstate = pCars.mCrashState
            longivelo = pCars.mLocalVelocity[2]
            def worldpose():
                wopo=[]
                while True:
                    for player in pCars.pose():
                        wopo.append(player['position'])
                    return wopo

            # self.vision = grab2np()
            self.pose = np.array(worldpose()) # pCars2 X Y Z == X Z Y
            self.racing_line = racing_line()
            self.euler = np.array([pCars.mOrientation[0], pCars.mOrientation[1], pCars.mOrientation[2]])
            self.lVelocity = np.array([pCars.mLocalVelocity[0], pCars.mLocalVelocity[1], pCars.mLocalVelocity[2]])
            self.aVelocity = np.array([pCars.mAngularVelocity[0], pCars.mAngularVelocity[1], pCars.mAngularVelocity[2]])
            self.lAccel = np.array([pCars.mLocalAcceleration[0], pCars.mLocalAcceleration[1], pCars.mLocalAcceleration[2]])

            # position = self.pose + self.lVelocity * self.dt + 0.5 * self.lAccel * self.dt**2
            # self.lVelocity += self.lAccel * self.dt
            
            # self.angles = self.euler + self.angvelo * self.dt

            # the direct input controller press the key of braking to stop the car when going off-track, too much propulsion, spinning or rolling.
            # if crashstate == 1:
            #    braking(0.8)
                # visionT.append(self.vision)
                # poseG.append(self.pose)
                # self.done = True

            # if crashstate == 2:
            #     ReleaseKey(accel)
            #    time.sleep(.2)

            if crashstate > 1:
                braking(0.8)
                self.done = True

            if racing_line_delta() > abs(20):
                braking(0.8)
                self.done = True

            else:
                # throttle
                # if longivelo > -20: 
                accelerating(abs(throttle))
                # visionT.append(self.vision)
                # poseG.append(self.pose)
                
                # turn left or right
                if left > right:
                    turn_left(abs(left))
                elif right > left:
                    turn_right(abs(right))

                # else:
                #    ReleaseKey(turn_left)
                #    ReleaseKey(turn_right)
                #    time.sleep(.2)

                # braking
                if longivelo < -30 and (0.9*brakm > throttle):
                    braking(abs(brakm))
                    self.brake_time =  timeit.default_timer() # time.time() - self.start_time
                    if longivelo > -0.09 and ((self.brake_time - self.start_time) > self.runtime):
                        self.done = True
                
                self.throttle_time =  timeit.default_timer() # time.time() - self.start_time
                if longivelo > -0.09 and ((self.throttle_time - self.start_time) > self.runtime):
                        self.done = True
           
                # if the lVelocity in 3rd position is over 1 (meaning if it go backward and reach this lVelocity) the step is done.
                if longivelo > 2:
                    accelerating(1)
                    self.done = True

            self.visnext = self.vision  # np.array(visionT) # then vision is set as observation of the result of the next_step done
            self.motnext = np.hstack((self.pose, self.euler, self.lVelocity, self.aVelocity, self.lAccel))
            self.goalnext = self.racing_line

            return self.done

        