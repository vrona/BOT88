import numpy as np
from heartmap import live
import statedata
from vision_physics_vrona_asymgoal import VisionPhysics
from racing_line_delta import racing_line_delta
pCars  = live()

class Task():
    # Task (environment) that defines the lapdist_target and provides feedback to the agent

    def __init__(self, vision=None, pose=None, racing_line=None, euler=None, lVelocity=None, aVelocity=None, lAccel=None,runtime=5):

        # Initialize a Task object.
        self.visiomo = VisionPhysics(vision, pose, racing_line, euler, lVelocity, aVelocity, lAccel, runtime) # clapdist, 
                
        # self.action_repeat = 1
        
        # state made of PCars2 output on screen in a window captured
        self.visionstate_size = (89,120,3)
        self.motionstate_size = 3 + 3 + 3 + 3 + 3  # wPose, Euler, lVelo, aVelo, lAccel
        self.goal_size = 3 # racingLine
        self.action_low = 0.0  #.6
        self.action_high = 1.0 #2.
        self.action_size = 4  # (accelerating, braking, turn_left, turn_right)

        # target
        self.lapdist_target = pCars.mTrackLength # lapdist_target is the track length meters and it's fixed
        
        
    def get_reward(self):
        
        ## ===== Reward #1 more the meters decrease between car and finish line more reward Bot88 get and agent penalized when 4 wheels are offtrack.
        # reward = -0.05*(self.lapdist_target - self.visiomo.clapdist) - 10*pCars.mCrashState
        
        ## ===== Reward #2 more the meters decrease between car and finish line more reward Bot88 get - stick to the racing_line agent penalized when 4 wheels are offtrack.

        # reward = -0.009*(self.lapdist_target - self.visiomo.clapdist) - .7*racing_line_delta() - 10*pCars.mCrashState

        ## ===== Reward #3 more Bot88 have got velocity with too much acceleration - stick to the racing line agent penalized when 4 wheels are offtrack.
        
        # reward = -0.5*(pCars.mLocalVelocity[2]-0.1*pCars.mLocalAcceleration[2]) - 1.8*racing_line_delta() - 10*pCars.mCrashState

        ## ===== Reward #4 more Bot88 have got velocity with too much acceleration + closing to the finish line - stick to the racing line agent penalized when 4 wheels are offtrack.
        
        # reward = np.tanh(-0.7*(pCars.mLocalVelocity[2]-0.5*pCars.mLocalAcceleration[2]) - 0.09*(self.lapdist_target - self.visiomo.clapdist) - 0.7*racing_line_delta() - 10*pCars.mCrashState)
        # reward = np.tanh(-(pCars.mLocalVelocity[2]-0.5*pCars.mLocalAcceleration[2]) - 0.5*(self.lapdist_target - self.visiomo.clapdist) - 5*racing_line_delta() - 10*pCars.mCrashState)
        
        ## ===== Reward #5 more Bot88 have got velocity with too much acceleration + closing to the finish line - stick to the racing line agent penalized when 4 wheels are offtrack.
        
        # reward = np.tanh(-(pCars.mLocalVelocity[2]-0.5*pCars.mLocalAcceleration[2]) - 5*racing_line_delta() - 10*pCars.mCrashState)
        # reward = 0 # np.tanh(-(pCars.mLocalVelocity[2]-pCars.mLocalVelocity[0]-0.5*pCars.mLocalAcceleration[2]) - 2*racing_line_delta())
        # reward = np.tanh(-(pCars.mLocalVelocity[2]- .5 * pCars.mLocalVelocity[0]) - 2 *racing_line_delta() - .2 (-(pCars.mLocalVelocity[2])*racing_line_delta()))
        # penalty = 0
        reward = np.tanh(-(pCars.mLocalVelocity[2]) - 2*racing_line_delta())
        # if racing_line_delta() <= abs(3):
        #     reward += 1
        
        # elif (racing_line_delta() > abs(3)) and (racing_line_delta() <= abs(5)):
        #     reward += 0.5
        
        # elif racing_line_delta() > abs(5):
        #     reward += -1
        # elif pCars.mCrashState > 0:
        #    penalty += pCars.mCrashState
        

        return reward


    def step(self, piloting):
        # Uses actions to obtain next state, reward, done.
        reward = 0
        num_step = 0
        visionall=[]
        motall=[]
        goalall = []
        
        done = self.visiomo.next_timestep(piloting)
        reward += self.get_reward()
        # reward = reward.item(0)
        visionall.append(self.visiomo.visnext)
        motall.append(self.visiomo.motnext)
        goalall.append(self.visiomo.goalnext)
        next_state_vi = np.concatenate(visionall)
        next_state_mo = np.concatenate(motall)
        next_state_go = np.concatenate(goalall)
        num_step += 1 # counting the step
 
        return next_state_vi, next_state_mo, next_state_go, reward, done, num_step

    def reset(self):
        #Reset the sim to start a new episode.
        self.visiomo.reset()
        state_vi = self.visiomo.vision # reset the flow of vision input
        state_mo = np.hstack((self.visiomo.pose, self.visiomo.euler, self.visiomo.lVelocity, self.visiomo.aVelocity, self.visiomo.lAccel))
        state_go = self.visiomo.racing_line
        return state_vi, state_mo, state_go
