import numpy as np
from heartmap import live
import statedata
from vrona_vision_physics import VisionPhysics
from racing_line_delta import racing_line_delta
livegame  = live()

class Task():
    # Task (environment) that defines the lapdist_target and provides feedback to the agent

    def __init__(self, vision=None, currentlapdistance=None, runtime=5):

        # Initialize a Task object.
        self.visionphy = VisionPhysics(vision, currentlapdistance, runtime)
        
        # self.action_repeat = 1
        
        # state made of PCars2 output on screen in a window captured
        self.state_size = (89, 120, 3)  # (635, 800, 3)
        
        self.action_low = 0.0  #.6
        self.action_high = 2.5 #2.
        self.action_size = 4  # (accelerating, braking, turn_left, turn_right)

        # target
        self.lapdist_target = livegame.mTrackLength # lapdist_target is the track length meters and it's fixed
        
        # cars' data motion -- projcars_api
        self.botvelocity = livegame.mLocalVelocity[2]          # negative velocity means velocity moving forward
        self.botacceleration = livegame.mLocalAcceleration[2]  # negative acceleration means velocity moving forward
        
    def get_reward(self):
      
        ## ===== Reward #1 more the meters decrease between car and finish line more reward Bot88 get and agent penalized when 4 wheels are offtrack.
        # reward = -0.05*(self.lapdist_target - self.visionphy.currentlapdistance) - 10*livegame.mCrashState
        
        ## ===== Reward #2 more the meters decrease between car and finish line more reward Bot88 get - stick to the racing_line agent penalized when 4 wheels are offtrack.

        # reward = -0.009*(self.lapdist_target - self.visionphy.currentlapdistance) - .7*racing_line_delta() - 10*livegame.mCrashState

        ## ===== Reward #3 more Bot88 have got velocity with too much acceleration - stick to the racing line agent penalized when 4 wheels are offtrack.
        
        reward = -0.5*(livegame.mLocalVelocity[2]-0.1*livegame.mLocalAcceleration[2]) - 1.8*racing_line_delta() - 10*livegame.mCrashState

        ## ===== Reward #4 more Bot88 have got velocity with too much acceleration + closing to the finish line - stick to the racing line agent penalized when 4 wheels are offtrack.
        
        # reward = -0.7*(livegame.mLocalVelocity[2]-0.1*livegame.mLocalAcceleration[2]) + 0.009*(self.lapdist_target - self.visionphy.currentlapdistance) - 0.7*racing_line_delta() - 10*livegame.mCrashState
        
        return reward


    def step(self, piloting):
        # Uses actions to obtain next state, reward, done.
        reward = 0
        num_step = 0
       
        done = self.visionphy.next_timestep(piloting)
        reward += self.get_reward()
        
        next_state = self.visionphy.allvisphy
        num_step += 1 # counting the step
        return next_state, reward, done, num_step

    def reset(self):
        #Reset the sim to start a new episode.
        self.visionphy.reset()
        state = self.visionphy.allvisphy # reset the flow of vision input
        return state