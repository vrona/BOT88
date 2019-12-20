# BOT88
Self Driving Racing Car Agent

## Udacity Capstone Project - Machine Learning Nanodegree 2019

## Abstract

Contextually, autonomous vehicle is a hot topic for years. The future of racing is a related topic on which mainly car manufacturers are trying to anticipate guided by software, energetic and safety developments.
Even though full autonomous technology is not yet widely accepted by day to day drivers, motor sports field acts as a full-scale lab and makes the advent of self-car driving even closer.
Indeed, the launch in 2019 of Roborace Season Alpha sets autonomous racing standard and is a remarkable achievement.

Technically, reinforcement learning enables engineers to develop machines that could learn on their own within a relatively simple deterministic space. Thus, for more complex state and action spaces the combination of deep learning with reinforcement learning is required as the nature of spaces changes to continuous.
On 2015/2016, a team from Google has published the Deep deterministic policy gradient (DDPG) (1) algorithm. Starting from it, DDPG algorithm and some of its variants, has helped to solve highly complex problems in continuous spaces.

VRONA® BOT88 is a self-driving car project that leans toward building an autonomous racing driver model based on plain input vision (no lane and surface detectors preprocessing). It learns to control a Porsche 911 GT3 R on (Barcelona) Catalunya GP racetrack in the notorious high quality racing simulation (sim-racing) title, Project Cars 2™.
This project aims to answer the following question: how to build a simulated racing track driver that would learn to drive based on vision with the intuition of a racing line?
The underlying idea here, is that like racing driver that uses his/her senses (then skills) to perform, the model should learn in the same context as a driver.

DDPG algorithm has been, then, used with an asymmetric actor-critic.

This publishing is focused on BOT88’s training phase.

### Project Description
Building an agent based on deep reinforcement leaning (DDPG - Deep Deterministic Policy Gradient algorithm) which learns to drive a racing car on sim-racing video game: Project cars 2 (racetrack: Barcelona Catalunya GP)

### Project Requirements
Python > 3.6 <br>
Keras <br>
NumPy <br>
Project Cars 2 license<br>
Project Cars 2 sharedmemory.h header(cf.API)<br>
Min. 6 Core CPU, 1 Tesla M60 GPU<br>


### Project Flow
  - #### Data:
  ##### Vision input: vrona_track_compviz.py<br>
  script that captures on screen, sim-racing window and preprocessed to detected and highlighted lines.
                
  ##### Low level motion: low_level_motion_data_process_retrievement.ipynb and low_level_motion_demo.ipynb<br>
  python client to retrieve data from Project Cars 2 API via shared memory.
                    
  ##### Racing line: display_of_racing_line_spainGP.ipynb<br>
  retrieve ideal racing line data from manual driving.<br>

  - #### Development:
  ##### Controller: pilote.py and directkeys.py<br>
  script that act (accelerate, brakes, turn left or right) via direct input on up, left, right, down PC keyboards.
  
  ##### Racing line delta:racing_line_delta.py<br>
  real time delta between position of racing car from racing line.
                    
  ##### Agent:vrona_task.py, vrona_vision_physics.py, model_vrona_cnn.py, agent_vrona.py
  task, model, algorithm

### Aditional documents
  - Test_266episode_bot88_7_nov.ipynb (concerns one test of 266 episodes)
  - final_RL_spainGP.csv (uses into racing_line_delta.py)
  
### Test steps
- After launching the game > change resolution to 800 x 635 in 'performances' option (needs to reboot game).<br>
- After reboot, poised the window on the upper left (stick to top and left bound of the screen).<br>
- Pick "private testing" in the menu.<br>
- Launch the test drive until the control pass to manual (yellow flag)<br>
- Run the 1st two cells of vrona_bot88.ipynb<br>
