# BOT88
Is a Self Driving Racing Car Agent Project. 

### bot88_DDPG_regular_AC folder
Initial development for my Udacity Capstone project within Machine Learning Nanodegree 2019 Certification.

### bot88_DDPG_asym_AC folder
Development more adapted to the problem to solve: how to build a simulated racing track driver that would learn to drive based on vision with the intuition of a racing line?
Publishing: LINK https://www.vrona.io/... COMING SOON

  > #### Abstract
  > Contextually, autonomous vehicle is a hot topic for years. The future of racing is a related topic on which mainly car    manufacturers are trying to anticipate guided by software, energetic and safety developments.
Even though full autonomous technology is not yet widely accepted by day to day drivers, motor sports field acts as a full-scale lab and makes the advent of self-car driving even closer.
Indeed, the launch in 2019 of Roborace Season Alpha sets autonomous racing standard and is a remarkable achievement.


### Project Description
Building an agent based on deep reinforcement leaning (DDPG - Deep Deterministic Policy Gradient algorithm) which learns to drive a racing car on sim-racing title: Project cars 2 (racetrack: Barcelona Catalunya GP).

### Project Requirements
Python > 3.6 <br>
Keras <br>
NumPy <br>
Project Cars 2 license<br>
Project Cars 2 sharedmemory.h header(cf.API)<br>
Min. 6 Core CPU, 1 Tesla M60 GPU<br>


### Project Flow
  - #### Data:
  ##### Vision input: track_compviz.py<br>
  script that captures on screen, sim-racing window and preprocessed to detected and highlighted lines. (regular folder)
                
  ##### Low level motion: <br>
  python client to retrieve data from Project Cars 2 API via shared memory.
                    
  ##### Racing line:<br>
  retrieve ideal racing line data from manual driving and preprocessed data for real-time matching with racing car motion.<br>

  - #### Development:
  ##### Controller: pilote.py and directkeys.py<br>
  script that act (accelerate, brakes, turn left or right) via direct input on up, left, right, down PC keyboards.
  
  ##### Racing line delta:racing_line_delta.py<br>
  real time delta between position of racing car from racing line.
                    
  ##### Agent: task.py, vision_physics.py, model.py, agent.py
  task, model, algorithm
