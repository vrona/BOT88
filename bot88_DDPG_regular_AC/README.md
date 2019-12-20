# BOT88
Self Driving Racing Car Agent

## Udacity Capstone Project - Machine Learning Nanodegree 2019

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