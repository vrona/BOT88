import numpy as np
import sys
from heartmap import live
import statedata
import csv
pCars  = live()

"""
After driving manually to get the racing line X,Y,Z coordonates (aka X,Z,Y from PCars2 perspective)
The goal is to retrieve in real time the delta between the racing line and the current position of
the car.
racingline_data.txt is processed to provide the final_RL_spainGP.csv. Which contain for each meters X,Y,Z coordinates

The principle here is
As the car drive, its current lap distance is compared to the race_line file.
When these 2 data are matching, live X,Y coordinates of the car are substract
to the X,Y coordinates of the ideal race_line.
"""

# loading final_RL_spainGP.csv
rl_data2np =np.genfromtxt('V:/vrona_bot88/final_RL_spainGP.csv', delimiter='\t')
# print(rl_data2np.shape)
rl_data2npint = rl_data2np.astype(np.int64)

# racing_line_delta function which will be used in the reward function included in Task scripts
def racing_line_delta():
    # getting live current lap distance of the car
    def currentlapdistance():
        # global currentlapdistance
        while True:
            for player in pCars.lapinfo():
                currentlapdistance = int(player['lap_distance'])
            return currentlapdistance

    # getting live X,Y,Z coordinates
    while True:
        def worldpose():
            while True:
                for player in pCars.pose():
                    wopo.append(int(player['position']))
                return wopo

        wopo=[]
        worldpose()
        
        # matching the lap distance between car and race_line
        matching = np.array(np.where(rl_data2npint[:,0] == [currentlapdistance()] )) # getting index of the row in the race_line where the matching operates
        botDindex = matching[0][0] # isoled the index
        rldist = rl_data2npint[:,0][botDindex] # index of row provide the value of the lap distance

        delta_x = rl_data2np[:,1][botDindex]-wopo[0] # samed index but in X column provide value of race_line X. Absolute value substraction between X car positing and X race_line prositionning
        delta_y = abs(wopo[1]-rl_data2np[:,2][botDindex])
        delta_z = rl_data2np[:,3][botDindex]-wopo[2]
        # delta_glob = delta_x + delta_z # relevance here doesn't concerne elevation: Z (aka Y in PCars2 perspective)
        # print("{} H {}\r".format(delta_x, delta_z), end='')
        # sys.stdout.flush()
        return np.array([delta_x, delta_z, delta_y])

"""    
# while True:
#    print(racing_line_delta())
       
       if too much out of the racing_line BANG
       if abs(delta_x) <= 2 and abs(delta_z) <= 2:
           crashespose.append(self.vision)
        if too much out of the racing_line BANG
                if abs(delta_x) > 2 and abs(delta_z) > 2:
                    crashespose.append(self.vision)
                    self.braking = braking(1)
                    self.done = True
"""