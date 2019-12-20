import tensorflow as tf
# from time import time
# from tensorflow.python.keras.callbacks import TensorBoard
from keras.utils import multi_gpu_model
from keras import layers, models, optimizers, regularizers
from keras.layers import Dense, Activation, Flatten, merge
from keras.initializers import VarianceScaling
from keras.utils import plot_model
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
from keras import backend as K
K.set_session(sess)

# from keras.utils import multi_gpu_model


# from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


class Actor:
    # Actor (Policy) Model

    def __init__(self, visionstate_size, action_size, action_low, action_high): # state_size
        # Initialize parameters and build mode

        # self.state_size = state_size   # state_size (int) or (tuple 3D): Dimension of each state
        self.visionstate_size = visionstate_size # 3D vision input 
        self.action_size = action_size # action_size (int): Dimension of each action
        self.action_low = action_low   # action_low (array): Min value of each action dimension
        self.action_high = action_high # action_high (array): Max value of each action dimension
        self.action_range = self.action_high - self.action_low

        # Initialize any other variables here

        self.build_model()

    def build_model(self):

        # Build an actor (policy) network that maps states -> actions.
        # Define input layer (states)
        # states = layers.Input(shape=self.state_size, name='states') # initial (635, 800, 3) resize(89, 120,3)

        states_vi = layers.Input(shape=self.visionstate_size, name='states_vi')

        ############## VISION NETWORK ############

        # Normalisation
        norma = layers.Lambda(lambda img: img / 255.0)(states_vi)

        # Add convolution layers
        conv1 = layers.Conv2D(filters=64, kernel_size=2, strides=1, activation='relu')(norma)
        pool1 = layers.MaxPooling2D(pool_size=2)(conv1)
        b_norm1 = layers.BatchNormalization()(pool1)
        
        conv2 = layers.Conv2D(filters=64, kernel_size=2, strides=1, padding='same', activation='relu')(b_norm1)
        pool2 = layers.MaxPooling2D(pool_size=2)(conv2)
        b_norm2 = layers.BatchNormalization()(pool2)

        conv3 = layers.Conv2D(filters=64, kernel_size=2, strides=1, padding='same', activation='relu')(b_norm2) # 32 2
        pool3 = layers.MaxPooling2D(pool_size=2)(conv3)
        b_norm3 = layers.BatchNormalization()(pool3)

        # conv4 = layers.Conv2D(filters=32, kernel_size=2, strides=1, padding='same', activation='relu')(b_norm3)
        # pool4 = layers.MaxPooling2D(pool_size=2)(conv4)
        # b_norm4 = layers.BatchNormalization()(pool4)

        flat = layers.Flatten()(b_norm3)

        visio_net = layers.Dense(units=150, activation='relu')(flat) # could be 'relu' activation
        visio_net = layers.BatchNormalization()(visio_net)
        
        visio_net = layers.Dense(units=300, activation='relu')(visio_net)
        visio_net = layers.BatchNormalization()(visio_net)

        visio_net = layers.Dense(units=300, activation='relu')(visio_net)
        visio_net = layers.BatchNormalization()(visio_net)

        ############## Action section ############

        # Add final output layer with softmax activation
        # raw_actions = layers.Dense(units=self.action_size, activation='sigmoid', name='raw_actions')(visio_net) 
        throttle = layers.Dense(units=1, activation='sigmoid', kernel_initializer=VarianceScaling(scale=1e-1), name='throttle')(visio_net) # 
        left = layers.Dense(units=1, activation='tanh', kernel_initializer=VarianceScaling(scale=1e-2), name='left')(visio_net)
        right = layers.Dense(units=1, activation='tanh', kernel_initializer=VarianceScaling(scale=1e-2), name='right')(visio_net)
        brake = layers.Dense(units=1, activation='sigmoid', kernel_initializer=VarianceScaling(scale=1e-4), name='brake')(visio_net)

        # Scale output for each action dimension to proper range
        # actions = layers.Lambda(lambda x: (x * self.action_range) + self.action_low, name='actions')(raw_actions)     
        
        actions = layers.concatenate([throttle, left, right, brake], axis=-1)

        # Create Keras model
        self.model = models.Model(inputs=states_vi, outputs=actions)
        self.model.save("actor_visiogo.h5")
        # self.parallel_model = multi_gpu_model(self.model, gpus=2)

        # tensorBoard = TensorBoard(log_dir='logs/{}'.format(time()))
        # Print summary of cnn model
        # print("Summary of Actor model:\n", self.model.summary())

        # plot graph
        # plot_model(self.model, to_file='actor_RGB.png')

        # Define loss function using action value (Q value) gradients
        action_gradients = layers.Input(shape=(self.action_size,))
        loss = K.mean(-action_gradients * actions)
        
        # Incorporate any additional losses here (e.g. from regularizers)

        # Define optimizer and training function
        optimizer = optimizers.Adam(0.0001)

        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        
        self.train_fn = K.function(inputs=[self.model.input, action_gradients, K.learning_phase()],
                                   outputs=[],
                                   updates=updates_op)



class Critic:
    #Critic (Value) Model.

    def __init__(self, motionstate_size, goal_size , action_size):
        # Initialize parameters and build model.

        self.motionstate_size = motionstate_size # int motion input
        self.action_size = action_size # action_size (int): Dimension of each action
        self.goal_size = goal_size
        # Initialize any other variables here

        self.build_model()

    def build_model(self):
        # Build a critic (Q value) network that maps (state, action) pairs -> Q-values.
        
        # Define input layers
        states_mo = layers.Input(shape=(self.motionstate_size,), name='states_mo')
        goal = layers.Input(shape=(self.goal_size,), name='goal')
        actions = layers.Input(shape=(self.action_size,), name='actions')
        
        ### MOTION STATE NETWORK ###
        mo_net = layers.BatchNormalization()(states_mo)
        mo_net = layers.Dense(units=150, activation='relu')(mo_net) # hidd1 actiondien
        mo_net = layers.BatchNormalization()(mo_net)
        
        mo_net = layers.Dense(units=300, activation='relu')(mo_net)  # hidd3 action
        mo_net = layers.BatchNormalization()(mo_net)

        mo_net = layers.Dense(units=300, activation='relu')(mo_net)  # hidd4 action
        mo_net = layers.BatchNormalization()(mo_net)

        ### MOTION STATE NETWORK ###
        go_net = layers.BatchNormalization()(goal)
        go_net = layers.Dense(units=150, activation='relu')(go_net) # hidd1 actiondien
        go_net = layers.BatchNormalization()(mo_net)
        
        go_net = layers.Dense(units=300, activation='relu')(go_net)  # hidd3 action
        go_net = layers.BatchNormalization()(mo_net)

        go_net = layers.Dense(units=300, activation='relu')(go_net)  # hidd4 action
        go_net = layers.BatchNormalization()(go_net)

        ### ACTION PATHWAY ###
        action_net = layers.Dense(units=150, activation='relu')(actions) # hidd1 action
        action_net = layers.BatchNormalization()(action_net)

        action_net = layers.Dense(units=300, activation='relu')(action_net)  # hidd3 action
        action_net = layers.BatchNormalization()(action_net)

        action_net = layers.Dense(units=300, activation='relu')(action_net)  # hidd4 action
        action_net = layers.BatchNormalization()(action_net)

        ############## Combine vision - motion nets + action net ############
        net = layers.Add()([mo_net, go_net, action_net])
        # net = layers.Activation('relu')(net)
        
        # net = layers.Flatten()(net)
        net = layers.Dense(units=600, activation='relu')(net)
        net = layers.BatchNormalization()(net)

        # Add final output layer to produce action values (Q values)
        Q_values = layers.Dense(units=1, activation='relu', name='q_values')(net) # au lieu de self.action_size 'relu'
        
        # Create Keras model
        self.model = models.Model(inputs=[states_mo, goal, actions], outputs=Q_values)
        self.model.save("critic_mogo.h5")
        # Print summary of cnn model
        # print("Summary of Critic model:\n", self.model.summary())
        
        # plot graph
        # plot_model(self.model, to_file='critic_motion.png')

        # Define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam(0.001)
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy']) # , 

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = K.gradients(Q_values, actions)

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(inputs=[*self.model.input, K.learning_phase()],
                                               outputs=action_gradients)
