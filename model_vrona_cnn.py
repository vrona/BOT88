from keras import layers, models, optimizers, regularizers
from keras.utils import plot_model
from keras import backend as K
# from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


class Actor:
    # Actor (Policy) Model

    def __init__(self, state_size, action_size, action_low, action_high):
        # Initialize parameters and build mode

        self.state_size = state_size   # state_size (int) or (tuple 3D): Dimension of each state
        self.action_size = action_size # action_size (int): Dimension of each action
        self.action_low = action_low   # action_low (array): Min value of each action dimension
        self.action_high = action_high # action_high (array): Max value of each action dimension
        self.action_range = self.action_high - self.action_low

        # Initialize any other variables here

        self.build_model()

    def build_model(self):

        # Build an actor (policy) network that maps states -> actions.
        # Define input layer (states)
        states = layers.Input(shape=self.state_size, name='states') # initial (635, 800, 3) resize(89, 120,3)

        # Normalisation
        norma = layers.Lambda(lambda img: img / 255.0)(states) 
        # Add convolution layers
        conv1 = layers.Conv2D(filters=32, kernel_size=8, strides=4, activation='relu')(norma)
        pool1 = layers.MaxPooling2D(pool_size=2)(conv1)
        b_norm1 = layers.BatchNormalization()(pool1)
        
        conv2 = layers.Conv2D(filters=64, kernel_size=4, strides=2, padding='same', activation='relu')(b_norm1)
        pool2 = layers.MaxPooling2D(pool_size=2)(conv2)
        b_norm2 = layers.BatchNormalization()(pool2)
        # drop = layers.Dropout(0.3)

        conv3 = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu')(b_norm2)
        pool3 = layers.MaxPooling2D(pool_size=2)(conv3)
        b_norm3 = layers.BatchNormalization()(pool3)

        flat = layers.Flatten()(b_norm3) # (conv3)

        hidden1 = layers.Dense(256, activation='relu')(flat) # could be 'relu' activation

        # Add final output layer with softmax activation
        raw_actions = layers.Dense(units=self.action_size, activation='softmax', name='raw_actions')(hidden1)
        
        # Scale [0, 1] output for each action dimension to proper range
        actions = layers.Lambda(lambda x: (x * self.action_range) + self.action_low, name='actions')(raw_actions)
       
        # Create Keras model
        self.model = models.Model(inputs=states, outputs=actions)

        # Print summary of cnn model
        # print("Summary of Actor model:\n", self.model.summary())

        # plot graph
        plot_model(self.model, to_file='actor_cnn_network.png')

        # Define loss function using action value (Q value) gradients
        action_gradients = layers.Input(shape=(self.action_size,))
        loss = K.mean(-action_gradients * actions)
        
        # Incorporate any additional losses here (e.g. from regularizers)

        # Define optimizer and training function
        optimizer = optimizers.Adam(0.001)
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = K.function(inputs=[self.model.input, action_gradients, K.learning_phase()],
                                   outputs=[],
                                   updates=updates_op)


class Critic:
    #Critic (Value) Model.

    def __init__(self, state_size, action_size):
        # Initialize parameters and build model.
        self.state_size = state_size   # state_size (int) or (tuple 3D): Dimension of each state
        self.action_size = action_size # action_size (int): Dimension of each action
        
        # Initialize any other variables here

        self.build_model()

    def build_model(self):
        # Build a critic (Q value) network that maps (state, action) pairs -> Q-values.
        
        # Define input layers
        # states = layers.Input(shape=(self.state_size,), name='states')
        states = layers.Input(shape=self.state_size, name='states') # initial (635, 800, 3) resize(89, 120,3)
        actions = layers.Input(shape=(self.action_size,), name='actions')

        ### Add hidden layer(s) for state pathway ###
        # Normalisation
        norma = layers.Lambda(lambda img: img / 255.0)(states) 
        conv1 = layers.Conv2D(filters=32, kernel_size=8, strides=4, activation='relu')(norma) # hidd1 state
        pool1 = layers.MaxPooling2D(pool_size=2)(conv1)
        b_norm1 = layers.BatchNormalization()(pool1)
        
        conv2 = layers.Conv2D(filters=64, kernel_size=4, strides=2, padding='same', activation='relu')(b_norm1) # hidd2 state
        pool2 = layers.MaxPooling2D(pool_size=2)(conv2)
        b_norm2 = layers.BatchNormalization()(pool2)
        # drop = layers.Dropout(0.3)

        conv3 = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu')(b_norm2) # hidd3 state
        pool3 = layers.MaxPooling2D(pool_size=2)(conv3)
        b_norm3 = layers.BatchNormalization()(pool3)

        flat = layers.Flatten()(b_norm3) # (conv3) 

        net_states = layers.Dense(256, activation='relu')(flat) # could be 'relu' activation # hidd4 state
                
        ### Add hidden layer(s) for action pathway ###
        net_actions = layers.Dense(units=50, activation='relu')(actions) # hidd1 action
        net_actions = layers.BatchNormalization()(net_actions)
        # drop = layers.Dropout(0.3)
        
        net_actions = layers.Dense(units=128, activation='relu')(net_actions)  # hidd2 action
        net_actions = layers.BatchNormalization()(net_actions)

        net_actions = layers.Dense(units=128, activation='relu')(net_actions)  # hidd3 action
        net_actions = layers.BatchNormalization()(net_actions)

        net_actions = layers.Dense(units=256, activation='relu')(net_actions)  # hidd4 action
        net_actions = layers.BatchNormalization()(net_actions)

        # Combine state and action pathways
        net = layers.Add()([net_states, net_actions])
        net = layers.Activation('relu')(net)

        # Add more layers to the combined network if needed
        net = layers.Dense(units=32, activation='relu')(net)
        
        # Add final output layer to produce action values (Q values)
        Q_values = layers.Dense(units=1, name='q_values')(net)
        net = layers.BatchNormalization()(net)
        
        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        # Print summary of cnn model
        # print("Summary of Critic model:\n", self.model.summary())
        
        # plot graph
        # plot_model(self.model, to_file='critic_neural_network.png')

        # Define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam(0.0001) # 0.013
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = K.gradients(Q_values, actions)

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(inputs=[*self.model.input, K.learning_phase()],
                                               outputs=action_gradients)