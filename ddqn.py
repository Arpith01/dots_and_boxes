import keras
import numpy as np
import tensorflow as tf

class QNetwork:
    def __init__(self, name, input_size, hidden_layers_sizes, output_size, gamma, learning_rate=0.0001):
        inputs = tf.keras.Input((input_size,))
        layer = inputs
        for units in hidden_layers_sizes:
            layer = tf.keras.layers.Dense(units, 'relu')(layer)
        output = tf.keras.layers.Dense(output_size)(layer)
        self.learning_rate = learning_rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        loss = tf.keras.losses.MeanSquaredError()
        self.model = tf.keras.Model(inputs = inputs, outputs = output, name = name)
        self.model.compile(optimizer=optimizer, loss=loss, metrics= ['accuracy'])
        print(self.model.summary())

class DoubleDeepQNetwork:
    def __init__(self, name, input_size, hidden_layers_sizes, output_size, gamma, memory, batch_size=150, copy_weights_switch=1000, learning_rate=0.0001):
        self.name = name
        self.input_size = input_size
        self.hidden_layers_size = hidden_layers_sizes
        self.output_size = output_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.copy_weights_switch = copy_weights_switch
        self.learning_rate = learning_rate
        self.replay_memory = memory
        self.prediction_network = QNetwork(name+"_prediction_model", input_size, hidden_layers_sizes, output_size, gamma, learning_rate).model
        self.target_network = QNetwork(name+"_target_model", input_size, hidden_layers_sizes, output_size, gamma, learning_rate).model
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learn_counter = 0

    def learn(self, batch_size = 150):
        if(self.replay_memory.counter%batch_size!=0 or self.replay_memory.counter == 0):
            # print("Skipping learning. Batch not full. Memory length: ", self.replay_memory.counter)
            pass
        else:
            # print("Learning...")
            self.batch_size = batch_size if self.batch_size is None else batch_size
            training_batch = self.replay_memory.sample_memory(self.batch_size)
            states = self.batch_to_array(training_batch, "state")
            actions = self.batch_to_array(training_batch, "action")
            next_states = self.batch_to_array(training_batch, "next_state")
            rewards = self.batch_to_array(training_batch, "reward")
            done = self.batch_to_array(training_batch, "done")

            states = states.reshape((self.batch_size,self.input_size))
            next_states = next_states.reshape((self.batch_size, self.input_size))
            
            # Option #1 Using target network to predict targets
            #target_values = self.target_network.predict(states)
            
            # Option #2 Using prediction network to predict targets
            target_values = self.prediction_network.predict(states)
            Q_future = np.max(self.target_network.predict(next_states), axis=1)
            Q_future[done] = 0
            target_values[np.arange(batch_size),actions] = rewards + self.gamma*Q_future
            # print("Learning...")
            self.learn_counter +=1
            history = self.prediction_network.fit(states, target_values, verbose = 0)

            if(self.learn_counter%20 == 0):
                print("\n", self.prediction_network.name,": ", self.learn_counter, " trainings done.\n", history.history)

            if (self.replay_memory.counter%(self.copy_weights_switch * self.batch_size)) == 0:
                self.copy_weights_to_target()
                print("Weights copied!")
            

    def act(self, state, epsilon):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)

        if np.random.random() < (epsilon if epsilon else self.epsilon):
            # print("Selecting random edge")
            return np.random.randint(0, self.output_size)
        # print("Using prediction network")
        prediction = self.prediction_network.predict(state)
        return np.argmax(prediction[0])

    def batch_to_array(self, batch, key):
        return np.array(list(map(lambda x: x[key].T, batch)))

    def copy_weights_to_target(self):
        model_weights = self.prediction_network.get_weights()
        self.target_network.set_weights(model_weights)
            
    def add_to_memory(self, memory_item):
        self.replay_memory.add_to_memory(memory_item)
    
    def save_models(self, filepath):
        self.prediction_network.save(filepath+"/p")
        self.target_network.save(filepath+"/t")        
    
    def load_models(self, filepath):
        self.prediction_network =  keras.models.load_model(filepath+"/p")
        self.target_network = keras.models.load_model(filepath+"/t")

if __name__ == "__main__":
    x = DoubleDeepQNetwork("Test",4, [10], 4, 0.4, None)
    x.save_models("saved_models")
    x.load_models("saved_models")