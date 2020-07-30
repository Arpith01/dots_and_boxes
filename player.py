from ddqn import DoubleDeepQNetwork
from experience_replay import ExperienceReplayMemory
import numpy as np

class Player:
    name = None
    id = None
    def __init__(self):
        pass

class Human(Player):
    def __init__(self, name):
        self.name = name
        super(Human, self).__init__()
    
    def get_edge(self, state, epsilon = None):
        edge_number = int(input("\nIt's your turn!\nEnter edge number to be played: "))
        return edge_number

class RLAgent(Player):
    def __init__(self, name, gamma, learning_rate, batch_size, copy_weights_switch, input_size,hidden_layers_sizes, output_size, replay_memory_size):
        self.name = name
        self.model = DoubleDeepQNetwork(self.name, input_size, hidden_layers_sizes, output_size, gamma, ExperienceReplayMemory(replay_memory_size),
         batch_size=batch_size, copy_weights_switch=copy_weights_switch, learning_rate=learning_rate)
        super(RLAgent, self).__init__()

    def get_edge(self, state, epsilon = None):
        return self.model.act(state, epsilon)

    def learn(self, batch_size=150):
        return self.model.learn(batch_size)

    def add_to_memory(self, data_items):
        state = self.id * data_items['state']
        next_state = self.id * data_items['next_state']
        memory_item = {}
        memory_item['state'] = state
        memory_item['next_state'] = next_state
        memory_item['action'] = np.array(data_items['action'])
        memory_item['reward'] = np.array(data_items['reward'])
        memory_item['done'] = np.array(data_items['done'])
        self.model.add_to_memory(memory_item)

    def save_models(self):
        self.model.save_models("saved_models/"+self.name)

    def load_models(self):
        self.model.load_models("saved_models/"+self.name)