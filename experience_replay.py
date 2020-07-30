from collections import deque
import random
class ExperienceReplayMemory():
    def __init__(self, maximum_length = 100000, seed = None):
        self.memory = deque(maxlen=maximum_length)
        self._counter = 0
        if(seed):
            random.seed(seed)

    @property
    def counter(self):
        return self._counter

    def _increment_counter(self):
        self._counter+=1

    def add_to_memory(self, data):
        self.memory.append(data)
        self._increment_counter()

    def sample_memory(self, sample_size):
        return random.sample(self.memory, sample_size)