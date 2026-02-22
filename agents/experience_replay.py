from collections import deque
import random

# define memory for experience replay
class ReplayMemory():
  def __init__(self, maxlen, seed=None):
    self.memory = deque([], maxlen=maxlen)
    
    # optional seed for reproducibility
    if seed is not None:
      random.seed(seed)
  
  # transition is a tuple of (state, action, reward, next_state, terminated)
  def append(self, transition):
    self.memory.append(transition)
    
  # randomly pulls a batch of experiences for training
  def sample(self, sample_size):
    return random.sample(self.memory, sample_size)
  
  def __len__(self):
    return len(self.memory)