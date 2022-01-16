# Sango - A Behavior Tree library in Python

*Sango* makes behavior tree creation easy by making definition hierarchical.

## Diving In

Behavior trees provide a modular approach to programming. The design is procedural which makes them ideal for modeling complex logic. They are also hierarchical in nature so can factorize a program into high level and low level logic. These traits have lead them to find use in control problems such as Game AI and Robotics.

## Diving Deeper

Sango represents the hierarchy of execution visually in code. This makes Behavior Tree creation intuitive. Here is an example of a tree for training a neural network:

```
class train(Tree):

  @task
  class entry(Sequence):
    
    @task
    class set_data(Fallback):
      # get the next training data
      next_ = action('get_next_data')
      # if that fails create a new data iterator
      load = action('load_data')
     
    # next update based on the current learning data
    learn = action('learn')
  
  def __init__(self, dataset):
    # store the current data iterator
    self._data_loader_iter = None
    # store the current training data
    self._training_data = None
    self._dataset = dataset
 
  def learn(self) -> Status:
    pass
    
  def load_data(self) -> Status:
    # create a new data loader based on the dataloader
    pass
    
  def get_next_data(self) -> Status:
    # get the next item in dataloader
    pass
    
```

Here, *entry* indicates the entry task of the tree.  
