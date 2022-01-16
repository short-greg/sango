# Sango - Intuitive Behavior Tree Design in Python

*Sango* makes behavior tree creation easy by making definition hierarchical.

## Diving In

Behavior trees provide a modular approach to programming. The design is procedural which makes them ideal for modeling complex logic. They are also hierarchical in nature so can factorize a program into high level and low level logic. These traits have lead them to find use in control problems such as Game AI and Robotics.

Personally, I am working on this project to use them in machine learning.

## Diving Deeper

For Sango, I aimed to have the following to make it easy to define trees.

* Visual hierarchy of the tasks in the tree: To make it easy to understand the logic
* Cohesion between the tree structure and its tasks: To make it easy to understand and edit the tree
* Easy way to pass data between nodes: To prevent coupling of tasks that should not be coupled

The behavior scripts are similar to those in Panda BT for Unity, but they are defined directly in Python. This increases cohesion between the tree and actions when it is best not  to decouple them. Decoupling is also easy when that is desirable.

Here is an example of a tree for training a neural network:

```
class train(Tree):

  @task
  @upto(10)  # decorator to repeat up to 10 times unless a failure status is returned
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

Here, *entry* indicates the entry task of the tree. Here, it is a sequence task so successive subtasks are executed when success is returned.. *set_data* is a Fallback task (like a logical 'or'). Successive tasks are executed only on failure.
