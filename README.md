# Sango - Intuitive development of complex procedures for AI and control

Behavior trees are a control algorithm commonly used in game AI and robotics. With *Sango*, separate high level logic from low level logic by using defining the tree with a visual hierarchy.

## Diving In

Behavior Trees provide a modular approach to programming. The design is procedural which makes them ideal for modeling complex logic. They are also hierarchical in nature so can factorize a program into high level and low level logic. These traits have lead them to find use in control problems such as Game AI and Robotics.

Behavior Trees typically consist of four types of nodes (tasks)
* Composite: Executes multiple sub tasks 
* Action: Changes the state of the system
* Conditional: Returns Success or Failure based on the current state of the system or environment
* Decorator: Alters the status returned by another task

In addition to those standard tasks
* State Machine: Make a state machine as a task. Used to create complex state machines. A tree can also be a state in a state machine
* Tree - Make the behavior tree more modular and more hierarchical.

## Diving Deeper

Sango has two main modules std (standard) and ext (extension). The standard module is for defining behavior trees in a 'typical' fashion.

```
sequence = std.Sequence([action1, action2, action3])
```

where each action is executed sequentially. This makes it easier to create behavior trees in routines or class methods.

The ext module is used to intuitively define trees in a visual hierarchy using Python's class declaration syntax.

```
class sequence(ext.Sequence):
  action1 = SomeAction
  action2 = OtherAction
  action3 = FinalAction
```

This makes it easy to visualize the procedure.

The ext module has the following benefits
* Visual hierarchy of the tasks in the tree: To make it easy to understand the logic
* Cohesion between the tree structure and its tasks: To make it easy to understand and edit the tree
* Easy way to pass data between nodes: To prevent coupling of tasks that should not be coupled
* Integration of state machines: To make it possible to implement complex state machines such as hierarchical state machines, factorized state machines etc using the behavior tree architecture

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

Here, **entry** indicates the entry task of the tree. Here, it is a sequence task so successive subtasks are executed when success is returned.. **set_data** is a Fallback task (like a logical 'or'). Successive tasks are executed only on failure.
