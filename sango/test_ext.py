from functools import wraps
from .vars import ref_
import pytest

from .vars import Storage
from .ext import (
    vals, Action, Conditional, Sequence, task, 
    var_, ClassArgFilter, TypeFilter, Fallback,
    Tree, task_, Parallel, loads, TaskLoader, loads_,
    actionf, condf, condvar, Emission,
    Running, Failure, Success, Discrete, Ready, FSM, StateID, StateVar, state_,
    fsmstate, to_state, to_status
)

from .std import (
    Status, neg, succeed, fail, until, 
    TickDecorator2nd, fail_on_first, succeed_on_first
)



class TestVals:

    def test_gets_value_with_no_annotation(self):

        class T:
            x = 2
        
        _iter = vals(T)
        var, annotation, val = next(_iter)
        assert var == 'x'
        assert annotation is None
        assert val == 2
            
    def test_gets_value_with_annotation(self):

        class T:
            x = 2
            x2: int = 3
            # x: float
        
        _iter = vals(T)
        var, annotation, val = next(_iter)
        var, annotation, val = next(_iter)
        assert var == 'x2'
        assert annotation == int
        assert val == 3

    def test_value_with_only_annotation_doesnt_exist(self):

        class T:
            x: float
        
        _iter = vals(T)
        with pytest.raises(StopIteration):
            next(_iter)


class DummyAction(Action):

    def act(self):
        return Status.RUNNING


class DummyAction2(Action):

    x = var_[int](1)

    def act(self):
        return Status.RUNNING


class DummyNegative(Conditional):

    def check(self):
        return False


class DummyPositive(Conditional):

    def check(self):
        return True

class DummyRange(Conditional):

    def __init__(self, name: str = ''):
        super().__init__(name=name)
        self._idx = 0

    def check(self):
        if self._idx < 2:
            self._idx += 1
            return False
        return True


class TestArgFilter:

    pass


class TestArgManager:

    def test_with_two_init_vars(self):
        
        class T:
            x = 2
            y = 3
        vars = ClassArgFilter([TypeFilter(int)]).filter(T)
        assert vars['x'] == 2
        assert vars['y'] == 3

    def test_with_no_init_vars(self):
        
        class T:
            x = 3
        vars = ClassArgFilter([TypeFilter(float)]).filter(T)
        assert len(vars) == 0


class TestCreateAtomicTask:

    def test_conditional_tick(self):

        conditional = DummyPositive(name='x')
        assert conditional.tick() == Status.SUCCESS
    
    def test_conditional_check(self):

        conditional = DummyPositive(name='x')
        assert conditional.check() == True
    
    def test_action_tick(self):

        actor = DummyAction(name='x')
        assert actor.act() == Status.RUNNING

    def test_action_start_status(self):

        actor = DummyAction(name='x')
        assert actor.status == Status.READY

    def test_action_tick(self):

        actor = DummyAction(name='x')
        assert actor.name == 'x'

    def test_action_tick(self):

        actor = DummyAction('x')
        assert actor.name == 'x'

    def test_action_tick(self):

        actor = DummyAction2('x')
        assert actor.name == 'x'


class TestSequenceTask:

    def test_num_elements_with_one_element(self):

        class Pos(Sequence):
            pos = task(DummyPositive)
        
        seq = Pos()
        assert seq.n == 1

    def test_tick_with_one_element(self):

        class Pos(Sequence):
            pos = task(DummyPositive)
        
        seq = Pos()
        status = seq.tick()
        assert status == Status.SUCCESS

    def test_tick_once_with_two_elements(self):

        class Pos(Sequence):
            pos = task(DummyPositive)
            neg = task(DummyNegative)
        
        seq = Pos()
        status = seq.tick()
        assert status == Status.RUNNING

    def test_tick_twice_with_two_elements(self):

        class Pos(Sequence):
            pos = task(DummyPositive)
            neg = task(DummyNegative)
        
        seq = Pos()
        status = seq.tick()
        status = seq.tick()
        assert status == Status.FAILURE
    
    def test_tick_twice_with_first_fail(self):

        class Pos(Sequence):
            neg = task(DummyNegative)
            pos = task(DummyPositive)
        
        seq = Pos()
        status = seq.tick()
        assert status == Status.FAILURE

    def test_tick_with_no_tasks(self):

        class Pos(Sequence):
            pass
        
        seq = Pos()
        status = seq.tick()
        assert status == Status.NONE

    def test_pos_with_store(self):

        class Pos(Sequence):
            neg = task(DummyPositive)
            data = var_(2)
        
        seq = Pos()
        assert seq.data.val == 2

    def test_pos_with_store_and_new_val(self):

        class Pos(Sequence):
            neg = task(DummyPositive)
            data = var_(2)
        
        seq = Pos(data=3)
        assert seq.data.val == 3


class TestFallbackTask:

    def test_num_elements_with_one_element(self):

        class Pos(Fallback):
            pos = task(DummyPositive)
        
        seq = Pos()
        assert seq.n == 1

    def test_tick_with_one_element(self):

        class Pos(Fallback):
            pos = task(DummyPositive)
        
        seq = Pos()
        status = seq.tick()
        assert status == Status.SUCCESS

    def test_tick_once_with_two_elements(self):

        class Pos(Fallback):
            pos = task(DummyPositive)
            neg = task(DummyNegative)
        
        seq = Pos()
        status = seq.tick()
        assert status == Status.SUCCESS

    def test_tick_twice_with_two_elements(self):

        class Pos(Fallback):
            neg = task(DummyNegative)
            pos = task(DummyPositive)
        
        seq = Pos()
        status = seq.tick()
        assert status == Status.RUNNING
    
    def test_tick_twice_with_first_fail(self):

        class Pos(Fallback):
            neg = task(DummyNegative)
            pos = task(DummyNegative)
        
        seq = Pos()
        status = seq.tick()
        status = seq.tick()
        assert status == Status.FAILURE

    def test_tick_with_no_tasks(self):

        class Pos(Fallback):
            pass
        
        seq = Pos()
        status = seq.tick()
        assert status == Status.NONE



class TestTree:

    def test_tree_with_one_sequence(self):

        class X(Tree):
            entry = task_(DummyPositive)
        
        seq = X()
        assert isinstance(seq, Tree)
        assert isinstance(seq.entry, DummyPositive)

    def test_tree_with_one_fallback(self):

        class X(Tree):

            class entry(Fallback):
                pos = task_(DummyPositive)
                neg = task_(DummyPositive)
        
        seq = X()
        assert isinstance(seq, Tree)
        assert isinstance(seq.entry, Fallback)

    def test_tick_with_one_element(self):

        class X(Tree):

            class entry(Fallback):
                pos = task_(DummyPositive)
        
        tree = X()
        status = tree.tick()
        assert status == Status.SUCCESS

    def test_tick_with_two_parallel_element(self):

        class X(Tree):

            class entry(Parallel):
                pos = task_(DummyPositive)
                neg = task_(DummyNegative)
        
        tree = X()
        status = tree.tick()
        assert status == Status.FAILURE



class TestDecorators:

    def test_neg_with_one_sequence(self):

        class X(Tree):
            @loads(neg)
            @task
            class entry(Fallback):
                pos = task_(DummyPositive)
                neg = task_(DummyNegative)
        
        tree = X()
        status = tree.tick()
        assert status == Status.FAILURE

    def test_neg_preceding_dirctor_with_one_sequence(self):

        class X(Tree):
            @neg
            class entry(Fallback):
                pos = task_(DummyPositive)
                neg = task_(DummyNegative)
        
        tree = X()
        status = tree.tick()
        assert status == Status.FAILURE

    def test_fail_preceding_decorator_with_one_sequence(self):

        class X(Tree):
        
            @succeed
            class entry(Sequence):
                neg = task_(DummyNegative)
                pos = task_(DummyPositive)
        
        tree = X()
        status = tree.tick()
        assert status == Status.SUCCESS
    
    def test_until_neg_with_sequence(self):

        class X(Tree):

            @until
            @neg
            class entry(Sequence):

                neg = task_(DummyNegative)
                pos = task_(DummyPositive)
        
        tree = X()
        status = tree.tick()
        assert status == Status.SUCCESS

    def test_until_with_one_sequence(self):

        class X(Tree):
            
            @until
            class entry(Sequence):
                range = task_(DummyRange)
                pos = task_(DummyPositive)
        
        tree = X()
        status = tree.tick()
        assert status == Status.RUNNING
        status = tree.tick()
        status = tree.tick()
        status = tree.tick()
        assert status == Status.SUCCESS


class TestParallelTask:

    def test_num_elements_with_one_element(self):

        class Pos(Parallel):
            pos = task_(DummyPositive)
        
        seq = Pos()
        assert seq.n == 1

    def test_tick_with_one_element(self):

        class Pos(Parallel):
            pos = task_(DummyPositive)
        
        seq = Pos()
        status = seq.tick()
        assert status == Status.SUCCESS

    def test_fail_on_first_with_two_elements(self):

        class X(Tree):
            @task
            @fail_on_first
            class entry(Parallel):
                pos = task_(DummyAction)
                neg = task_(DummyNegative)

        
        tree = X()
        status = tree.tick()
        assert status == Status.FAILURE


    def test_succeed_on_first_with_two_elements(self):

        class X(Tree):
            @task
            @succeed_on_first
            class entry(Parallel):
                pos = task_(DummyAction)
                neg = task_(DummyPositive)

        tree = X()
        status = tree.tick()
        assert status == Status.SUCCESS


class iterate_over(TickDecorator2nd):

    def __init__(self, iterations: int):
        super().__init__()
        self._iterations = iterations

    def decorate_tick(self, node):
        i = 0

        tick = node.tick
        def _(*args, **kwargs):
            nonlocal i
            status = tick(*args, **kwargs)
            i += 1
            if i >= self._iterations:
                return Status.SUCCESS
            return Status.FAILURE
        return _


class TestDecoratorLoader:

    def test_with_one_decorator_and_loader(self):

        loader = task_(DummyPositive) << loads(fail)
        assert isinstance(loader, TaskLoader)
    
    def test_with_two_decorators(self):

        loader = loads(fail) << loads(succeed)
        assert isinstance(loader.decorators[0].tick_decorator, succeed)
        assert isinstance(loader.decorators[1].tick_decorator, fail)

    def test_with_two_decorators(self):

        loader = task_(DummyPositive) << loads(fail) << loads(succeed)
        sequence = loader.decorator

        assert isinstance(sequence.decorators[0].tick_decorator, succeed)
        assert isinstance(sequence.decorators[1].tick_decorator, fail)

    def test_tick_with_two_decorators(self):

        loader = task_(DummyPositive) << loads(fail) << loads(succeed)
        task = loader.load(Storage(), "dummy")
        status = task.tick()
        assert status == Status.SUCCESS

    def test_tick_with_two_decorators_is_success(self):

        loader =  task_(DummyPositive) << loads(succeed) << loads(fail)
        task = loader.load(Storage(), "dummy")
        status = task.tick()
        assert status == Status.FAILURE

    def test_tick_with_two_decorators_and_second_order_is_success(self):

        loader = task_(DummyPositive)<< loads(fail) <<loads_(iterate_over, 1) 
        task = loader.load(Storage(), "dummy")
        status = task.tick()
        assert status == Status.SUCCESS

    def test_tick_with_two_decorators_and_second_order_is_failure(self):

        loader =task_(DummyPositive) << loads(fail) << loads_(iterate_over, 2)
        task = loader.load(Storage(), "dummy")
        status = task.tick()
        assert status == Status.FAILURE


class TestTreeReference:

    def test_trial_tree_x(self):

        class TrialTree(Tree):

            @task
            class entry(Sequence):
                x = actionf("x")

            def x(self):
                return Status.SUCCESS
            
        tree = TrialTree()
        assert tree.tick() == Status.SUCCESS

    def test_trial_tree_with_condition(self):
        
        class TrialTree(Tree):

            @task
            class entry(Sequence):
                x = condf("x")

            def x(self):
                return True
            
        tree = TrialTree()
        assert tree.tick() == Status.SUCCESS   

    def test_trial_tree_with_condvar(self):
        
        class TrialTree(Tree):

            def __init__(self, name: str=''):
                super().__init__(name)
                self.x = True

            @task
            class entry(Sequence):
                x = condvar("x")
            
        tree = TrialTree()
        assert tree.tick() == Status.SUCCESS   

    def test_trial_tree_with_sequence_depth(self):
        
        class TrialTree(Tree):

            def __init__(self, name: str=''):
                super().__init__(name)
                self.x = True
                self.y = True

            @task
            class entry(Sequence):
                x = condvar("x")

                @task
                @neg
                class t(Sequence):
                    z = actionf('z')
                    y = actionf('t')
    
            def t(self):
                return Status.FAILURE
        
            def z(self):
                print('z')
                return Status.SUCCESS
            
        tree = TrialTree()
        tree.tick()
        tree.tick()
        assert tree.tick() == Status.SUCCESS   

    def test_trial_tree_with_fallback_depth(self):
        
        class TrialTree(Tree):

            def __init__(self, name: str=''):
                super().__init__(name)
                self.x = False

            class entry(Fallback):
                x = condvar("x")

                @neg
                class t(Fallback):
                    z = actionf('t')
                    y = actionf('z')
    
            def t(self):
                return Status.SUCCESS
        
            def z(self):
                return Status.FAILURE
            
        tree = TrialTree()
        tree.tick()
        status = tree.tick()
        # negative fallback ends on first failure
        assert status == Status.FAILURE   

    def test_trial_tree_with_fallback_until_depth(self):
        
        class TrialTree(Tree):

            def __init__(self, name: str=''):
                super().__init__(name)
                self.x = True
                self._count = 0

            @task
            class entry(Sequence):
                x = condvar("x")

                @task
                @until
                class t(Sequence):
                    z = actionf('t')
                    y = actionf('z')
    
            def t(self):
                if self._count < 2:
                    self._count += 1
                    return Status.RUNNING
                return Status.SUCCESS
        
            def z(self):
                return Status.SUCCESS
            
        tree = TrialTree()
        tree.tick()
        tree.tick()
        tree.tick()
        tree.tick()
        status = tree.tick()
        # negative fallback ends on first failure
        assert status == Status.SUCCESS   

    def test_trial_tree_with_fallback_until_depth(self):
         
        class TrialTree(Tree):

            t = var_()

            def __init__(self, name: str=''):
                super().__init__(name)
                self._count = 0
                self.t.val = 2

            @task
            class entry(Sequence):
                x = actionf("x", ref_.t)
            
            def x(self, t):
                if t.val == 2:
                    return Status.SUCCESS
                return Status.FAILURE
        
        tree = TrialTree()
        assert tree.tick() == Status.SUCCESS


class TestExternalAction:

    def test_num_elements_with_one_element(self):

        class GreaterThan0(Action):

            x = var_()

            def act(self):
                if self.x.val > 0:
                    return Status.SUCCESS
                return Status.FAILURE

        class Pos(Tree):

            entry = task_(GreaterThan0, x=2)
        
        tree = Pos()
        assert tree.tick() == Status.SUCCESS


class SimpleState(Running, Discrete):

    def update(self) -> Emission[None]:
        return Emission(self)


class EmissionState(Failure, Discrete[float]):

    def update(self) -> Emission[float]:
        return Emission(self, 2)


class FloatState2(Running, Discrete[float]):

    x = var_(3.)

    def update(self) -> Emission[float]:
        return Emission(self, self.x.val)


class FloatState3(Running, Discrete[float]):
    
    x = var_(3.)

    def __init__(self, next_state: StateVar, name: str=''):
        self._next_state = next_state
        self._name = name

    def update(self) -> Emission[float]:
        self.x.val += 1
        return Emission(self._next_state, self.x.val)
    
    def reset(self):

        self.x.val *= 0.
    

class TestEmission:

    def test_next_state_is_correct(self):

        state = SimpleState()
        emission = Emission[None](state).emit()
        assert emission.next_state == state

    def test_value_is_correct(self):

        state = SimpleState()
        emission = Emission[None](state, None).emit()
        assert emission.value is None

    def test_value_when_float(self):

        state = EmissionState()
        emission = Emission[float](state, 2.).emit()
        assert emission.value == 2.


class TestStateWithStore:

    def test_float_emission_is_2(self):

        state = FloatState2(name='x')
        emission = state.update().emit()
        assert emission.value == 3.

    def test_name_is_correct(self):
        state = FloatState2(name='x')
        assert state.name == 'x'

    def test_value_is_correct_after_enter(self):
        next_state = FloatState2(name='x')
        state = FloatState3(next_state, name='x')
        emission = state.update().emit({})

        assert emission.next_state == next_state

    def test_status_is_correct(self):
        next_state = FloatState2(name='x')
        state = FloatState3(next_state, name='first')
        emission = state.update().emit()

        assert emission.next_state == next_state


class MachineTest(FSM):

    start = state_(FloatState3, next_state=StateID('state2'))
    state2 = state_(FloatState3, next_state=StateID('state3'))
    state3 = state_(EmissionState)


class TestFSM:

    def test_basic_machine_start_state_is_correct(self):
        machine = MachineTest('tester')
        assert machine.cur_state.name == "start"

    def test_basic_machine_next_status_is_correct(self):
        machine = MachineTest('tester')
        status = machine.tick()
        assert status == Status.RUNNING

    def test_basic_machine_next_state_is_correct(self):
        machine = MachineTest('tester')
        machine.tick()
        assert machine.cur_state.name == 'state2'

    def test_basic_machine_reset_is_correct(self):
        machine = MachineTest('tester')
        machine.tick()
        machine.reset()
        assert machine.status == Status.RUNNING


class FloatStateFailure(Failure, Discrete[float]):

    x = var_(3.)

    def update(self) -> Emission[float]:
        return Emission(self, self.status)


class FloatStateSuccess(Success, Discrete[float]):

    x = var_(3.)

    def update(self) -> Emission[float]:
        return Emission(self, self.status)


class TestFSMTaskInTree:

    class X(Tree):

        @task
        class entry(FSM):

            start = state_(FloatState3, next_state=StateID('state2'))
            state2 = state_(FloatState3, next_state=StateID('state3'))
            state3 = state_(FloatStateSuccess)


    class X2(Tree):

        @task
        class entry(FSM):

            start = state_(FloatState3, next_state=StateID('state2'))
            state2 = state_(FloatState3,  next_state=StateID('state3'))
            state3 = state_(FloatStateFailure)

    def test_tree_returns_success(self):
        tree = TestFSMTaskInTree.X()

        assert tree.tick() == Status.RUNNING

    def test_tree_returns_failure(self):
        tree = TestFSMTaskInTree.X2()
        tree.tick()
        tree.tick()
        assert tree.tick() == Status.FAILURE

    def test_tree_returns_success(self):
        tree = TestFSMTaskInTree.X()
        tree.tick()
        tree.tick()
        assert tree.tick() == Status.SUCCESS


class HierarchicalMachineTest(FSM):

    start = state_(FloatState3, next_state=StateID('state2'))
    
    @fsmstate(to_state(state3='state4'))
    class state2(FSM):
        start = state_(FloatState3, next_state=StateID('state3'))
        state3 = state_(EmissionState)
    state4 = state_(EmissionState)


class HierarchicaStatusMachineTest(FSM):

    start = state_(FloatState3, next_state=StateID('state2'))
    @fsmstate(to_status(failure='state4', success='state4'))
    class state2(FSM):
        start = state_(FloatState3, next_state=StateID('state3'))
        state3 = state_(EmissionState)
    state4 = state_(EmissionState)



class TestStateLink:

    def test_get_item_returns_correct_item(self):
        state = EmissionState('x')
        link = to_state(x='y')([state])
        assert link[state].ref == 'y'

    def test_to_status_maps_to_correct_state(self):
        state = EmissionState('x')
        link = to_status(failure='y')([state])
        assert link[state].ref == 'y'

    def test_to_state_raises_exception_if_state_not_final(self):
        state = FloatState2("x")
        with pytest.raises(ValueError):
            to_state(x='y')([state])

    def test_to_state_raises_exception_if_state_invalid(self):
        state = FloatState2("z")
        with pytest.raises(ValueError):
            to_state(x='y')([state])

    def test_to_status_raises_exception_if_not_defined(self):
        state = EmissionState("z")
        with pytest.raises(ValueError):
            to_status(success='y')([state])


class TestHierarchicalFSM:

    def test_basic_machine_start_state_is_correct(self):
        machine = HierarchicalMachineTest('tester')
        assert machine.cur_state.name == "start"

    def test_basic_machine_next_status_is_correct(self):
        machine = HierarchicalMachineTest('tester')
        status = machine.tick()
        assert status == Status.RUNNING

    def test_basic_machine_next_state_is_correct(self):
        machine = HierarchicalMachineTest('tester')
        machine.tick()
        assert machine.cur_state.name == 'state2'

    def test_basic_machine_next_state_after_two_ticks_is_correct(self):
        machine = HierarchicalMachineTest('tester')
        machine.tick()
        machine.tick()
        assert machine.cur_state.name == 'state4'

    def test_basic_machine_next_state_after_three_ticks_is_correct(self):
        machine = HierarchicalMachineTest('tester')
        machine.tick()
        machine.tick()
        result = machine.tick()
        assert result == Status.FAILURE


    def test_basic_machine_next_state_after_three_ticks_is_correct(self):
        machine = HierarchicaStatusMachineTest('tester')
        machine.tick()
        machine.tick()
        result = machine.tick()
        assert result == Status.FAILURE

    def test_basic_machine_next_state_after_three_ticks_is_correct(self):
        machine = HierarchicaStatusMachineTest('tester')
        machine.tick()
        assert machine.cur_state.name == 'state2'
