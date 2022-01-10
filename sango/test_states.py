from sango.nodes import Status, Tree, task, var
from sango.vars import Args, Ref, Var
from .states import FSM, Discrete, Emission, State, StateMachine, StateRef, StateType, StateVar, state, state_


class SimpleState(Discrete[None]):

    def update(self) -> Emission[None]:
        return Emission(self)


class EmissionState(Discrete[float]):

    def update(self) -> Emission[float]:
        return Emission(self, 2)


class FloatState2(Discrete[float]):

    x = var(3.)

    def update(self) -> Emission[float]:
        return Emission(self, self.x.value)


class FloatState3(Discrete[float]):
    
    x = var(3.)

    def __init__(self, next_state: StateVar, status: StateType=StateType.RUNNING, name: str=''):
        self._next_state = next_state
        self._name = name
        self._status = status

    def update(self) -> Emission[float]:
        self.x.value += 1
        return Emission(self._next_state, self.x.value)
    
    def reset(self):

        self.x.value *= 0.
    

class TestEmission:

    def test_next_state_is_correct(self):

        state = SimpleState(status=StateType.RUNNING)
        next_state, value = Emission[None](state).emit()
        assert next_state == state

    def test_value_is_correct(self):

        state = SimpleState()
        next_state, value = Emission[None](state, None).emit()
        assert value is None

    def test_value_when_float(self):

        state = EmissionState(status=StateType.RUNNING)
        next_state, value = Emission[float](state, 2.).emit()
        assert value == 2.


class TestStateWithStore:

    def test_float_emission_is_2(self):

        state = FloatState2(name='x')
        _, value = state.update().emit()
        assert value == 3.

    def test_name_is_correct(self):
        state = FloatState2(name='x')
        assert state.name == 'x'

    def test_value_is_correct_after_enter(self):
        next_state = FloatState2(name='x')
        state = FloatState3(next_state, name='x')
        next_state_, _ = state.update().emit({})

        assert next_state_ == next_state

    def test_status_is_correct(self):
        next_state = FloatState2(name='x', status=StateType.SUCCESS)
        state = FloatState3(next_state, name='first')
        next_state_, _ = state.update().emit()

        assert next_state_ == next_state


class MachineTest(FSM):

    start = state_(FloatState3, next_state=StateRef('state2'))
    state2 = state_(FloatState3, next_state=StateRef('state3'))
    state3 = state_(EmissionState, status=StateType.SUCCESS)


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

class FloatState2(Discrete[float]):

    x = var(3.)

    def update(self) -> Emission[float]:
        return Emission(self, self.x.value)

class TestFSMTaskInTree:

    class X(Tree):

        @task
        class entry(FSM):

            start = state_(FloatState3, status=StateType.READY, next_state=StateRef('state2'))
            state2 = state_(FloatState3, status=StateType.RUNNING, next_state=StateRef('state3'))
            state3 = state_(EmissionState, status=StateType.SUCCESS)


    class X2(Tree):

        @task
        class entry(FSM):

            start = state_(FloatState3, status=StateType.READY, next_state=StateRef('state2'))
            state2 = state_(FloatState3, status=StateType.RUNNING, next_state=StateRef('state3'))
            state3 = state_(EmissionState, status=StateType.FAILURE)

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
