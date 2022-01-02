from sango.nodes import Status, var
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
        emission = Emission[None](state)
        assert emission.next_state == state

    def test_value_is_correct(self):

        state = SimpleState()
        emission = Emission[None](state, None)
        assert emission.value is None

    def test_value_when_float(self):

        state = EmissionState(status=StateType.RUNNING)
        emission = Emission[float](state, 2.)
        assert emission.value == 2.


class TestStateWithStore:

    def test_float_emission_is_2(self):

        state = FloatState2(name='x')
        emission = state.update()
        assert emission.value == 3.

    def test_name_is_correct(self):
        state = FloatState2(name='x')
        assert state.name == 'x'

    def test_value_is_correct_after_enter(self):
        next_state = FloatState2(name='x')
        state = FloatState3(next_state, name='x')
        emission = state.update()

        assert emission.next_state == next_state

    def test_status_is_correct(self):
        next_state = FloatState2(name='x', status=StateType.FINAL)
        state = FloatState3(next_state, name='first')
        emission = state.update()

        assert emission.next_state == next_state


class MachineTest(FSM):

    start = state_(FloatState3, next_state=StateRef('state2'))
    state2 = state_(FloatState3, next_state=StateRef('state3'))
    state3 = state_(EmissionState, status=StateType.FINAL)


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

    # TODO: NEED TO UPDATE THIS!
    # def test_basic_machine_reset_is_correct(self):
    #     machine = MachineTest('tester')
    #     machine.tick()
    #     status = machine.tick()
    #     assert status == Status.SUCCESS
