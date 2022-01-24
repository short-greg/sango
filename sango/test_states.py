import pytest
from sango.nodes import Status, Tree, task, var_
from sango.vars import Args, Ref, Var
from .states import FSM, Discrete, Emission, Failure, Running, StateID, StateLink, StateVar, Success, fsmstate, state, state_, to_state, to_status


class SimpleState(Running[None]):

    def update(self) -> Emission[None]:
        return Emission(self)


class EmissionState(Failure[float]):

    def update(self) -> Emission[float]:
        return Emission(self, 2)


class FloatState2(Running[float]):

    x = var_(3.)

    def update(self) -> Emission[float]:
        return Emission(self, self.x.val)


class FloatState3(Running[float]):
    
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


class FloatStateFailure(Failure[float]):

    x = var_(3.)

    def emit_value(self) -> Emission[float]:
        return self.x.val


class FloatStateSuccess(Success[float]):

    x = var_(3.)

    def emit_value(self) -> Emission[float]:
        return self.x.val


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
