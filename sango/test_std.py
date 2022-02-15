import random

import pytest
from .std import StatusFilter, Sequence, ShufflePlanner, Status, LinearPlanner, Conditional


class TestStatus:

    def test_done_if_successful(self):
        status = Status.SUCCESS
        assert status.done 

    def test_done_if_failed(self):
        status = Status.SUCCESS
        assert status.done 

    def test_done_if_done(self):
        status = Status.DONE
        assert status.done 


class DummyNegative(Conditional):

    def check(self):
        return False


class DummyPositive(Conditional):

    def check(self):
        return True


class TestLinearPlanner:

    def test_loop_over_linear_planner(self):

        planner = LinearPlanner([DummyPositive(name='x'), DummyNegative(name='y')])
        assert isinstance(planner.cur, DummyPositive)
        planner.adv()
        assert isinstance(planner.cur, DummyNegative)
        
    def test_loop_over_linear_planner(self):

        planner = LinearPlanner([DummyPositive(name='x'), DummyNegative(name='y')])
        assert isinstance(planner.cur, DummyPositive)
        planner.adv()
        assert isinstance(planner.cur, DummyNegative)


class TestShufflePlanner:

    def test_first_element_in_plan_is_correct(self):

        random.seed(2)
        planner = ShufflePlanner(LinearPlanner(
            [DummyNegative(name='x'), DummyPositive(name='y')]
        ))
        assert isinstance(planner.cur, DummyPositive)

    def test_second_element_in_plan_is_correct(self):

        random.seed(2)
        planner = ShufflePlanner(LinearPlanner(
            [DummyNegative(name='x'), DummyPositive(name='y')]))
        planner.adv()
        assert isinstance(planner.cur, DummyNegative)

    def test_third_element_in_plan_is_end(self):

        random.seed(2)
        planner = ShufflePlanner(LinearPlanner(
            [DummyNegative(name='x'), DummyPositive(name='y')]))
        planner.adv()
        planner.adv()
        assert planner.end()

    def test_rev_past_beginning_returns_false(self):

        random.seed(2)
        planner = ShufflePlanner(LinearPlanner(
            [DummyNegative(name='x'), DummyPositive(name='y')]))
        planner.adv()
        planner.rev()

        assert planner.rev() is False

    def test_adv_past_end_returns_false(self):

        random.seed(2)
        planner = ShufflePlanner(LinearPlanner(
            [DummyNegative(name='x'), DummyPositive(name='y')]))
        planner.adv()
        planner.adv()

        assert planner.adv() is False


class TestIteration:

    def test_iterate_over_sequence(self):

        sequence = Sequence([DummyNegative(name='x'), DummyPositive(name='y')])
        nodes = [node for node in sequence.iterate()]
        assert isinstance(nodes[0], DummyNegative)
        assert isinstance(nodes[1], DummyPositive)

    def test_iterate_over_action(self):
        action = DummyNegative(name='x')
        nodes = [node for node in action.iterate()]
        assert len(nodes) == 0

    def test_iterate_over_two_sequences(self):
        sequence1 = Sequence([DummyNegative(name='x'), DummyPositive(name='y')])
        sequence2 = Sequence([DummyNegative(name='x'), DummyPositive(name='y')])
        sequence3 = Sequence([sequence1, sequence2])
        nodes = [node for node in sequence3.iterate()]
        assert len(nodes) == 6

    def test_iterate_over_two_sequences_not_depth(self):
        sequence1 = Sequence([DummyNegative(name='x'), DummyPositive(name='y')])
        sequence2 = Sequence([DummyNegative(name='x'), DummyPositive(name='y')])
        sequence3 = Sequence([sequence1, sequence2])
        nodes = [node for node in sequence3.iterate(deep=False)]
        assert len(nodes) == 2

    def test_iterate_over_two_sequences_running_status(self):
        sequence1 = Sequence([DummyPositive(name='x'), DummyPositive(name='y')])
        sequence2 = Sequence([DummyPositive(name='x'), DummyPositive(name='y')])
        sequence3 = Sequence([sequence1, sequence2])
        sequence3.tick()
        nodes = [node for node in sequence3.iterate(
            StatusFilter([Status.RUNNING]
        ), deep=True)]

        assert len(nodes) == 1

    def test_iterate_over_two_sequences_running_status(self):
        sequence1 = Sequence([DummyPositive(name='x'), DummyPositive(name='y')])
        sequence2 = Sequence([DummyPositive(name='x'), DummyPositive(name='y')])
        sequence3 = Sequence([sequence1, sequence2])
        sequence3.tick()
        nodes = [node for node in sequence3.iterate(
            StatusFilter([Status.RUNNING, Status.SUCCESS]), 
            deep=True)]

        assert len(nodes) == 2
