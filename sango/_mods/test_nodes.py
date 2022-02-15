from .nodes import ShufflePlanner
from .std import Conditional, LinearPlanner
import random

class DummyNegative(Conditional):

    def check(self):
        return False


class DummyPositive(Conditional):

    def check(self):
        return True


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

