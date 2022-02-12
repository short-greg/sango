from .std import Status, LinearPlanner, Conditional


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
