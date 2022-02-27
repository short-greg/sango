import typing
from ._mods.std import Filter, Task


class IntersectFilter(Filter):

    def __init__(self, filters: typing.List[Filter]):
        """initializer

        Args:
            filters (typing.List[Filter]): Filters to intersect
        """
        self._filters = filters
    
    def check(self, node) -> bool:
        """Check if the node passes the filter

        Args:
            node: Node to filter

        Returns:
            bool
        """

        for filter in self._filters:
            if not filter.check(node):
                return False
        return True


class UnionFilter(Filter):

    def __init__(self, filters: typing.List[Filter]):
        """initializer

        Args:
            filters (typing.List[Filter]): Filters to intersect
        """
        self._filters = filters
    
    def check(self, node) -> bool:
        """Check if the node passes the filter

        Args:
            node: Node to filter

        Returns:
            bool
        """

        for filter in self._filters:
            if filter.check(node):
                return True
        return False


class StatusFilter(Filter):
    """Filter based on the status of a node
    """

    def __init__(self, statuses: typing.Iterable):
        """initializer

        Args:
            statuses (typing.Iterable): Statuses to filter out
        """

        self._statuses = set(statuses)
    
    def check(self, node) -> bool:
        """Filter based 

        Args:
            node

        Returns:
            bool
        """
        if isinstance(node, Task):
            return node.status in self._statuses
        return False

class StatusExcluder(Filter):
    """Filter out based on the status of a node
    """

    def __init__(self, statuses: typing.Iterable):
        """initializer

        Args:
            statuses (typing.Iterable): Statuses to filter out
        """

        self._statuses = set(statuses)
    
    def check(self, node) -> bool:
        """Filter based 

        Args:
            node

        Returns:
            bool
        """
        if isinstance(node, Task):
            return node.status not in self._statuses
        return False
