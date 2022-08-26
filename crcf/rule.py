from __future__ import annotations
from typing import Type
import numpy as np
from abc import abstractmethod, ABC


class Rule(ABC):
    """
    A generalized representation of a decision tree rule.

    Public Methods:
        - evaluate(x): determines whether the outcome is true or false from the rule
        - generate(bounding_box): create a new rule given a bounding box

    To create a new kind of rule one must implement the following:
        - _evaluate(x): determines the output of a single point
        - _generate_uniform(bounding_box): generates a new rule at uniform from the bounding box
        - _generate_biased(bounding_box): generates a new biased sample rule from the bounding box
    """
    def __init__(self) -> None:
        pass

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Decide the path of a set of points using the rule


        Parameters
        ----------
        x : np.ndarray
            a set of points, e.g. np.array([[1,2,3],[4,5,6]])

        Returns
        -------
        np.ndarray of bool
            True if point goes to left side, False for right side
        """
        return np.array([self._evaluate(xx) for xx in x])
        # TODO: this is much slower than it has to be, use array methods

    @abstractmethod
    def _evaluate(self, x: np.ndarray) -> bool:
        """Score a single point

        Parameters
        ----------
        x : np.ndarray
            a single example, 1-D numpy array

        Returns
        -------
        bool
            True if point goes to left side, False for right side
        """
        pass

    @classmethod
    def generate(cls, bounding_box: np.ndarray, mode: str = "uniform") -> Rule:
        """Generates a new rule from the bounding box.

        Parameters
        ----------
        bounding_box : np.ndarray
            the minimal axis parallel bounding box that contains all the data points
            for a given node, e.g. [[1,2], [1,10]] this means the value in first dimension has values
            from 1 to 2 and in the second dimension 1 to 10.
        mode : str
            the strategy of picking a dimension:
            - "uniform": all dimensions are equally considered
            - "biased": dimensions with larger value ranges are weighted proportionally more
                in two dimensions if the bounding box were [[1,2], [1,10]] this means the
                value in first dimension has values from 1 to 2 and in the second dimension 1 to 10.
                In the biased setting the first dimension has a weight of 2-1=1 and the second dimension
                has weight 10-1=9. The second dimension is 9 times more likely to be chosen.


        Returns
        -------
        Rule
            a new rule
        """

        # switch on the mode and call the appropriate function
        if mode == "uniform":
            return cls._generate_uniform(bounding_box)
        elif mode == "biased":
            return cls._generate_biased(bounding_box)
        else:
            raise RuntimeError("mode must either be uniform or biased, not {}".format(mode))

    @classmethod
    @abstractmethod
    def _generate_uniform(cls, bounding_box: np.ndarray) -> Rule:
        """Generate a rule with no special attention to the bounding box, i.e. all dimeensions are equally important


        Parameters
        ----------
        bounding_box : np.ndarray
            the minimal axis parallel bounding box that contains all the data points
            for a given node, e.g. [[1,2], [1,10]] this means the value in first dimension has values
            from 1 to 2 and in the second dimension 1 to 10.

        Returns
        -------
        Rule
            a new random rule
        """
        pass

    @classmethod
    @abstractmethod
    def _generate_biased(cls, bounding_box: np.ndarray) -> Rule:
        """Generate a rule with no weighted attention to the bounding box, i.e. wider dimensions are more important


        Parameters
        ----------
        bounding_box : np.ndarray
            the minimal axis parallel bounding box that contains all the data points
            for a given node, e.g. [[1,2], [1,10]] this means the value in first dimension has values
            from 1 to 2 and in the second dimension 1 to 10.

        Returns
        -------
        Rule
            a new random rule
        """
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass

    @abstractmethod
    def __eq__(self, other: Type[Rule]) -> bool:
        pass


class AxisAlignedRule(Rule):
    """
    This rule uses cuts that are axis aligned. Thus, the rule is defined by its dimension and what threshold value
    for that dimension
    """
    def __init__(self, dimension: int, value: float):
        """Initialize an AxisAlignedRule

        Parameters
        ----------
        dimension : int
            the number, 0-indexed, describing the dimension of the cut
        value : float
            the  value to threshold on, i.e. x < value is true
        """
        super().__init__()
        self.dimension, self.value = dimension, value

    def _evaluate(self, x: np.ndarray) -> bool:
        """Determine the path for a single point, points less than the threshold value are true


        Parameters
        ----------
        x : np.ndarray
             single example, e.g. np.ndarray([1,2,3])

        Returns
        -------
        bool
            true if x[dimension] < value and false if x[dimension] >= value
        """
        return x[self.dimension] < self.value

    @classmethod
    def _generate_uniform(cls, bounding_box: np.ndarray) -> AxisAlignedRule:
        """Generate a rule with no special attention to the bounding box, i.e. all dimeensions are equally important


        Parameters
        ----------
        bounding_box : np.ndarray
            the minimal axis parallel bounding box that contains all the data points
            for a given node, e.g. [[1,2], [1,10]] this means the value in first dimension has values
            from 1 to 2 and in the second dimension 1 to 10.

        Returns
        -------
        Rule
            a new random rule
        """
        dimension = np.random.randint(0, bounding_box.shape[0])
        value = np.random.uniform(bounding_box[dimension][0], bounding_box[dimension][1])
        return AxisAlignedRule(dimension, value)

    @classmethod
    def _generate_biased(cls, bounding_box: np.ndarray) -> AxisAlignedRule:
        """Generate a rule with no weighted attention to the bounding box, i.e. wider dimensions are more important


        Parameters
        ----------
        bounding_box : np.ndarray
            the minimal axis parallel bounding box that contains all the data points
            for a given node, e.g. [[1,2], [1,10]] this means the value in first dimension has values
            from 1 to 2 and in the second dimension 1 to 10.

        Returns
        -------
        Rule
            a new random rule
        """
        lengths = np.diff(bounding_box)
        dimension = np.random.choice(np.arange(bounding_box.shape[0]),
                                     p=lengths.flatten()/np.sum(lengths))
        value = np.random.uniform(bounding_box[dimension][0], bounding_box[dimension][1])
        return AxisAlignedRule(dimension, value)

    def __str__(self) -> str:
        return f"x[{self.dimension}]<{self.value:.2f}"

    def __eq__(self, other: AxisAlignedRule) -> bool:
        return isinstance(other, AxisAlignedRule) and other.dimension == self.dimension and other.value == self.value


class NonAxisAlignedRule(Rule):
    """
    A cut is instead a hyperplane that divides the space. This hyperplane is thus a linear combination of dimensions.
    It is described by a normal vector to the hyperplane and a point the hyperplane passes through.
    """
    def __init__(self, normal: np.ndarray, point: np.ndarray):
        """Create a non-axis-aligned rule

        Parameters
        ----------
        normal : np.ndarray
            the normal vector for the hyperplane
        point : np.ndarray
            a point the hyperplane must pass through
        """
        super().__init__()
        self.normal, self.point = normal, point
        self.offset = normal.dot(point)  # the offset used in calculations

    def _evaluate(self, x: np.ndarray) -> bool:
        """Score a single point

        Parameters
        ----------
        x : np.ndarray
            a single example, 1-D numpy array

        Returns
        -------
        bool
            True if point goes to left side, False for right side
        """
        return np.inner(self.normal, x) < self.offset

    @classmethod
    def _generate_uniform(cls, bounding_box: np.ndarray) -> NonAxisAlignedRule:
        """Generate a rule with no special attention to the bounding box, i.e. all dimeensions are equally important


        Parameters
        ----------
        bounding_box : np.ndarray
            the minimal axis parallel bounding box that contains all the data points
            for a given node, e.g. [[1,2], [1,10]] this means the value in first dimension has values
            from 1 to 2 and in the second dimension 1 to 10.

        Returns
        -------
        Rule
            a new random rule
        """
        normal = np.random.uniform(-1, 1, size=bounding_box.shape[0])
        point = np.array(np.random.uniform(low, high) for low, high in bounding_box)
        return NonAxisAlignedRule(normal, point)

    @classmethod
    def _generate_biased(cls, bounding_box: np.ndarray) -> NonAxisAlignedRule:
        """Generate a rule with no weighted attention to the bounding box, i.e. wider dimensions are more important


        Parameters
        ----------
        bounding_box : np.ndarray
            the minimal axis parallel bounding box that contains all the data points
            for a given node, e.g. [[1,2], [1,10]] this means the value in first dimension has values
            from 1 to 2 and in the second dimension 1 to 10.

        Returns
        -------
        Rule
            a new random rule
        """
        raise NotImplementedError("Will be added in a later version.")

    def __str__(self) -> str:
        return "x^T{}<{:.2f}".format("[" + ("{:.2f},"*self.normal.shape[0]).format(self.normal) + "]",
                                     self.offset)

    def __eq__(self, other: NonAxisAlignedRule) -> bool:
        return isinstance(other, NonAxisAlignedRule) and np.all(other.normal == self.normal) and\
            np.all(other.point == self.point)
