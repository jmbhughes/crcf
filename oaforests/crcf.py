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
    def __init__(self):
        pass

    def evaluate(self, X):
        """
        Decide the path of a set of points using the rule
        :param X: a set of points, e.g. np.array([[1,2,3],[4,5,6]])
        :type X: np.ndarray
        :return: true if goes to left side, false for right side
        :rtype: bool
        """
        return np.array([self._evaluate(x) for x in X])

    @abstractmethod
    def _evaluate(self, x):
        """
        Score a single point
        :param x: a single example, e.g. np.ndarray([1,2,3])
        :type x: np.ndarray
        :return: true if goes to left side, false for right side
        :rtype: bool        """
        pass

    @classmethod
    def generate(cls, bounding_box, mode="uniform"):
        """
        Generates a new rule from the bounding box.
        The mode determines the strategy of picking a dimension.

        :param bounding_box: the minimal axis parallel bounding box that contains all the data points
            for a given node, e.g. [[1,2], [1,10]] this means the value in first dimension has values
            from 1 to 2 and in the second dimension 1 to 10.
        :type bounding_box: np.ndarray
        :param mode: the strategy of picking a dimension:
            - "uniform": all dimensions are equally considered
            - "biased": dimensions with larger value ranges are weighted proportionally more
                in two dimensions if the bounding box were [[1,2], [1,10]] this means the
                value in first dimension has values from 1 to 2 and in the second dimension 1 to 10.
                In the biased setting the first dimension has a weight of 2-1=1 and the second dimension
                has weight 10-1=9. The second dimension is 9 times more likely to be chosen.
        :type mode: str
        :return: a new rule
        :rtype: Rule
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
    def _generate_uniform(cls, bounding_box):
        """
        Generate a rule with no special attention to the bounding box, i.e. all dimeensions are equally important
        :param bounding_box: the minimal axis parallel bounding box that contains all the data points
            for a given node, e.g. [[1,2], [1,10]] this means the value in first dimension has values
            from 1 to 2 and in the second dimension 1 to 10.
        :type bounding_box: np.ndarray
        :return: a new rule
        :rtype: Rule
        """
        pass

    @classmethod
    @abstractmethod
    def _generate_biased(cls, bounding_box):
        """
        Generate a rule with no weighted attention to the bounding box, i.e. wider dimensions are more important
        :param bounding_box: the minimal axis parallel bounding box that contains all the data points
            for a given node, e.g. [[1,2], [1,10]] this means the value in first dimension has values
            from 1 to 2 and in the second dimension 1 to 10.
        :type bounding_box: np.ndarray
        :return: a new rule
        :rtype: Rule
        """
        pass


class AxisAlignedRule(Rule):
    """
    This rule uses cuts that are axis aligned. Thus, the rule is defined by its dimension and what threshold value
    for that dimension
    """
    def __init__(self, dimension, value):
        """
        :param dimension: the number, 0-indexed, describing the dimension of the cut
        :type dimension: int
        :param value: the  value to threshold on, i.e. x < value is true
        :type value: float
        """
        super().__init__()
        self.dimension, self.value = dimension, value

    def _evaluate(self, x):
        """
        Determine the path for a single point, points less than the threshold value are true
        :param x: a single example, e.g. np.ndarray([1,2,3])
        :type x: np.ndarray
        :return: true if x[dimension] < value and false if x[dimension] >= value
        :rtype: bool
        """
        return x[self.dimension] < self.value

    @classmethod
    def _generate_uniform(cls, bounding_box):
        """
        Generate a rule with no special attention to the bounding box, i.e. all dimeensions are equally important
        :param bounding_box: the minimal axis parallel bounding box that contains all the data points
            for a given node, e.g. [[1,2], [1,10]] this means the value in first dimension has values
            from 1 to 2 and in the second dimension 1 to 10.
        :type bounding_box: np.ndarray
        :return: a new rule
        :rtype: AxisAlignedRule
        """
        dimension = np.random.randint(0, bounding_box.shape[0])
        value = np.random.uniform(bounding_box[dimension][0], bounding_box[dimension][1])
        return AxisAlignedRule(dimension, value)

    @classmethod
    def _generate_biased(cls, bounding_box):
        """
        Generate a rule with no weighted attention to the bounding box, i.e. wider dimensions are more important
        :param bounding_box: the minimal axis parallel bounding box that contains all the data points
            for a given node, e.g. [[1,2], [1,10]] this means the value in first dimension has values
            from 1 to 2 and in the second dimension 1 to 10.
        :type bounding_box: np.ndarray
        :return: a new rule
        :rtype: AxisAlignedRule
        """
        lengths = np.diff(bounding_box)
        dimension = np.random.choice(np.arange(bounding_box.shape[0]),
                                     p=lengths.flatten()/np.sum(lengths))
        value = np.random.uniform(bounding_box[dimension][0], bounding_box[dimension][1])
        return AxisAlignedRule(dimension, value)


class NonAxisAlignedRule(Rule):
    """
    A cut is instead a hyperplane that divides the space. This hyperplane is thus a linear combination of dimensions.
    It is described by a normal vector to the hyperplane and a point the hyperplane passes through.
    """
    def __init__(self, normal, point):
        """
        Create a non-axis aligned rule
        :param normal: the normal vector for the hyperplane
        :type normal: np.ndarray
        :param point: a point the hyperplane must pass through
        :type point: np.ndarray
        """
        super().__init__()
        self.normal, self.point = normal, point

    def _evaluate(self, x):
        pass

    @classmethod
    def _generate_uniform(cls, bounding_box):
        pass

    @classmethod
    def _generate_biased(cls, bounding_box):
        pass


class CombinationTree:
    """
    Combination Robust Cut Trees are a generalization of robust random cut trees of Guha et al. (2016)
    and isolation trees of Liu et al. (2010). The parameters can be set to get the exact formulation of each or
    a tree that is an interpolation of the two modes:

    ISOLATION TREE PARAMETERS

    ROBUST RANDOM CUT TREE PARAMETERS
    """
    def __init__(self, X=None, depth_limit=None):
        self.rule = None
        self.count = 0
        self.bounding_box = None
        self.is_leaf = True

        # manage the dependencies
        self.parent = None
        self.left_child = None
        self.right_child = None

    def displacement(self, x):
        return 0

    def codisplacement(self, x):
        return 0

    def depth(self, x):
        return 0

    def insert(self, x):
        pass

    def remove(self, x):
        pass

    def is_leaf(self):
        return self.is_leaf

    def is_internal(self):
        return not self.is_leaf

    def is_root(self):
        return self.parent is None

    def score(self, X, theta=1, use_codisplacement=True):
        if use_codisplacement:
            if theta == 1: # only use depth
                return np.array([self.depth(x) for x in X])
            elif theta == 0: # only use codisplacement
                return np.array([self.codisplacement(x) for x in X])
            else: # use combination of both
                return np.array([theta * self.depth(x) + (1-theta) * self.codisplacement(x) for x in X])
        else:  # use displacement
            if theta == 1:  # only use depth
                return np.array([self.depth(x) for x in X])
            elif theta == 0:  # only use displacement
                return np.array([self.displacement(x) for x in X])
            else:  # use combination of both
                return np.array([theta * self.depth(x) + (1 - theta) * self.displacement(x) for x in X])

    def save(self, path):
        pass

    def load(self, path):
        pass


class IsolationTree(CombinationTree):
    def __init__(self, X=None, depth_limits=None):
        super().__init__(X, depth_limits)

    def codisplacement(self, x):
        raise NotImplementedError("Isolation trees do not have codisplacement. " +
                                  "See the CombinationTree or the RobustRandomCutTree instead.")

    def displacement(self, x):
        raise NotImplementedError("Isolation trees do not have displacement. " +
                                  "See the CombinationTree or the RobustRandomCutTree instead.")


class RobustRandomCutTree(CombinationTree):
    def __init__(self, X=None):
        super().__init__(X, depth_limit=None)


class Forest:
    def __init__(self, num_trees=100, tree_properties=None):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass

    def insert(self, path):
        pass

    def remove(self, path):
        pass

    def score(self, path):
        pass


class IsolationForest(Forest):
    def __init__(self, num_trees=100, tree_properties=None):
        super().__init__(num_trees=num_trees, tree_properties=tree_properties)


class RobustRandomCutForest(Forest):
    def __init__(self, num_trees=100, tree_properties=None):
        super().__init__(num_trees=num_trees, tree_properties=tree_properties)
