"""
TODO: module doc string
"""
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
        self.offset = normal.dot(point) # the offset used in calculations

    def _evaluate(self, x):
        """
        Determine the path for a single point, points less than the threshold value are true
        :param x: a single example, e.g. np.ndarray([1,2,3])
        :type x: np.ndarray
        :return: true if x[dimension] < value and false if x[dimension] >= value
        :rtype: bool
        """
        return np.inner(self.normal, x) < self.offset

    @classmethod
    def _generate_uniform(cls, bounding_box):
        """
        Generate a rule with no special attention to the bounding box, i.e. all dimeensions are equally important
        :param bounding_box: the minimal axis parallel bounding box that contains all the data points
            for a given node, e.g. [[1,2], [1,10]] this means the value in first dimension has values
            from 1 to 2 and in the second dimension 1 to 10.
        :type bounding_box: np.ndarray
        :return: a new rule
        :rtype: NonAxisAlignedRule
        """
        normal = np.random.uniform(-1, 1, size=bounding_box.shape[0])
        point = np.array(np.random.uniform(low, high) for low, high in bounding_box)
        return NonAxisAlignedRule(normal, point)

    @classmethod
    def _generate_biased(cls, bounding_box):
        raise NotImplementedError("Will be added in a later version.")


class CombinationTree:
    """
    Combination Robust Cut Trees are a generalization of robust random cut trees of Guha et al. (2016)
    and isolation trees of Liu et al. (2010). The parameters can be set to get the exact formulation of each or
    a tree that is an interpolation of the two modes:

    ISOLATION TREE PARAMETERS:
    TODO: complete

    ROBUST RANDOM CUT TREE PARAMETERS:
    TODO: complete

    """
    def __init__(self, x=None, depth_limit=None, rule_kind=AxisAlignedRule, rule_mode="uniform"):
        # properties of the tree
        self.depth_limit = depth_limit
        self.rule_kind = rule_kind
        self.rule_mode = rule_mode

        # properties of every node
        self.rule = None
        self.count = 0
        self.bounding_box = None
        self.is_leaf_ = True
        self.depth_ = 0

        # manage the relationships between other points
        self.parent = None
        self.left_child = None
        self.right_child = None

        # build the tree if any points are passed to it
        if x is not None:
            self._build(x)

    def _build(self, x):
        """
        Grow a tree from the points in x
        :param x: a set of points
        :type x: np.ndarray
        :return: the newly built node
        :rtype: CombinationTree
        """
        # update the rule and local properties with regards to x
        self.count = x.shape[0]
        self.bounding_box = np.array([[np.nanmin(x[:, i]), np.nanmax(x[:, i])] for i in range(x.shape[1])])
        if self.count == 1:  # is a leaf node
            self.is_leaf_ = True
        else:
            self.is_leaf_ = False
            self.rule = self.rule_kind.generate(self.bounding_box, mode=self.rule_mode)

            # create the children nodes and create relationships
            evaluation = self.rule.evaluate(x)  # whether the points go to the left subtree
            left_child = CombinationTree(x=x[evaluation])
            right_child = CombinationTree(x=x[np.logical_not(evaluation)])
            left_child.parent, right_child.parent = self, self
            self.left_child, self.right_child = left_child, right_child
            self.left_child.depth_ = self.depth_ + 1
            self.right_child.depth_ = self.depth_ + 1
        return self

    def depth(self, x, estimated=False):
        """
        Determine the depth of where x is in the tree
        :param x: a point
        :type x: np.ndarray
        :param estimated: if True will use the counts at a leaf node
            to estimate how far down the tree the point would be if it had been grown completely
        :type estimated: bool
        :return: the depth of the point
        :rtype: int
        """
        def harmonic(n):
            """
            :param n: index
            :type n: int
            :return: the nth harmonic number
            :rtype: float
            """
            return np.log(n) + np.euler_gamma

        def expected_length(n):
            """
            :param n: count remaining in leaf node
            :type n: int
            :return: the expected average length had the tree continued to grow
            :rtype: float
            """
            return 2 * harmonic(n-1) - (2*(n-1) / n) if n > 1 else 0

        leaf = self.find(x)
        extension = expected_length(leaf.count) if estimated else 0
        return leaf.depth_ + extension

    def displacement(self, x):
        """
        The displacement of a point x in the tree
        :param x: a data sample
        :type x: np.ndarray
        :return: the "surprise" or displacement induced by including x in the tree
        :rtype: float
        """
        leaf = self.find(x)
        sibling = leaf.parent.left_child if leaf.parent.left_child is not leaf else leaf.parent.right_child
        return sibling.count

    def codisplacement(self, x):
        """
        Codisplacement allows for colluders in the displacement per RRCF paper [Guha+2016]
        :param x: a data sample
        :type x: np.ndarray
        :return: the collusive displacement induced by including x in the tree
        :rtype: float
        """
        node = self.find(x)
        best_codisp = 0

        # work upward from the leaf node to the parent considering all paths allong
        while not node.is_root():
            sibling = node.parent.left_child if node.parent.left_child is not node else node.parent.right_child
            collusive_size, sibling_size = node.count, sibling.count
            this_codisp = sibling_size/collusive_size

            # compare to the best_codisp and update if higher
            if this_codisp > best_codisp:
                best_codisp = this_codisp
            node = node.parent
        return best_codisp

    def find(self, x):
        """
        Find the correct leaf node for an input point x
        :param x: a data point
        :type x: np.ndarray
        :return: the leaf node x would belong with
        :rtype: CombinationTree
        """
        node = self
        while not node.is_leaf_:
            node = node.left_child if node.rule.evaluate(np.array([x])) else node.right_child
        return node

    def insert(self, x):
        raise NotImplementedError("will be implemented in a later version")

    def remove(self, x):
        raise NotImplementedError("will be implemented in a later version")

    def is_leaf(self):
        """
        :return: True if the Tree consists of only a leaf node, False otherwise
        :rtype: bool
        """
        return self.is_leaf_

    def is_internal(self):
        """
        :return: True if current node is internal, otherwise False
        :rtype: bool
        """
        return not self.is_leaf()

    def is_root(self):
        """
        :return: True if the current node is the root of the tree, false otherwise
        :rtype: bool
        """
        return self.parent is None

    def score(self, X, theta=1, use_codisplacement=True, estimated=False):
        """
        Calculate the anomaly score
        :param X: a set of points to score
        :type X: np.ndarray
        :param theta: the combination value for theta * depth + (1-theta) * [co]disp,
            theta=0 is the isolation forest version
            theta=1 is the robust random cut forest version
        :type theta: float
        :param use_codisplacement: if True uses codisplacement, if false uses displacement
        :type use_codisplacement: bool
        :param estimated: whether to use the absolute depth or the estimated depths from count, see depth()
        :type estimated: bool
        :return: the anomaly score
        :rtype: float
        """
        if use_codisplacement:
            if theta == 1:  # only use depth
                return np.array([self.depth(x) for x in X])
            elif theta == 0:  # only use codisplacement
                return np.array([self.codisplacement(x) for x in X])
            else:  # use combination of both
                return np.array([theta * self.depth(x, estimated=estimated)
                                 + (1-theta) * self.codisplacement(x) for x in X])
        else:  # use displacement
            if theta == 1:  # only use depth
                return np.array([self.depth(x, estimated=estimated) for x in X])
            elif theta == 0:  # only use displacement
                return np.array([self.displacement(x) for x in X])
            else:  # use combination of both
                return np.array([theta * self.depth(x, estimated=estimated)
                                 + (1 - theta) * self.displacement(x) for x in X])

    def save(self, path):
        raise NotImplementedError("will be implemented in a later version")

    def load(self, path):
        raise NotImplementedError("will be implemented in a later version")


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
        raise NotImplementedError("will be implemented in a later version")

    def save(self, path):
        raise NotImplementedError("will be implemented in a later version")

    def load(self, path):
        raise NotImplementedError("will be implemented in a later version")

    def insert(self, path):
        raise NotImplementedError("will be implemented in a later version")

    def remove(self, path):
        raise NotImplementedError("will be implemented in a later version")

    def score(self, path):
        raise NotImplementedError("will be implemented in a later version")


class IsolationForest(Forest):
    def __init__(self, num_trees=100, tree_properties=None):
        super().__init__(num_trees=num_trees, tree_properties=tree_properties)


class RobustRandomCutForest(Forest):
    def __init__(self, num_trees=100, tree_properties=None):
        super().__init__(num_trees=num_trees, tree_properties=tree_properties)
