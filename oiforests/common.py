from abc import ABCMeta, abstractmethod


class Node:
    """
    A basic node for a binary tree that contains some 'content'
    """

    def __init__(self, content=None, left=None, right=None, rule=None, is_leaf=True):
        """
        Node in general binary tree
        :param content: content stored at the node
        :param left: left child, None if empty
        :param right: right child, None if empty
        :param rule: rule to determine if left or right
        :param is_leaf: whether node is a leaf node
        """
        self.content = content
        self.left = left
        self.right = right
        self.rule = rule
        self.is_leaf = is_leaf

    def traverse(self, x):
        """
        Travel through the node determining if you should branch left or right, if it's a leaf node just returns itself
        :param x: data point
        :return: the child node the data point would pass to if the rule is executed, or self if a leaf node
        """
        if self.is_leaf:
            return self
        else:  # is not a leaf node
            return self.left.traverse(x) if self.rule.evaluate(x) else self.right.traverse(x)


class Rule(metaclass=ABCMeta):
    """
    An abstract rule that could be used at each node
    """
    def __init__(self):
        pass

    @abstractmethod
    def evaluate(self, x):
        """
        Determines if the rule passes to the left or the right
        :param x: input data point or set of data points
        :return: which direction to go, True is left, False is right
        """
        pass


class AxisAlignedRule(Rule):
    """
    A rule that looks at one dimension of a data point and compares it to a threshold, an axis aligned cut
    """
    def __init__(self, dimension_number, threshold):
        """
        :param dimension_number: which dimension to consider
        :param threshold: the threshold for decisions
        """
        super(AxisAlignedRule, self).__init__()
        self.dimension_number = dimension_number
        self.threshold = threshold

    def evaluate(self, x):
        """
        Tell which side the rule sends x, true is left, false is right
        :param x: a data point or a set of data
        :return: a single boolean or a boolean array
        """
        if len(x.shape) == 1:  # a single number is passed
            return x[self.dimension_number] < self.threshold
        else:  # a set of numbers are passed
            return x[:, self.dimension_number] < self.threshold


class Tree(metaclass=ABCMeta):
    """
    An abstract representation of a tree
    """
    def __init__(self):
        self.root = Node()

    def traverse(self, x):
        """
        travel through the tree to a leaf node
        :param x: a data point
        :return: leaf node that x lands in
        """
        return self.root.traverse(x)
