from .common import Node, AxisAlignedRule, Tree
import numpy as np


class LiuNode(Node):
    """
    A node in an isolation forest as originally proposed by Liu
    """
    def __init__(self, left=None, right=None, rule=None, is_leaf=True, depth=0, size=0):
        """
        Node in Liu tree
        :param left: left child
        :param right: right child
        :param rule: rule to determine if left or right child, left is true, right is false
        :param is_leaf: if node is a leaf node
        :param depth: the current depth of the node
        :param size: how many data points are at this node (if not completely isolated will be nonzero)
        """
        super(LiuNode, self).__init__(content=None, left=left, right=right, rule=rule, is_leaf=is_leaf)
        self.depth = depth
        self.size = size

    def split(self, x, max_depth=25):
        """
        Splits the node on data x
        if x is a single point, nothing happens
        :param x: data point
        :param max_depth: maximum depth a node can achieve
        """
        if x.shape[0] > 1:  # there are data points to split
            self.size = x.shape[0]

            # determine the rule and save it
            dimension = np.random.choice(np.arange(x.shape[1]))
            low, high = np.min(x[:, dimension]), np.max(x[:, dimension])
            threshold = np.random.uniform(low, high)
            self.rule = AxisAlignedRule(dimension, threshold)

            # make it not a leaf node
            self.is_leaf = False
            self.left = LiuNode(depth=self.depth+1)
            self.right = LiuNode(depth=self.depth+1)

            # split the data into the children nodes
            left_indices = self.rule.evaluate(x)
            right_indices = np.logical_not(left_indices)
            self.left.size = np.sum(left_indices)
            self.right.size = np.sum(right_indices)

            # continue splitting if possible
            if self.depth < max_depth:
                self.left.split(x[left_indices, :])
                self.right.split(x[right_indices, :])


class LiuIsolationTree(Tree):
    """
    An isolation tree as proposed by liu
    """
    def __init__(self, max_depth=25):
        """
        :param max_depth: maximum depth a tree can grow to
        """
        super(LiuIsolationTree, self).__init__()
        self.count = 0
        self.max_depth = max_depth
        self.root = LiuNode()

    def fit(self, x):
        """
        grow the tree from input data
        :param x: input data
        """
        self.count = x.shape[0]
        self.root.split(x, max_depth=self.max_depth)

    def score(self, x):
        """
        Score an input data
        :param x: input data point
        :return: score
        """
        leaf_node = self.traverse(x)

        def remaining_path_length(size):
            if size > 2:
                def harmonic_number(i):
                    return np.log(i) + 0.5772156649
                return 2 * harmonic_number(size) - (2 * (size - 1) / size)
            elif size==2:
                return 1
            else:
                return 0
        return leaf_node.depth + remaining_path_length(leaf_node.size)


class LiuIsolationForest:
    """
    An ensemble of Liu Trees
    """
    def __init__(self, tree_count=1, subsample_size=255, max_depth=25):
        """
        An ensemble of Liu Isolation Trees
        :param tree_count: how many trees to grow
        :param subsample_size: how large the subsample size should be for each
        :param max_depth: the maximum depth any tree can grow to
        """
        self.max_depth = max_depth
        self.trees = [LiuIsolationTree(max_depth=self.max_depth) for _ in range(tree_count)]
        self.subsample_size = subsample_size

    def fit(self, x):
        """
        fit all the trees on the data
        :param x: input data
        """
        for tree in self.trees:
            # pick a random sample of the requested size and train the tree
            sample = x[np.random.choice(np.arange(x.shape[0]), size=self.subsample_size), :]
            tree.fit(sample)

    def score(self, x):
        """
        score the point
        :param x: input point
        :return: isolation score
        """
        expectation = np.mean([tree.score(x) for tree in self.trees])

        def harmonic_number(i):
            return np.log(i) + 0.5772156649
        return np.power(2, -expectation/harmonic_number(self.subsample_size))
