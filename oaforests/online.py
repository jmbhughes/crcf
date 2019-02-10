from .common import Node, AxisAlignedRule, Tree
import numpy as np


class OnlineNode(Node):
    """
    A node in an isolation forest as originally proposed by Liu
    """
    def __init__(self, content=None, split_count=2, left=None, right=None, rule=None, is_leaf=True, depth=0, size=0):
        """
        Node in Liu tree
        :param left: left child
        :param right: right child
        :param rule: rule to determine if left or right child, left is true, right is false
        :param is_leaf: if node is a leaf node
        :param depth: the current depth of the node
        :param size: how many data points are at this node (if not completely isolated will be nonzero)
        """
        super(OnlineNode, self).__init__(content=content, left=left, right=right, rule=rule, is_leaf=is_leaf)
        self.content = None
        self.depth = depth
        self.size = size
        self.split_count = split_count

    def split(self, x, max_depth=25):
        """
        Splits the node on data x
        :param x: data point
        :param max_depth: maximum depth a node can achieve
        """
        if self.content is None:
            self.content = np.empty((0, x.shape[0]))

        if self.depth < max_depth and 1 + self.content.shape[0] >= self.split_count:  # there are data points to split
            x = np.concatenate([self.content, x.reshape((1, -1))], axis=0)
            self.size = x.shape[0]

            # determine the rule and save it
            dimension = np.random.choice(np.arange(x.shape[1]))
            low, high = np.min(x[:, dimension]), np.max(x[:, dimension])
            threshold = np.random.uniform(low, high)
            self.rule = AxisAlignedRule(dimension, threshold)

            # make it not a leaf node
            self.is_leaf = False
            self.left = OnlineNode(split_count=self.split_count, depth=self.depth+1)
            self.right = OnlineNode(split_count=self.split_count, depth=self.depth+1)

            # split the data into the children nodes
            left_indices = self.rule.evaluate(x)
            right_indices = np.logical_not(left_indices)
            self.left.size = np.sum(left_indices)
            self.right.size = np.sum(right_indices)
        else:
            self.content = np.concatenate([self.content, x.reshape((1, -1))], axis=0)


class OnlineIsolationTree(Tree):
    """
    An isolation tree as proposed by liu
    """
    def __init__(self, max_depth=25, split_count=2):
        """
        :param max_depth: maximum depth a tree can grow to
        """
        super(OnlineIsolationTree, self).__init__()
        self.count = 0
        self.max_depth = max_depth
        self.root = OnlineNode(split_count=split_count)
        self.split_count = 2

    def update(self, x):
        """
        grow the tree from input data iteratively
        :param x: input data
        """
        self.count += x.shape[0]
        self.traverse(x).split(x, max_depth=self.max_depth)

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
            elif size == 2:
                return 1
            else:
                return 0
        return leaf_node.depth + remaining_path_length(leaf_node.size)


class OnlineIsolationForest:
    """
    An ensemble of Liu Trees
    """
    def __init__(self, tree_count=1, subsample_size=255, max_depth=25, split_count=2):
        """
        An ensemble of Liu Isolation Trees
        :param tree_count: how many trees to grow
        :param subsample_size: how large the subsample size should be for each
        :param max_depth: the maximum depth any tree can grow to
        """
        self.max_depth = max_depth
        self.trees = [OnlineIsolationTree(max_depth=self.max_depth, split_count=split_count) for _ in range(tree_count)]
        self.subsample_size = subsample_size

    def update(self, x):
        """
        fit all the trees on the data iteratively
        :param x: input data
        """
        # for tree in self.trees:
        #     # pick a random sample of the requested size and train the tree
        tree = np.random.choice(self.trees)
        tree.update(x)

    def fit(self, x):
        for xx in x:
            self.update(xx)

    def score(self, x):
        """
        score the point
        :param x: input point
        :return: isolation score
        """
        # expectation = np.mean([tree.score(x) for tree in self.trees])

        def harmonic_number(i):
            return np.log(i) + 0.5772156649
        return np.mean([np.power(2, -tree.score(x)/harmonic_number(tree.count)*2) for tree in self.trees])
