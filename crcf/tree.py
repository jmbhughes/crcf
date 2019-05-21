from __future__ import annotations
from typing import Optional, Type
import numpy as np
from graphviz import Digraph
from .rule import Rule, AxisAlignedRule, NonAxisAlignedRule
from dataclasses import dataclass
import pickle


@dataclass(repr=False)
class Node:
    """
    A container object for nodes in a tree. 
    """
    __slots__ = ['is_leaf', 'parent', 'left_child', 'right_child', 'rule', 'count', 'bounding_box']
    is_leaf: bool
    parent: Optional[Node]
    left_child: Optional[Node]
    right_child: Optional[Node]
    rule: Optional[Rule]
    count: int
    bounding_box: Optional[np.ndarray]

    def sibling(self) -> Optional[Node]:
        """
        :return: the sibling node, returns None if there is no sibling or the parent is not defined
        """
        if self.parent is None:
            return None
        else:
            return self.parent.left_child if self.parent.left_child is not self else self.parent.right_child

    def depth(self) -> int:
        """
        :return: the depth of the node in the tree
        """
        depth = 0
        current = self
        while current.parent is not None:
            current = current.parent
            depth += 1
        return depth

    def __str__(self):
        return str(self.is_leaf) + str(self.rule) + str(self.bounding_box)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other: Node):
        return self.is_leaf == other.is_leaf \
               and self.left_child == other.left_child \
               and self.right_child == other.right_child \
               and self.rule == other.rule \
               and self.count == other.count \
               and np.all(self.bounding_box == other.bounding_box)


class CombinationTree:
    """
    Combination Robust Cut Trees are a generalization of robust random cut trees of Guha et al. (2016)
    and isolation trees of Liu et al. (2010). The parameters can be set to get the exact formulation of each or
    a tree that is an interpolation of the two modes:
    """
    def __init__(self, depth_limit: Optional[int] = None,
                 rule_kind: Type[Rule] = AxisAlignedRule,
                 rule_mode: str = "uniform") -> None:

        """
        Initialize a Combination Tree
        :param depth_limit: the maximum depth the tree can grow to,
                            a depth limit of None indicates there is no limit
        :param rule_kind: whether to use axis aligned or non-axis aligned rules
        :param rule_mode: the mode by which the rules are generated, see the rule for documentation
        """
        # properties of the tree
        self.depth_limit = depth_limit
        self.rule_kind = rule_kind
        self.rule_mode = rule_mode
        self.nodes = dict()  # mapping to find nodes by label
        self.labels = dict()  # a mapping to find labels by node
        self.root = None  # tree has not been grown at all yet

    def fit(self, x: np.ndarray) -> None:
        """
        Fit a tree from scratch.
        Note: that if the tree is already fitted this will overwrite completely.
        :param x: the data points to fit  e.g. x = [[1, 2, 3], [3, 4, 5]] is a list of two 3-d points
        """
        self.root = Node(is_leaf=True,
                         parent=None,
                         left_child=None,
                         right_child=None,
                         rule=None,
                         count=0,
                         bounding_box=None)
        self._fit(self.root, x)

    def _fit(self, node: Node, x: np.ndarray, depth: int = 0) -> None:
        """
        Helper function for fit that recursively grows the tree
        :param node: the current node (the root node of whatever subtree you're growing)
        :param x: the points you're fitting
        """
        # update the rule and local properties with regards to x
        node.count = x.shape[0]
        node.bounding_box = np.array([[np.nanmin(x[:, i]), np.nanmax(x[:, i])] for i in range(x.shape[1])])
        # if the depth limit has been hit, the tree can no longer grow
        # have to check the parent depth since the current node is not set yet
        if self.depth_limit and depth >= self.depth_limit - 1:
            node.is_leaf = True
        elif node.count == 1:  # will be a leaf node
            node.is_leaf = True
        else:  # will not be a leaf node since it can split
            node.is_leaf = False
            node.rule = self.rule_kind.generate(node.bounding_box, mode=self.rule_mode)

            # create the children nodes and create relationships
            evaluation = node.rule.evaluate(x)  # whether the points go to the left subtree

            # left child setup
            left_child = Node(is_leaf=True,
                              parent=node,
                              left_child=None,
                              right_child=None,
                              rule=None,
                              count=0,
                              bounding_box=None)
            node.left_child = left_child
            self._fit(left_child, x[evaluation], depth=depth+1)

            # right child setup
            right_child = Node(is_leaf=True,
                               parent=node,
                               left_child=None,
                               right_child=None,
                               rule=None,
                               count=0,
                               bounding_box=None)
            node.right_child = right_child
            self._fit(right_child, x[np.logical_not(evaluation)], depth=depth+1)

    def count(self) -> int:
        """
        :return: number of data points in the tree
        """
        return 0 if self.root is None else self.root.count

    def show(self, path: Optional[str] = None):
        """
        draw the tree to a file
        :param path: the path to save image to
        :return: a drawn tree
        """
        dot = Digraph()
        queue = [(0, self.root)]  # start at the root with no parent
        index = 1

        # for all the nodes
        while queue:
            parent_index, node = queue.pop()
            if node is not None:  # it isn't the placeholder child of a leaf node

                # determine the label and color from its leaf status
                if node.is_leaf:
                    node_label = "{}".format(node.count)
                    node_attr = {"fillcolor": "gray", "style": "filled"}
                else:
                    node_label = str(node.rule)
                    node_attr = dict()

                # draw the node
                dot.node(str(index), node_label, _attributes=node_attr)

                if parent_index != 0:  # it isn't the root node
                    dot.edge(str(parent_index), str(index))  # draw the edge connecting to the parent

                # recurse on the children
                queue.append((index, node.right_child))
                queue.append((index, node.left_child))
                index += 1

        # show the graph
        dot.view(filename=path)

    def find(self, x: np.ndarray) -> (Node, int):
        """
        Find the correct leaf node for an input point x
        :param x: a data point
        :return: the leaf node x would belong with, depth of that node
        """
        node = self.root
        depth = 0
        while not node.is_leaf:
            node = node.left_child if node.rule.evaluate(np.array([x])) else node.right_child
            depth += 1
        return node, depth

    def get_node(self, label: str) -> Node:
        """
        Find a node using its label
        :param label: label of node
        :return: the node with that label
        """
        try:
            node = self.labels[label]
        except KeyError:
            raise KeyError("There is no node labeled {}".format(label))
        else:
            return node

    def save(self, path: str) -> None:
        """
        Save the tree to a file for later opening
        :param path: the path/name of the new file, should have '.pkl' extension
        """
        # TODO: change this to a more robust save method
        if not path.lower().endswith('.pkl'):
            raise RuntimeError("Save paths should end with .pkl")
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> CombinationTree:
        """
        Load a tree from a file
        :param path: location of the file
        :return: the loaded tree
        """
        with open(path, 'rb') as f:
            tree = pickle.load(f)
        return tree

    def depth(self, x: np.ndarray,
              estimated: bool = True) -> float:
        """
        Determine the depth of where x is in the tree
        :param x: a point
        :param estimated: if True will use the counts at a leaf node
            to estimate how far down the tree the point would be if it had been grown completely
        :return: the depth of the point
        """
        def harmonic(n: int) -> float:
            """
            :param n: index
            :return: the nth harmonic number
            """
            return np.log(n) + np.euler_gamma

        def expected_length(n: int) -> float:
            """
            :param n: count remaining in leaf node
            :return: the expected average length had the tree continued to grow
            """
            return 2 * harmonic(n-1) - (2*(n-1) / n) if n > 1 else 0

        leaf, depth = self.find(x)
        extension = expected_length(leaf.count) if estimated else 0
        return depth + extension

    def displacement(self, x: np.ndarray) -> float:
        """
        The displacement of a point x in the tree
        :param x: a data sample
        :return: the "surprise" or displacement induced by including x in the tree
        """
        leaf, depth = self.find(x)
        sibling = leaf.parent.left_child if leaf.parent.left_child is not leaf else leaf.parent.right_child
        return sibling.count

    def codisplacement(self, x: np.ndarray) -> float:
        """
        Codisplacement allows for colluders in the displacement per RRCF paper [Guha+2016]
        :param x: a data sample
        :return: the collusive displacement induced by including x in the tree
        """
        node, depth = self.find(x)
        best_codisp = 0

        # work upward from the leaf node to the parent considering all paths along
        while node.parent is not None:
            sibling = node.parent.left_child if node.parent.left_child is not node else node.parent.right_child
            collusive_size, sibling_size = node.count, sibling.count
            this_codisp = sibling_size/collusive_size

            # compare to the best_codisp and update if higher
            if this_codisp > best_codisp:
                best_codisp = this_codisp
            node = node.parent
        return best_codisp

    def score(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Calculate the anomaly score
        :param x: a set of points to score

        :Keyword Arguments:
            *  use_codisplacement: if True uses codisplacement, if false uses displacement, default=True
            *  estimated: whether to use the absolute depth or the estimated depths from count, see depth(),
                          default=False
            *  alpha: the combination value for alpha * depth + (1-beta) * [co]disp, default=1
            *  beta: see alpha, default=0
            *  normalized: whether or not to attempt to normalize the score, default=False
        :return: the anomaly score
        """
        params = {"use_codisplacement": False,
                  "estimated": False,
                  "alpha": 1,
                  "beta": 0,
                  "normalized": False}
        for k, v in kwargs.items():
            try:
                params[k] = v
            except KeyError:
                raise RuntimeWarning("{} is not a defined parameter. See the docstring.".format(k))
        disp = np.array([self.codisplacement(xx) for xx in x]) if params['use_codisplacement'] \
            else np.array([self.displacement(xx) for xx in x])
        depths = np.array([self.depth(xx, estimated=params['estimated']) for xx in x])
        if params['normalized']:
            disp = disp / (self.root.count - 1)
            depths = depths / self.root.count
        return params['alpha'] * depths + params['beta'] * disp

    def remove_by_node(self, node: Node) -> None:
        """
        Remove the subtree at this node using the RRCF rules.
        below we delete the R node:
                   |               |
                   G               G
                  / |             / |
                 o   P    TO     o  S
                    / |
                   S   R
        :param node: node at root of tree or single node to remove
        """
        node_to_delete = node
        parent = node_to_delete.parent

        if parent is None:  # we are deleting the root of the tree
            # so reset everything
            del self.root
            self.root = None
            self.labels = dict()
            self.nodes = dict()
        elif parent.parent is None:  # the parent is the root of the tree
            self._remove_gen1(node)
        else:  # this is the normal case that occurs, the node is somewhere with depth 2+
            self._remove_normal(node)

    def _remove_normal(self, node: Node) -> None:
        """
        Remove a node that is deeper in the tree, i.e. it is not a child of the root
        :param node: the root of the subtree to delete
        """
        self._clear_labels(node)

        parent = node.parent
        grandparent = parent.parent
        sibling = node.sibling()
        sibling.parent = grandparent

        # TODO: update the bounding boxes

    def _remove_gen1(self, node: Node) -> None:
        """
        Remove a node that is a child of the root, meaning that the sibling becomes the new root
        :param node: root of the subtree to remove
        """
        self._clear_labels(node)
        sibling = node.sibling()
        self.root = sibling
        self.root.parent = None

    def _clear_labels(self, node: Node) -> None:
        """
        Clear the labels of all nodes in the subtree rooted at node
        :param node: the root of the subtree to clear
        """
        nodes = [node]
        while nodes:
            current = nodes.pop()

            # if its not a leaf it has children so add them to clear next
            if not current.is_leaf:
                nodes.append(current.left_child)
                nodes.append(current.right_child)

            # if a label existed for it
            if current in self.labels:
                label = self.labels[current]
                del self.nodes[label]
                del self.labels[current]

    def remove_by_label(self, label: str) -> None:
        """
        Remove a node from its label
        Raises a KeyError if a node does not exist with that label
        :param label: the label of the node
        """
        node = self.get_node(label)
        self.remove_by_node(node)

    def remove_by_value(self, x: np.ndarray) -> None:
        """
        Removes a node based associated with the value x
        :param x: the value of node
        """
        node, depth = self.find(x)
        self.remove_by_node(node)
    #
    # def insert(self, x: np.ndarray, label: Optional[str] = None) -> None:
    #     """
    #     Using the InsertPoint algorithm of RRCF insert a point so it conserves the distribution of trees
    #     :param x: a data point to insert
    #     :param label: an optional label to assign it, ignores the label if it's None
    #     """
    #     pass

    def name(self, node: Node, label: str):
        """
        Assign the specified node the passed label. If the label is already used raises a KeyError.
        :param node: node to name
        :param label: label to give node
        """
        if label in self.labels:
            raise KeyError("A node already exists labeled {}".format(label))
        else:
            self.nodes[label] = node
            self.labels[node] = label


class IsolationTree(CombinationTree):
    def __init__(self, depth_limit=None):
        """
        Initialize an isolation tree
        :param depth_limit: the maximum depth the tree can grow to,
                            a depth limit of None indicates there is no limit        """
        super().__init__(depth_limit=depth_limit)

    def codisplacement(self, x):
        raise NotImplementedError("Isolation trees do not have codisplacement. " +
                                  "See the CombinationTree or the RobustRandomCutTree instead.")

    def displacement(self, x):
        raise NotImplementedError("Isolation trees do not have displacement. " +
                                  "See the CombinationTree or the RobustRandomCutTree instead.")

    def score(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Calculate the anomaly score
        :param x: a set of points to score

        :Keyword Arguments:
            *  estimated: whether to use the absolute depth or the estimated depths from count, see depth(),
                          default=False
        :return: the anomaly score
        """
        params = {"estimated": False}
        for k, v in kwargs.items():
            try:
                params[k] = v
            except KeyError:
                raise RuntimeWarning("{} is not a defined parameter. See the docstring.".format(k))

        depths = np.array([self.depth(xx, estimated=params['estimated']) for xx in x])
        return depths


class RobustRandomCutTree(CombinationTree):
    def __init__(self,  depth_limit: Optional[int] = None) -> None:
        super().__init__(depth_limit=depth_limit, rule_kind=AxisAlignedRule, rule_mode="biased")

    def depth(self, x: np.ndarray,
              estimated: bool = True) -> int:
        raise NotImplementedError("RobustRandomCutTrees do not have depth. " +
                                  "See the CombinationTree or IsolationTree instead.")

    def score(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Calculate the anomaly score
        :param x: a set of points to score
        :return: the anomaly score
        """
        params = {}
        for k, v in kwargs.items():
            try:
                params[k] = v
            except KeyError:
                raise RuntimeWarning("{} is not a defined parameter. See the docstring.".format(k))
        disp = np.array([self.codisplacement(xx) for xx in x])
        return disp


class ExtendedIsolationTree(CombinationTree):
    def __init__(self, depth_limit=None):
        """
        Initialize an extended isolation tree
        :param depth_limit: the maximum depth the tree can grow to,
                            a depth limit of None indicates there is no limit"""
        super().__init__(depth_limit=depth_limit, rule_kind=NonAxisAlignedRule)

    def codisplacement(self, x):
        raise NotImplementedError("Isolation trees do not have codisplacement. " +
                                  "See the CombinationTree or the RobustRandomCutTree instead.")

    def displacement(self, x):
        raise NotImplementedError("Isolation trees do not have displacement. " +
                                  "See the CombinationTree or the RobustRandomCutTree instead.")

    def score(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Calculate the anomaly score
        :param x: a set of points to score

        :Keyword Arguments:
            *  estimated: whether to use the absolute depth or the estimated depths from count, see depth(),
                          default=False
        :return: the anomaly score
        """
        params = {"estimated": False}
        for k, v in kwargs.items():
            try:
                params[k] = v
            except KeyError:
                raise RuntimeWarning("{} is not a defined parameter. See the docstring.".format(k))

        depths = np.array([self.depth(xx, estimated=params['estimated']) for xx in x])
        return depths
