from __future__ import annotations
from typing import Optional, Type, Tuple
import numpy as np
from graphviz import Digraph
from .rule import Rule, AxisAlignedRule, NonAxisAlignedRule
from dataclasses import dataclass


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
        """Get sibling node

        Returns
        -------
        Node or None
            the sibling node, returns None if there is no sibling or the parent is not defined
        """
        if self.parent is None:
            return None
        else:
            return self.parent.left_child if self.parent.left_child is not self else self.parent.right_child

    def depth(self) -> int:
        """Get depth of node

        Returns
        -------
        int
            the depth of the node in the tree
        """
        depth = 0
        current = self
        while current.parent is not None:
            current = current.parent
            depth += 1
        return depth

    def __str__(self) -> str:
        return str(self.is_leaf) + str(self.rule) + str(self.bounding_box)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other: Node) -> bool:
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
        """Initialize a combination tree.

        Parameters
        ----------
        depth_limit : int or None
            the maximum depth the tree can grow to, None indicates no depth limiting
        rule_kind : Rule
            whether to use axis aligned or non-axis aligned rules
        rule_mode : str
            the mode by which the rules are generated, see the rule for documentation
        """
        # properties of the tree
        self.depth_limit = depth_limit
        self.rule_kind = rule_kind
        self.rule_mode = rule_mode
        self.nodes = dict()  # mapping to find nodes by label
        self.labels = dict()  # a mapping to find labels by node
        self.root = None  # tree has not been grown at all yet

    def fit(self, x: np.ndarray) -> None:
        """Fit a tree from scratch

        Parameters
        ----------
        x : np.ndarray
            the data points to fit  e.g. x = [[1, 2, 3], [3, 4, 5]] is a list of two 3-d points

        Notes
        -----
        If the tree is already fitted this will overwrite completely.

        Returns
        -------
        None
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
        """Helper function that recursively grows the tree

        Parameters
        ----------
        node : Node
            the current node (the root node of whatever subtree you're growing)
        x : np.ndarray
            the points you're fitting
        depth : int
            current depth of the node

        Returns
        -------
        None
        """
        # update the rule and local properties in regard to x
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
        """Total count of data points in the tree

        Returns
        -------
        number of data points in the tree
        """
        return 0 if self.root is None else self.root.count

    def show(self, path: Optional[str] = None) -> None:
        """Draw the tree

        Parameters
        ----------
        path : str or None
            the path to save the image to

        Returns
        -------
        None
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

    def find(self, x: np.ndarray) -> Tuple[Node, int]:
        """Find the correct leaf node for an input point x

        Parameters
        ----------
        x : np.ndarray
             a single data point

        Returns
        -------
        Tuple[Node, int]
            the leaf node that x would belong to, and the depth of that node
        """
        node: Node = self.root
        depth: int = 0
        while not node.is_leaf:
            node = node.left_child if node.rule.evaluate(np.array([x])) else node.right_child
            depth += 1
        return node, depth

    def get_node(self, label: str) -> Node:
        """Find a node using its label

        Parameters
        ----------
        label : str
            label of a node

        Returns
        -------
        Node
            the node with the given label
        """
        try:
            node = self.labels[label]
        except KeyError:
            raise KeyError("There is no node labeled {}".format(label))
        else:
            return node

    def save(self, path: str) -> None:
        """Save the tree to a file for later opening

        Parameters
        ----------
        path : str
            the path/name of the new file, should have '.pkl' extension

        Returns
        -------
        None
        """
        raise NotImplementedError("Saving was removed for a pickle vulnerability.")

    @classmethod
    def load(cls, path: str) -> CombinationTree:
        """Load a tree from a file

        Parameters
        ----------
        path : str
            the location of the file

        Returns
        -------
        CombinationTree
            the loaded tree
        """
        raise NotImplementedError("Loading was removed for a pickle vulnerability.")

    def depth(self, x: np.ndarray,
              estimated: bool = True) -> float:
        """Determine the depth of x in the tree

        Parameters
        ----------
        x : np.ndarray
            a data point
        estimated : bool
            if True will use the counts at a leaf node to estimate how far down the tree the point
                would be if it had been grown completely

        Returns
        -------
        int
            the depth of the point
        """
        def harmonic(n: int) -> float:
            """n-th harmonic number

            Parameters
            ----------
            n : int
                index of the harmonic number

            Returns
            -------
            float
                the n-th harmonic number
            """
            return np.log(n) + np.euler_gamma

        def expected_length(n: int) -> float:
            """Expected depth in the tree given count in the leaf node

            Parameters
            ----------
            n : int
                the number of data points in a leaf node

            Returns
            -------
            float
                the expected average length had the tree continued to grow
            """
            return 2 * harmonic(n-1) - (2*(n-1) / n) if n > 1 else 0

        leaf, depth = self.find(x)
        extension = expected_length(leaf.count) if estimated else 0
        return depth + extension

    def displacement(self, x: np.ndarray) -> float:
        """Displacement of a point x in the tree

        Parameters
        ----------
        x : np.ndarray
            a single data point

        Returns
        -------
        float
            the "surprise" or displacement induced by including x in the tree
        """
        leaf, depth = self.find(x)
        sibling = leaf.parent.left_child if leaf.parent.left_child is not leaf else leaf.parent.right_child
        return sibling.count

    def codisplacement(self, x: np.ndarray) -> float:
        """Get codisplacement of a data point

        Parameters
        ----------
        x : np.ndarray
            a data sample

        Notes
        -----
        Codisplacement allows for colluders in the displacement per RRCF paper [Guha+2016]

        Returns
        -------
        float
            codisplacement of a data point

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

    def score(self, x: np.ndarray,
              normalize=False,
              use_codisplacement=False,
              use_depth_estimation=False,
              depth_weight=1.0,
              disp_weight=0.0) -> np.ndarray:
        """ Calculate the anomaly score

        Parameters
        ----------
        x : np.ndarray
            a set of points to score
        normalize : bool
            whether to attempt to normalize the score, default=False
        use_codisplacement : bool
            if True uses codisplacement, if False uses displacement; default=False
        use_depth_estimation : bool
            whether to use the absolute depth or the estimated depths from count, see depth(),
                  default=False
        depth_weight : float
            the weight for depth in the score, depth_weight * depth + disp_weight * [co]disp, default=1
        disp_weight : float
            see depth_weight

        Returns
        -------
        np.ndarray
            scores of the form depth_weight * depth + disp_weight * [co]disp for each point in the dataset
        """
        disp = np.array([self.codisplacement(xx) for xx in x]) if use_codisplacement \
            else np.array([self.displacement(xx) for xx in x])
        depths = np.array([self.depth(xx, estimated=use_depth_estimation) for xx in x])
        if normalize:
            disp = disp / (self.root.count - 1)
            depths = depths / self.root.count
        return depth_weight * 1/depths + disp_weight * disp

    def remove_by_node(self, node: Node) -> None:
        """Remove the specified node using RRCF rules

        Parameters
        ----------
        node : Node
            node to remove

        Notes
        -----
        below we delete the R node:
                   |               |
                   G               G
                  / |             / |
                 o   P    TO     o  S
                    / |
                   S   R

        Returns
        -------
        None

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
        """Remove a node that is deeper in the tree, i.e. it is not a child of the root

        Parameters
        ----------
        node : Node
            node that is not the root of a tree

        Returns
        -------
        None
        """
        self._clear_labels(node)

        parent = node.parent
        grandparent = parent.parent
        sibling = node.sibling()
        sibling.parent = grandparent

        # TODO: update the bounding boxes

    def _remove_gen1(self, node: Node) -> None:
        """Remove a node that is a child of the root, meaning that the sibling becomes the new root

        Parameters
        ----------
        node : Node
            the node to remove

        Returns
        -------
        None
        """
        self._clear_labels(node)
        sibling = node.sibling()
        self.root = sibling
        self.root.parent = None

    def _clear_labels(self, node: Node) -> None:
        """Clear the labels of all nodes in the subtree rooted at node


        Parameters
        ----------
        node : Node
            the root of the subtree to clear

        Returns
        -------
        None
        """
        nodes = [node]
        while nodes:
            current = nodes.pop()

            # if it's not a leaf it has children so add them to clear next
            if not current.is_leaf:
                nodes.append(current.left_child)
                nodes.append(current.right_child)

            # if a label existed for it
            if current in self.labels:
                label = self.labels[current]
                del self.nodes[label]
                del self.labels[current]

    def remove_by_label(self, label: str) -> None:
        """Remove a node by its label

        Parameters
        ----------
        label : str
            label of a node

        Raises
        ------
        KeyError
            if a node does not exist with that label

        Returns
        -------
        None
        """
        node = self.get_node(label)
        self.remove_by_node(node)

    def remove_by_value(self, x: np.ndarray) -> None:
        """Removes a node associated with the value x

        Parameters
        ----------
        x : np.ndarray
            a data point

        Returns
        -------
        None
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

    def name(self, node: Node, label: str) -> None:
        """Assign the specified node the passed label.

        Parameters
        ----------
        node : Node
            which node to name
        label : str
            label to give the node

        Raises
        ------
        KeyError
            If the label is already used, it raises a KeyError.

        Returns
        -------
        None
        """
        if label in self.labels:
            raise KeyError("A node already exists labeled {}".format(label))
        else:
            self.nodes[label] = node
            self.labels[node] = label


class IsolationTree(CombinationTree):
    def __init__(self, depth_limit=None):
        """Initialize an isolation tree

        Parameters
        ----------
        depth_limit : int
            the maximum depth a tree can grow to, None indicates there is no limit
        """
        super().__init__(depth_limit=depth_limit)

    def codisplacement(self, x):
        raise NotImplementedError("Isolation trees do not have codisplacement. " +
                                  "See the CombinationTree or the RobustRandomCutTree instead.")

    def displacement(self, x):
        raise NotImplementedError("Isolation trees do not have displacement. " +
                                  "See the CombinationTree or the RobustRandomCutTree instead.")

    def score(self, x: np.ndarray, estimated: bool = False) -> np.ndarray:
        """Calculate the anomaly score

        Parameters
        ----------
        x : np.ndarray
            a set of points to score
        estimated : bool
            if True will use the counts at a leaf node to estimate how far down the tree the point
                would be if it had been grown completely

        Returns
        -------
        np.ndarray
            the scores!
        """
        depths = np.array([self.depth(xx, estimated=estimated) for xx in x])
        return depths


class RobustRandomCutTree(CombinationTree):
    def __init__(self, depth_limit: Optional[int] = None) -> None:
        super().__init__(depth_limit=depth_limit, rule_kind=AxisAlignedRule, rule_mode="biased")

    def depth(self, x: np.ndarray,
              estimated: bool = True) -> int:
        raise NotImplementedError("RobustRandomCutTrees do not have depth. " +
                                  "See the CombinationTree or IsolationTree instead.")

    def score(self, x: np.ndarray, **kwargs) -> np.ndarray:
        disp = np.array([self.codisplacement(xx) for xx in x])
        return disp


class ExtendedIsolationTree(CombinationTree):
    def __init__(self, depth_limit=None):
        """Initialize an extended isolation tree

        Parameters
        ----------
        depth_limit : int
            the maximum depth the tree can grow to, a depth limit of None indicates there is no limit
        """
        super().__init__(depth_limit=depth_limit, rule_kind=NonAxisAlignedRule)

    def codisplacement(self, x):
        raise NotImplementedError("Isolation trees do not have codisplacement. " +
                                  "See the CombinationTree or the RobustRandomCutTree instead.")

    def displacement(self, x):
        raise NotImplementedError("Isolation trees do not have displacement. " +
                                  "See the CombinationTree or the RobustRandomCutTree instead.")

    def score(self, x: np.ndarray, estimated: bool = False) -> np.ndarray:
        """Calculate the anomaly score

        Parameters
        ----------
        x : np.ndarray
            a set of points to score
        estimated : bool
            if True will use the counts at a leaf node to estimate how far down the tree the point
                would be if it had been grown completely

        Returns
        -------
        np.ndarray
            the scores!
        """
        depths = np.array([self.depth(xx, estimated=estimated) for xx in x])
        return depths
