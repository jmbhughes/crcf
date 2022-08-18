import crcf.tree
import crcf.rule
import numpy as np
from pytest import fixture


@fixture
def example_tree():
    """
      builds the following tree:
                            o
                           / | split on 0.5 in dimension 0
                          o  o
                         / |   split on 0.5 in dimension 1
                        o  o
                          / |  split on 0.5 in dimension 2
                         o  o
    :return: sample tree
    :rtype: crcf.CombinationTree
    """
    tree = crcf.tree.CombinationTree()
    tree.root = crcf.tree.Node(is_leaf=False,
                               parent=None,
                               left_child=None,
                               right_child=None,
                               rule=crcf.rule.AxisAlignedRule(0, 0.5),
                               count=4,
                               bounding_box=None)

    tree.root.right_child = crcf.tree.Node(is_leaf=True,
                                           parent=tree.root,
                                           left_child=None,
                                           right_child=None,
                                           rule=None,
                                           count=1,
                                           bounding_box=None)

    tree.root.left_child = crcf.tree.Node(is_leaf=False,
                                          parent=tree.root,
                                          left_child=None,
                                          right_child=None,
                                          rule=crcf.rule.AxisAlignedRule(1, 0.5),
                                          count=3,
                                          bounding_box=None)

    tree.root.left_child.left_child = crcf.tree.Node(is_leaf=True,
                                                     parent=tree.root.left_child,
                                                     left_child=None,
                                                     right_child=None,
                                                     rule=None,
                                                     count=1,
                                                     bounding_box=None)

    tree.root.left_child.right_child = crcf.tree.Node(is_leaf=False,
                                                      parent=tree.root.left_child,
                                                      left_child=None,
                                                      right_child=None,
                                                      rule=crcf.rule.AxisAlignedRule(2, 0.5),
                                                      count=2,
                                                      bounding_box=None)

    tree.root.left_child.right_child.left_child = crcf.tree.Node(is_leaf=True,
                                                                 parent=tree.root.left_child.right_child,
                                                                 left_child=None,
                                                                 right_child=None,
                                                                 rule=None,
                                                                 count=1,
                                                                 bounding_box=None)

    tree.root.left_child.right_child.right_child = crcf.tree.Node(is_leaf=True,
                                                                  parent=tree.root.left_child.right_child,
                                                                  left_child=None,
                                                                  right_child=None,
                                                                  rule=None,
                                                                  count=1,
                                                                  bounding_box=None)
    return tree


def test_tree_remove_root(example_tree):
    """ make sure we can delete the root node of the tree"""
    root = example_tree.root
    example_tree.remove_by_node(root)
    assert example_tree.root is None
    assert len(example_tree.labels) == 0
    assert len(example_tree.nodes) == 0


def test_tree_remove_gen1(example_tree):
    """ make sure we can delete a node that is the child of the root, a generation one, or depth 1 node """
    node_to_delete = example_tree.root.left_child
    new_root = example_tree.root.right_child
    example_tree.remove_by_node(node_to_delete)
    assert example_tree.root is new_root
    assert example_tree.root.depth() == 0


def test_tree_remove_normal(example_tree):
    """ make sure we can delete a node that is deeper in the tree, i.e. depth >= 2"""
    node_to_delete = example_tree.root.left_child.right_child


def test_tree_find_sibling():
    """ make sure that the Node can find its sibling in the tree"""
    tree = crcf.tree.CombinationTree()
    tree.root = crcf.tree.Node(is_leaf=False,
                               parent=None,
                               left_child=None,
                               right_child=None,
                               rule=crcf.rule.AxisAlignedRule(0, 0.5),
                               count=4,
                               bounding_box=None)

    tree.root.right_child = crcf.tree.Node(is_leaf=True,
                                           parent=tree.root,
                                           left_child=None,
                                           right_child=None,
                                           rule=None,
                                           count=1,
                                           bounding_box=None)

    tree.root.left_child = crcf.tree.Node(is_leaf=False,
                                          parent=tree.root,
                                          left_child=None,
                                          right_child=None,
                                          rule=crcf.rule.AxisAlignedRule(1, 0.5),
                                          count=3,
                                          bounding_box=None)
    assert tree.root.right_child.sibling() is tree.root.left_child
    assert tree.root.right_child.depth() == 1
    assert tree.root.left_child.depth() == 1
    assert tree.root.depth() == 0


def test_build_empty_tree():
    """ ensuring an empty tree can be built """
    tree = crcf.tree.CombinationTree()
    assert tree.root is None
    assert tree.count() == 0


def test_build_tree_from_single_point():
    """ ensuring a tree can be built with only one data point"""
    x = np.array([[5, 6]])
    tree = crcf.tree.CombinationTree()
    tree.fit(x)
    assert tree.root.count == 1
    assert tree.root.depth() == 0
    assert tree.root.left_child is None
    assert tree.root.right_child is None
    assert tree.root.rule is None
    assert tree.root.bounding_box is not None


def test_build_tree_from_full_dataset():
    """ ensuring a full tree can be built"""
    x = np.arange(5*10).reshape((5, 10))
    tree = crcf.tree.CombinationTree()
    tree.fit(x)
    assert not tree.root.is_leaf
    assert tree.root.count == 5
    assert tree.root.left_child is not None
    assert tree.root.right_child is not None
    assert tree.root.depth() == 0
    assert tree.root.left_child.depth() == 1
    assert tree.root.right_child.depth() == 1


def test_tree_find():
    """ making sure an element can be found"""
    x = np.arange(5*10).reshape((5, 10))
    tree = crcf.tree.CombinationTree()
    tree.fit(x)
    leaf, depth = tree.find(x[0])
    assert leaf.is_leaf


def test_displacement(example_tree):
    """
    tests a simple case to make sure displacement behaves reasonably
    """

    x = np.array([0.7, 0.1, 0.1])
    leaf, depth = example_tree.find(x)
    assert leaf.is_leaf is True
    assert leaf.count == 1
    assert leaf.left_child is None
    assert leaf.right_child is None
    assert example_tree.displacement(x) == 3
    assert example_tree.displacement(np.array([0.3, 0.3, 0.1])) == 2
    assert example_tree.displacement(np.array([0.3, 0.6, 0.1])) == 1


def test_codisp(example_tree):
    """
    make sure codisp calculated without error...
    """
    # TODO: test for actual calculated values instead of >0
    x = np.array([0.7, 0.1, 0.1])
    assert example_tree.codisplacement(x) > 0
    assert example_tree.codisplacement(np.array([0.3, 0.3, 0.1])) > 0
    assert example_tree.codisplacement(np.array([0.3, 0.6, 0.1])) > 0


def test_depth(example_tree):
    """
    check that depth calculations work correctly in both the estimated and nonestimated cases
    """
    assert example_tree.depth(np.array([0.7, 0.1, 0.1]), estimated=False) == 1
    assert example_tree.depth(np.array([0.3, 0.3, 0.1]), estimated=False) == 2
    assert example_tree.depth(np.array([0.3, 0.6, 0.1]), estimated=False) == 3
    assert example_tree.depth(np.array([0.3, 0.6, 0.1]), estimated=True) == 3

    example_tree.root.left_child.right_child.right_child.count = 500
    example_tree.root.count += 499
    assert example_tree.depth(np.array([0.3, 0.6, 0.7]), estimated=False) == 3
    assert np.abs(example_tree.depth(np.array([0.3, 0.6, 0.7]), estimated=True) - 14.5) < 0.5


def test__score(example_tree):
    """
    small sanity check of scoring
    """
    assert example_tree.score(np.array([[0.7, 0.1, 0.1]]))
