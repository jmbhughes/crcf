import crcf.tree
import crcf.rule
import numpy as np


class TestCombinationTree:
    def test_remove_root(self):
        """ make sure we can delete the root node of the tree"""
        tree = self.build_test_tree()
        root = tree.root
        tree.remove_by_node(root)
        assert tree.root is None
        assert len(tree.labels) == 0
        assert len(tree.nodes) == 0

    def test_remove_gen1(self):
        """ make sure we can delete a node that is the child of the root, a generation one, or depth 1 node """
        tree = self.build_test_tree()
        node_to_delete = tree.root.left_child
        new_root = tree.root.right_child
        tree.remove_by_node(node_to_delete)
        assert tree.root is new_root
        assert tree.root.depth() == 0

    def test_remove_normal(self):
        """ make sure we can delete a node that is deeper in the tree, i.e. depth >= 2"""
        tree = self.build_test_tree()
        node_to_delete = tree.root.left_child.right_child


    def test_sibling(self):
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

    def test_build_empty(self):
        """ ensuring an empty tree can be built """
        tree = crcf.tree.CombinationTree()
        assert tree.root is None
        assert tree.count() == 0

    def test_build_single(self):
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

    def test_build_full(self):
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

    def test_find(self):
        """ making sure an element can be found"""
        x = np.arange(5*10).reshape((5, 10))
        tree = crcf.tree.CombinationTree()
        tree.fit(x)
        leaf, depth = tree.find(x[0])
        assert leaf.is_leaf

    @classmethod
    def build_test_tree(cls):
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

    def test_displacement(self):
        """
        tests a simple case to make sure displacement behaves reasonably
        """

        x = np.array([0.7, 0.1, 0.1])
        tree = TestCombinationTree.build_test_tree()
        leaf, depth = tree.find(x)
        assert leaf.is_leaf is True
        assert leaf.count == 1
        assert leaf.left_child is None
        assert leaf.right_child is None
        assert tree.displacement(x) == 3
        assert tree.displacement(np.array([0.3, 0.3, 0.1])) == 2
        assert tree.displacement(np.array([0.3, 0.6, 0.1])) == 1

    def test_codisp(self):
        """
        make sure codisp calculated without error...
        """
        # TODO: test for actual calculated values instead of >0
        x = np.array([0.7, 0.1, 0.1])
        tree = TestCombinationTree.build_test_tree()
        assert tree.codisplacement(x) > 0
        assert tree.codisplacement(np.array([0.3, 0.3, 0.1])) > 0
        assert tree.codisplacement(np.array([0.3, 0.6, 0.1])) > 0

    def test_depth(self):
        """
        check that depth calculations work correctly in both the estimated and nonestimated cases
        """
        tree = TestCombinationTree.build_test_tree()
        assert tree.depth(np.array([0.7, 0.1, 0.1]), estimated=False) == 1
        assert tree.depth(np.array([0.3, 0.3, 0.1]), estimated=False) == 2
        assert tree.depth(np.array([0.3, 0.6, 0.1]), estimated=False) == 3
        assert tree.depth(np.array([0.3, 0.6, 0.1]), estimated=True) == 3

        tree.root.left_child.right_child.right_child.count = 500
        tree.root.count += 499
        assert tree.depth(np.array([0.3, 0.6, 0.7]), estimated=False) == 3
        assert np.abs(tree.depth(np.array([0.3, 0.6, 0.7]), estimated=True) - 14.5) < 0.5

    def test__score(self):
        """
        small sanity check of scoring
        """
        tree = TestCombinationTree.build_test_tree()
        assert tree.score(np.array([[0.7, 0.1, 0.1]]))
