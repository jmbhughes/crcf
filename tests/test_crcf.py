from crcf import crcf
import numpy as np


class TestAxisAlignedRule:
    def test_init(self):
        """ make sure the rule initializes and stores the correct values"""
        rule = crcf.AxisAlignedRule(5, 0)
        assert rule.dimension == 5
        assert rule.value == 0

    def test_evaluate(self):
        """ make sure simple evaluates perform as expected"""
        rule = crcf.AxisAlignedRule(5, 0)
        x = np.array([[0, 0, 0, 0, 0, -1],
                      [0, 0, 0, 0, 0, 1]])
        results = rule.evaluate(x)
        assert results[0]
        assert not results[1]

    def test_generate_biased(self):
        """ make sure new rules are generated according to a bias"""
        # a deterministic bias check
        bounding_box = np.array([[0, 0], [0, 1]])
        rule = crcf.AxisAlignedRule.generate(bounding_box, mode="biased")
        assert rule.dimension == 1
        assert rule.value >= 0
        assert rule.value <= 1

        # a rough attempt at testing the bias rate
        bounding_box = np.array([[0, 2], [0, 1]])
        num_rules = 1000
        rules = [crcf.AxisAlignedRule.generate(bounding_box, mode='biased') for _ in range(num_rules)]
        dimension_zeros = [1 if rule.dimension==0 else 0 for rule in rules]
        assert np.sum(dimension_zeros) > num_rules * 0.5
        assert np.sum(dimension_zeros) < num_rules * 0.8

    def test_generate_uniform(self):
        """ make sure we can generate a rule at uniform"""
        # a deterministic check
        bounding_box = np.array([[0, 0], [0, 1]])
        rule = crcf.AxisAlignedRule.generate(bounding_box, mode="uniform")
        assert rule.dimension == 0 or rule.dimension == 1
        assert rule.value >= 0
        assert rule.value <= 1


class TestNonAxisAlignedRule:
    def test_init(self):
        """ make sure the rule initializes and stores the correct values"""
        rule = crcf.NonAxisAlignedRule(np.array([1, 2, 3]),
                                       np.array([1, 2, 4]))
        assert np.all(rule.point == np.array([1, 2, 4]))
        assert np.all(rule.normal == np.array([1, 2, 3]))
        assert rule.normal.dot(rule.point) == rule.offset

    def test_evaluate(self):
        """ make sure we can evaluate points"""
        normal = np.array([1, 2, 3])
        point = np.array([1, 2, 3])
        offset = normal.dot(point)
        rule = crcf.NonAxisAlignedRule(normal, point)
        points = np.array([[1, 2, 3],
                           [0, 0, 0],
                           [5, 4, 6]])
        assert np.all(rule.evaluate(points) == (np.inner(normal, points) < offset))

    def test_generate_biased(self):
        """ make sure new rules are generated according to a bias"""
        # TODO: write
        pass

    def test_generate_uniform(self):
        """ make sure we can generate a rule at uniform"""
        # a deterministic check
        # TODO: write
        pass


class TestCombinationTree:
    def test_build_empty(self):
        tree = crcf.CombinationTree()
        assert tree.is_root()
        assert tree.count == 0
        assert tree.left_child is None
        assert tree.right_child is None
        assert tree.depth_ == 0
        assert tree.rule is None
        assert tree.bounding_box is None

    def test_build_full(self):
        x = np.arange(5*10).reshape((5, 10))
        tree = crcf.CombinationTree(x=x)
        assert tree.is_root()
        assert tree.count == 5
        assert tree.left_child is not None
        assert tree.right_child is not None
        assert tree.depth_ == 0
        assert tree.left_child.depth_ == 1
        assert tree.right_child.depth_ == 1

    def test_find(self):
        x = np.arange(5*10).reshape((5, 10))
        tree = crcf.CombinationTree(x=x)
        leaf = tree.find(x[0])
        assert leaf.is_leaf_

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
        tree = crcf.CombinationTree()
        tree.count = 4
        tree.rule = crcf.AxisAlignedRule(0, 0.5)
        tree.is_leaf_ = False

        tree.right_child = crcf.CombinationTree()
        tree.right_child.count = 1
        tree.right_child.is_leaf_ = True
        tree.right_child.depth_ = 1
        tree.right_child.parent = tree

        tree.left_child = crcf.CombinationTree()
        tree.left_child.count = 3
        tree.left_child.depth_ = 1
        tree.left_child.is_leaf_ = False
        tree.left_child.rule = crcf.AxisAlignedRule(1, 0.5)
        tree.left_child.parent = tree

        tree.left_child.left_child = crcf.CombinationTree()
        tree.left_child.left_child.count = 1
        tree.left_child.left_child.depth_ = 2
        tree.left_child.left_child.is_leaf_ = True
        tree.left_child.left_child.parent = tree.left_child

        tree.left_child.right_child = crcf.CombinationTree()
        tree.left_child.right_child.count = 2
        tree.left_child.right_child.depth_ = 2
        tree.left_child.right_child.is_leaf = False
        tree.left_child.right_child.rule = crcf.AxisAlignedRule(2, 0.5)
        tree.left_child.right_child.parent = tree.left_child

        tree.left_child.right_child.left_child = crcf.CombinationTree()
        tree.left_child.right_child.left_child.count = 1
        tree.left_child.right_child.left_child.depth_ = 3
        tree.left_child.right_child.left_child.is_leaf = True
        tree.left_child.right_child.left_child.parent = tree.left_child.right_child

        tree.left_child.right_child.right_child = crcf.CombinationTree()
        tree.left_child.right_child.right_child.count = 1
        tree.left_child.right_child.right_child.depth_ = 3
        tree.left_child.right_child.right_child.is_leaf = True
        tree.left_child.right_child.right_child.parent = tree.left_child.right_child
        return tree

    def test_displacement(self):
        """
        tests a simple case to make sure displacement behaves reasonably
        """

        x = np.array([0.7, 0.1, 0.1])
        tree = TestCombinationTree.build_test_tree()
        leaf = tree.find(x)
        assert leaf.is_leaf_ == True
        assert leaf.count == 1
        assert leaf.left_child is None
        assert leaf.right_child is None
        assert tree.displacement(x) == 3
        assert tree.displacement(np.array([0.3, 0.3, 0.1])) == 2
        assert tree.displacement(np.array([0.3, 0.6, 0.1])) == 1
