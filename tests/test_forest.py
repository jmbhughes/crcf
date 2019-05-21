import crcf.forest
import numpy as np


class TestCombinationForest:
    def test_build_empty(self):
        """ ensuring an empty forest can be built """
        count = 1000
        forest = crcf.forest.CombinationForest(num_trees=count)
        assert len(forest.trees) == count
        for tree in forest.trees:
            assert tree.root is None
            assert tree.count() == 0

    def test_build_single(self):
        """ ensuring a forest can be built with only one data point"""
        count = 1000
        x = np.array([[5, 6]])
        forest = crcf.forest.CombinationForest(num_trees=count)
        forest.fit(x)
        assert len(forest.trees) == count
        for tree in forest.trees:
            assert tree.root.count == 1
            assert tree.root.depth() == 0
            assert tree.root.left_child is None
            assert tree.root.right_child is None
            assert tree.root.rule is None
            assert tree.root.bounding_box is not None

    def test_build_full(self):
        """ ensuring a forest can be built from a full data set"""
        count = 1000
        x = np.arange(5*10).reshape((5, 10))
        forest = crcf.forest.CombinationForest(num_trees=count)
        forest.fit(x)
        assert len(forest.trees) == count
        for tree in forest.trees:
            assert not tree.root.is_leaf
            assert tree.root.count == 5
            assert tree.root.left_child is not None
            assert tree.root.right_child is not None
            assert tree.root.depth() == 0
            assert tree.root.left_child.depth() == 1
            assert tree.root.right_child.depth() == 1

    def test_displacement(self):
        # TODO: make a better test than just asserting they are positive
        count = 1000
        x = np.arange(5 * 10).reshape((5, 10))
        forest = crcf.forest.CombinationForest(num_trees=count)
        forest.fit(x)
        for xx in x:
            assert forest.displacement(xx) > 0

    def test_codisplacement(self):
        # TODO: make a better test than just asserting they are positive
        count = 1000
        x = np.arange(5 * 10).reshape((5, 10))
        forest = crcf.forest.CombinationForest(num_trees=count)
        forest.fit(x)
        for xx in x:
            assert forest.codisplacement(xx) > 0

    def test_depth(self):
        # TODO: make a better test than just asserting they are positive
        count = 1000
        x = np.arange(5*10).reshape((5, 10))
        forest = crcf.forest.CombinationForest(num_trees=count)
        forest.fit(x)
        for xx in x:
            assert forest.depth(xx) > 0

    def test_score(self):
        # TODO: make a better test than just asserting they are positive
        count = 1000
        x = np.arange(5 * 10).reshape((5, 10))
        forest = crcf.forest.CombinationForest(num_trees=count)
        forest.fit(x)
        assert np.all(forest.score(x) > 0)
