from __future__ import annotations
from typing import Optional, Type
import crcf.tree
import crcf.rule
import numpy as np


class CombinationForest:
    def __init__(self,
                 num_trees: int = 100,
                 tree_properties: Optional[dict] = None) -> None:
        """Creates a Combination Forest

        Parameters
        ----------
        num_trees : int
            number of trees
        tree_properties : Optional[dict[str, Any]]
            the properties of each tree
        """
        self.tree_properties = tree_properties if tree_properties is not None else dict()
        self.trees = [crcf.tree.CombinationTree(**self.tree_properties) for _ in range(num_trees)]

    def fit(self, x: np.ndarray) -> None:
        """Fit the forest. This overwrites any previous tree training.

        Parameters
        ----------
        x : np.ndarray
            data to fit on

        Returns
        -------
        None
        """
        for tree in self.trees:
            tree.fit(x)

    def save(self, path: str) -> None:
        """Save the forest.

        Parameters
        ----------
        path : str
            where to save the tree, must end in '.pkl' as forests are currently saved as pickle files

        Returns
        -------
        None
        """
        raise NotImplementedError("Saving was removed for a pickle vulnerability.")


    @classmethod
    def load(cls, path: str) -> CombinationForest:
        """Load a forest from file

        Parameters
        ----------
        path : str
            the pickle file to load the tree from

        Returns
        -------
        CombinationForest
            the loaded tree
        """
        raise NotImplementedError("Saving was removed for a pickle vulnerability.")

    def depth(self, x: np.ndarray,
              estimated: bool = True) -> np.ndarray:
        """Determine the average depth of where x is in the forest's trees

        Parameters
        ----------
        x : np.ndarray
            a single data point as a numpy array
        estimated : bool
            if True will use the counts at a leaf node to estimate how far down
            the tree the point would be if it had been grown completely

        Returns
        -------
        np.ndarray
            the depth of a point
        """
        return np.mean(np.array([tree.depth(x, estimated=estimated) for tree in self.trees]))

    def displacement(self, x: np.ndarray) -> np.ndarray:
        """The displacement of a point x in the forest

        Parameters
        ----------
        x : np.ndarray
            a single data point as a numpy array

        Returns
        -------
        np.ndarray
            the "surprise" or displacement induced by including x in the forest
        """

        return np.mean(np.array([tree.displacement(x) for tree in self.trees]))

    def codisplacement(self, x: np.ndarray) -> np.ndarray:
        """Codisplacement allows for colluders in the displacement per RRCF paper [Guha+2016]

        Parameters
        ----------
        x : np.ndarray
            a single data point as a numpy array

        Returns
        -------
        np.ndarray
            the collusive displacement induced by including x in the tree
        """

        return np.mean(np.array([tree.codisplacement(x) for tree in self.trees]))

    def score(self, x: np.ndarray,
              normalize=False,
              use_codisplacement=False,
              use_depth_estimation=False,
              depth_weight=1.0,
              disp_weight=0.0) -> np.ndarray:
        """Calculate the anomaly score

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
            the anomaly score
        """
        return np.mean(np.array([tree.score(x,
                                            normalize=normalize,
                                            use_codisplacement=use_codisplacement,
                                            use_depth_estimation=use_depth_estimation,
                                            depth_weight=depth_weight,
                                            disp_weight=disp_weight)
                                 for tree in self.trees]))

    # def _score_if(self, x: np.ndarray) -> np.ndarray:
    #     """
    #     :param X:
    #     :return:
    #     """
    #     average_depths = np.array([np.mean([tree.depth(xx, estimated=True) for tree in self.trees])
    #                                for xx in x])
    #
    #     def harmonic(n):
    #         """
    #         :param n: index
    #         :type n: int
    #         :return: the nth harmonic number
    #         :rtype: float
    #         """
    #         return np.log(n) + np.euler_gamma
    #
    #     def expected_length(n):
    #         """
    #         :param n: count remaining in leaf node
    #         :type n: int
    #         :return: the expected average length had the tree continued to grow
    #         :rtype: float
    #         """
    #         return 2 * harmonic(n-1) - (2*(n-1) / n) if n > 1 else 0
    #     # bounding_depths = np.array([expected_length(tree.count) for tree in self.trees])
    #     bounding_depth = expected_length(self.trees[0].count)
    #     scores = np.power(2, -average_depths/bounding_depth)
    #     return scores
    #
    # def _score_rrcf(self, x: np.ndarray) -> np.ndarray:
    #     average_codisp = np.array([np.mean([tree.codisplacement(xx) for tree in self.trees]) for xx in x])
    #     bounding_codisp = self.trees[0].count - 1
    #     return average_codisp / bounding_codisp
    #
    # def insert(self, x, label=None):
    #     raise NotImplementedError("will be implemented in a later version")
    #
    # def remove(self, entry):
    #     raise NotImplementedError("will be implmented in a later version")


class IsolationForest(CombinationForest):
    def __init__(self, num_trees=100, tree_properties=None):
        super().__init__(num_trees=num_trees, tree_properties=tree_properties)


class ExtendedIsolationForest(CombinationForest):
    def __init__(self, num_trees=100, tree_properties=None):
        super().__init__(num_trees=num_trees, tree_properties=tree_properties)


class RobustRandomCutForest(CombinationForest):
    def __init__(self, num_trees=100, tree_properties=None):
        super().__init__(num_trees=num_trees, tree_properties=tree_properties)
