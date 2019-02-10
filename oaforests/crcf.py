import numpy as np
from abc import abstractmethod, ABC


class Rule(ABC):
    def __init__(self):
        pass

    def evaluate(self, X):
        return np.array([self.__evaluate(x) for x in X])

    @abstractmethod
    def __evaluate(self, x):
        pass

    @classmethod
    def generate(cls, bounding_box, mode="uniform"):
        if mode == "uniform":
            return cls.__generate_uniform(bounding_box)
        elif mode == "biased":
            return cls.__generate_biased(bounding_box)
        else:
            raise RuntimeError("mode must either be uniform or biased, not {}".format(mode))

    @classmethod
    @abstractmethod
    def __generate_uniform(cls, bounding_box):
        pass

    @classmethod
    @abstractmethod
    def __generate_biased(cls, bounding_box):
        pass


class AxisAlignedRule(Rule):
    def __init__(self, dimension, value):
        super(AxisAlignedRule).__init__()
        self.dimension, self.value = dimension, value

    def __evaluate(self, x):
        return x[self.dimension] < self.value

    @classmethod
    def __generate_uniform(cls, bounding_box):
        dimension = np.random.randint(0, bounding_box.shape[0])
        value = np.random.uniform(bounding_box[dimension][0], bounding_box[dimension][1])
        return AxisAlignedRule(dimension, value)

    @classmethod
    def __generate_biased(cls, bounding_box):
        lengths = np.diff(bounding_box)
        dimension = np.random.choice(np.arange(bounding_box.shape[0]), p=lengths/np.sum(lengths))
        value = np.random.uniform(bounding_box[dimension][0], bounding_box[dimension][1])
        return AxisAlignedRule(dimension, value)


class NonAxisAlignedRule(Rule):
    def __init__(self, normal, point):
        super(NonAxisAlignedRule).__init__()
        self.normal, self.point = normal, point

    def __evaluate(self, x):
        pass

    @classmethod
    def __generate_uniform(cls, bounding_box):
        pass

    @classmethod
    def __generate_biased(cls, bounding_box):
        pass
