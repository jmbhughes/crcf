# Combination Robust Cut Forests
[![CodeFactor](https://www.codefactor.io/repository/github/jmbhughes/crcf/badge)](https://www.codefactor.io/repository/github/jmbhughes/crcf)
[![PyPI version](https://badge.fury.io/py/crcf.svg)](https://badge.fury.io/py/crcf)
[![codecov](https://codecov.io/gh/jmbhughes/crcf/branch/main/graph/badge.svg?token=YBZERHDU75)](https://codecov.io/gh/jmbhughes/crcf)

Isolation Forests **[Liu+2008]** and Robust Random Cut Trees **[Guha+2016]** are very similar in many ways, 
as outlined in the [supporting overview](overview.pdf). Most notably, they are extremes
of the same outlier scoring function: 

$$\theta \textrm{Depth} + (1 - \theta) \textrm{[Co]Disp}$$ 

The combination robust cut forest allows you to combine both scores by using an theta other than 0 or 1. 

# Install
You can install with through `pip install crcf`. Alternatively, you can download the repository and run 
`python3 setup.py install` or `pip3 install .` Please note that this package uses features from Python 3.7+
and is not compatible with earlier Python versions. 


# Tasks
- [X] complete basic implementation
- [X] provide clear documentation and usage instructions
- [ ] ensure interface allows for fitting and scoring on multiple points at the same time
- [ ] implement a better saving method than pickling
- [ ] use random tests with hypothesis
- [ ] implement tree down in cython
- [ ] accelerate forests with multi-threading
- [ ] incorporate categorical variable support, including categorical rules
- [ ] complete the write-up document with a benchmarking of performance

# References
- **[Liu+2008]**: [Liu, Fei Tony, Kai Ming Ting, and Zhi-Hua Zhou. 
"Isolation forest." In 2008 Eighth IEEE International Conference on Data Mining, 
pp. 413-422. IEEE, 2008.](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf?q=isolation-forest)
- **[Guha+2016]**: [Guha, Sudipto, Nina Mishra, Gourav Roy, and Okke Schrijvers. 
"Robust random cut forest based anomaly detection on streams." 
In International conference on machine learning, pp. 2712-2721. 2016.](http://proceedings.mlr.press/v48/guha16.pdf)
