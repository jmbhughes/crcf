# Combination Robust Cut Forests
Isolation Forests **[Liu+2008]** and Robust Random Cut Trees **[Guha+2016]** are very similar in many ways, 
as outlined in the [supporting overview](overview.pdf). Most notably, they are extremes
of the same outlier scoring function: 

![equation](https://latex.codecogs.com/gif.latex?%5Ctext%7BFor%20%7D%20%5Ctheta%20%5Cin%20%5B0%2C1%5D%20%5Ctext%7B%20let%20%7D%20%5Cmathrm%7Bscore%7D%28x%29%20%3D%20%5Ctheta%20%5Cmathrm%7Bdepth%7D%28x%29%20&plus;%20%281-%5Ctheta%29%5Cmathrm%7Bdisp%7D%28x%29)

The combination robust cut forest allows you to combine both scores by using an theta other than 0 or 1. 

# Install
Download the repository and run 
`python3 setup.py install` or `pip3 install .`

The tests can be run from `pytest` with `python3 setup.py test`.

# Tasks
- [ ] complete basic implementation
- [ ] fix documentation generation error
- [ ] provide clear documentation and usage instructions
- [ ] incorporate categorical variable support, including categorical rules
- [ ] complete the write-up document with a benchmarking of performance

# References
- **[Liu+2008]**: [Liu, Fei Tony, Kai Ming Ting, and Zhi-Hua Zhou. 
"Isolation forest." In 2008 Eighth IEEE International Conference on Data Mining, 
pp. 413-422. IEEE, 2008.](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf?q=isolation-forest)
- **[Guha+2016]**: [Guha, Sudipto, Nina Mishra, Gourav Roy, and Okke Schrijvers. 
"Robust random cut forest based anomaly detection on streams." 
In International conference on machine learning, pp. 2712-2721. 2016.](http://proceedings.mlr.press/v48/guha16.pdf)
