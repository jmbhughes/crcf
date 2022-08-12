# Welcome to `crcf`

This is the `crcf` package that unites Robust Cut Forests and Isolation Forests as
combination robust cut forests. 

Isolation Forests **[Liu+2008]** and Robust Random Cut Trees **[Guha+2016]** are very similar in many ways, 
as outlined in the [supporting overview](overview.pdf). Most notably, they are extremes
of the same outlier scoring function. The combination robust cut forest allows 
you to combine both scores by using a $\theta$ other than 0 or 1. 

$$
\theta \textrm{Depth} + (1 - \theta) \textrm{[Co]Disp}
$$

For a full walkthrough of the mathematics behind these forests, please see the 
[overview](https://github.com/jmbhughes/crcf/blob/master/overview.pdf). 

## References
- **[Liu+2008]**: [Liu, Fei Tony, Kai Ming Ting, and Zhi-Hua Zhou. 
"Isolation forest." In 2008 Eighth IEEE International Conference on Data Mining, 
pp. 413-422. IEEE, 2008.](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf?q=isolation-forest)
- **[Guha+2016]**: [Guha, Sudipto, Nina Mishra, Gourav Roy, and Okke Schrijvers. 
"Robust random cut forest based anomaly detection on streams." 
In International conference on machine learning, pp. 2712-2721. 2016.](http://proceedings.mlr.press/v48/guha16.pdf)