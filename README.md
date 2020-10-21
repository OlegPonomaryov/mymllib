# My ML library (mymllib)

While studying [Andrew Ng's Machine Learning course](https://www.coursera.org/learn/machine-learning) (which by the way I highly recommend alongside with his [Deep Learning specialization](https://www.coursera.org/specializations/deep-learning)) the only downside was that all programming assignments were using Octave/Matlab, while I was more intrested in Python. So to replace those practical excercises I've simply started implementing the algorithms I've learned in Python using NumPy and SciPy. They were combined into a single ML library.

Currently **mymllib** contains implementations of machine learning models ranging from linear regression to simple feed forward fully-connected neural network, preprocessing algorithms like feature normalization and polynomial features generation, dimensionality reduction tool (PCA), different metrics and two optimizers (simple gradient descent with learning rate reduction and an adapter to use any algorithm supported by scipy.optimize.minimize()).

There are some **demo notebooks** that show usage of the **mymllib**:
1. [Regression to predict MPG of cars](./demo/mpg_regression.ipynb)
2. [Classification of wine from different cultivars](./demo/wine_classification.ipynb)
3. [Clustering of benign and malignant cells](./demo/cells_clustering.ipynb)

This library is, of course, no more than a study project and it isn't meant for any kind of professional usage. [Scikit-learn](https://scikit-learn.org) is probably the most popular solution for the majority of ML tasks not involving Deep Learning.
