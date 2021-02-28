# My ML library (mymllib)


A library with implementations of some algorithms I've studied in
[Andrew Ng's Machine Learning course](https://www.coursera.org/learn/machine-learning) 
and some other [DeeoLearning.AI](https://www.deeplearning.ai/) courses.

Currently **mymllib** contains implementations of machine learning models ranging from linear regression to simple feed forward fully-connected neural network, preprocessing algorithms like feature normalization and polynomial features generation, dimensionality reduction tool (PCA), different metrics and two optimizers (simple gradient descent with learning rate reduction and an adapter to use any algorithm supported by scipy.optimize.minimize()).

There are some **demo notebooks** that show usage of the **mymllib**:
1. [Regression to predict MPG of cars](./demo/mpg_regression.ipynb)
2. [Classification of wine from different cultivars](./demo/wine_classification.ipynb)
3. [Clustering of benign and malignant cells](./demo/cells_clustering.ipynb)
4. [Tweets sentiment classification with naive Bayes](./demo/tweets_classification.ipynb)

This library is, of course, no more than a study project and it isn't meant for any kind of professional usage. [Scikit-learn](https://scikit-learn.org) is probably the most popular solution for the majority of ML tasks not involving Deep Learning.
