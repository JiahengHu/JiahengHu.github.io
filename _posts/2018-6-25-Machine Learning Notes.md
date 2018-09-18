---
layout: single
mathjax: true
title: Machine Learning Notes
---

In this post, I mainly post thoughts from CS 229 by Prof. Andrew Ng and Machine Learning by Prof. John Paisley. It is much like self-displined. Meanwhile, I am always trying to summarize and combine what it is worth in this notes from different machine learning classes that I have been involved. 

It is recommended that you have some basic knowledge about probability, linear algebra and vector calculus before reading the notes. The reading can be done simply from beginning to ending or be indexed from table of contents based on your needs. 

* This will become a table of contents (this text will be scraped).
{:toc}

# Lecture 1 Supervised Learning

A classical learning problem is called supervised learning. In this case, we usually have an input called features and output called target. The goal is that given some features we ask the trained model to predict the output. To do so, we collect a training dataset in which we have a number of pairs of training sample composing of a feature vector and its corresponding output. Since we have ground truth for every single input, we call this type of learning as supervised learning and the learned model as hypothesis. An example can be shown in the table below. 

![Supervise Learning Intuition](/images/cs229_lec1_intuit.png)

When the target output is in continuous space, we call it a regression problem. When the target output is in discrete space, we call it as a classification problem. 

## 1 Linear Regression

A linear regression probelm can be models as :

$$h(x) = \sum\limits_{i=0}^n \theta_i x_i = \theta^Tx$$

We have $\theta_0$ for the bias and sometimes it is called intercept term. Image that you try to regress for a line in 2D domain, the intercept term basically determines where the line crossed y-axis. $\theta$ is called parameters which we want to learn from training data. 

To learn it, we also define the cost function on which we are trying to minimize:

$$J(\theta) = \frac{1}{2}\sum\limits_{i=q}^m (h_{\theta}(x^{(i)}) - y^{(i)})^2$$

The goal is to find such $\theta$ that minimize the cost. The question is how.

### Least Mean Sqaure(LMS) algorithm

LMS algorithm essentially uses gradient descent to find the local min. To implement it, we start an initial guess $\theta = \overrightarrow{0}$ and then update repeatedly as:

$$\theta_j = \theta_j - \alpha \frac{\partial}{\partial \theta_j}J(\theta)$$

where j spans all the components in feature vector. $\alpha$ is called learning rate,controlling how fast it learns. 

Now,we can solve the partial derivative with respect to one sample as :

$$\begin{align}
\frac{\partial}{\partial \theta_j}J(\theta) &= \frac{\partial}{\partial \theta_j} \frac{1}{2} (h_{\theta}(x)-y)^2\\
&= 2\frac{1}{2}(h_{\theta}(x)-y) \frac{\partial}{\partial \theta_j} \frac{1}{2} (h_{\theta}(x)-y)\\
&= (h_{\theta}(x)-y) \frac{\partial}{\partial \theta_j}(\sum\limits_{i=0}^n \theta_i x_i - y) \\
&= (h_{\theta}(x)-y) x_j
\end{align}$$

So the update for all the samples are:

$$\theta_j = \theta_j + \alpha\sum\limits_{i=0}^m (y^{(i)} - h_{\theta}(x^{(i)}))x_j^{(i)}$$

where m is the number of training examples and j can span the dimension of feature vector. This algorithm takes all the factors from every single training sample. We call it **batch gradient descent**. This method is senstive to the local minimum (i.e. might arrive at saddle point) where we generally assume that the cost function has only global minimum which is the case(J is convex). An graphical illustration can be shown below. 

![Batch Gradient Descent](/images/cs229_lec1_bgd.png)

Note that in the updating, we run through all the samples to make one step forward to local min. This step is computationally expensive if m is very large. Thus,in this case, we introduce a similar algortihm called **stochastic gradient descent** where only a small part of samples are fed into the algorithm. By doing this, we can converge faster although it might oscillate a lot. It will produce good approximation to the global minimum. Thus, we use this often in reality. 
