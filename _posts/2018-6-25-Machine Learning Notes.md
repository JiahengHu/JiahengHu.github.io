---
layout: single
mathjax: true
toc: true
toc_sticky: true
category: Machine Learning
tags: [notes]
qr: machine_learning_notes.png
title: Machine Learning Notes
nav: MachineLearning
---

In this post, I mainly post thoughts from CS 229 by Prof. Andrew Ng and Machine Learning by Prof. John Paisley. It is much like self-displined. Meanwhile, I am always trying to summarize and combine what it is worth in this notes from different machine learning classes that I have been involved. 

It is recommended that you have some basic knowledge about probability, linear algebra and vector calculus before reading the notes. The reading can be done simply from beginning to ending or be indexed from table of contents based on your needs. 

**Note: You can share this post to social network at the bottom.**

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

## 2 Normal Equations

In the section above, we use the iterative algorithm to find the minimum. This method is used usually when the solution to the derivative equal to zero is intractable. If we are able to find the derivative and solve when it is zero, we can explicitly calculate the local minimum. Before going through, we need the memory refresher for the math!

### Matrix derivatives

Some of the concepts are discussed in the other post,which you can find it [here](https://wei2624.github.io/Useful-Formulas-for-Math/).

In this subsection, I will talk about trace operator in linear algebra. Basically, the trace operation is defined as:

$$trA = \sum\limits_{i=1}^n A_{ii}$$

where A must be a square matrix. Now, I will list the properties of trace and write proof if time permitted. 

$$trAB = trBA$$

$$trABC = trCAB = trBCA$$

$$trABCD = trDABC = trCDAB = trBCDA$$

$$trA = trA^T$$

$$tr(A+B) = trA + trB$$

$$tr\alpha A = \alpha trA$$

$$\triangledown_A trAB = B^T$$

$$\triangledown_{A^T}f(A) = (\triangledown_A f(A))^T$$

$$\triangledown_A trABA^TC = CAB + C^TAB^T$$

$$\triangledown_A \lvert A \rvert = \lvert A \rvert(A^{-1})^T$$

### Least Square revisited

So now instead of iteratively finding the solution, we explicitly calculate the derivative of the cost function and set to zero for producing the solution in one shot. 

We define training data as :

$$X = \begin{bmatrix} -(x^{(1)})^T-\\ -(x^{(2)})^T- \\ \vdots  \\ -(x^{(m)})^T- \end{bmatrix}$$

and its target values as:

$$\overrightarrow{y} = \begin{bmatrix} y^{(1)}\\ y^{(2)} \\ \vdots  \\ y^{(m)} \end{bmatrix}$$

Let hypothesis be $h_{\theta}(x^{(i)}) = (x^{(i)})^T\theta$ and we have:

$$X\theta - \overrightarrow{y} = \begin{bmatrix} h_{\theta}(x^{(1)}) - y^{(1)}\\ h_{\theta}(x^{(2)}) - y^{(2)} \\ \vdots  \\ h_{\theta}(x^{(m)}) - y^{(m)} \end{bmatrix}$$

Thus, 

$$J(\theta) = \frac{1}{2}(X\theta - \overrightarrow{y})^T(X\theta - \overrightarrow{y}) = \frac{1}{2}\sum\limits_{i=1}^m (h_{\theta}(x^{(i)}) - y^{(i)})^2$$

So at this point,we need to find the the derivative of J with respect to $\theta$. From the properties of trace, we know that:

$$\triangledown_{A^T}trABA^TC = B^TA^TC^T + BA^TC$$

We also know the trace of scaler is itself. Then:

$$\begin{align}
\triangledown_{\theta}J(\theta) &= \triangledown_{\theta}\frac{1}{2}(X\theta - \overrightarrow{y})^T(X\theta - \overrightarrow{y})\\
&= \frac{1}{2}\triangledown_{\theta} tr(\theta^TX^TX\theta - \theta^TX^T\overrightarrow{y} - \overrightarrow{y}^TX\theta + \overrightarrow{y}^T\overrightarrow{y} \\
&= \frac{1}{2}\triangledown_{\theta} (tr\theta^TX^TX\theta - 2tr\overrightarrow{y}^TX\theta)\\
&= \frac{1}{2}(X^TX\theta + X^TX\theta - 2X^T\overrightarrow{y})\\
&= X^X\theta - X^T\overrightarrow{y}
\end{align}$$

We set it to zero and we obtain normal equation:

$$X^TX\theta = X^T\overrightarrow{y}$$

Then, we should update parameter as:

$$\theta = (X^TX)^{-1}X^T\overrightarrow{y}$$

## 3 Probabilistic interpretation

We assume that the target variable and the inputs are related as:

$$y^{(i)} = \theta^Tx^{(i)} + \epsilon^{(i)}$$

where $\epsilon^{(i)}$ is random variable which can capture noise and unmodeled effects. We also assume that noise are distributed IID from Gaussianw with zero mean and some variance $\sigma^2$, which is a traditional way to model. Now, we can say:

$$p(y^{(i)} \lvert x^{(i)};\theta) = \frac{1}{\sqrt{2\pi \sigma}}\exp\big(-\frac{(y^{(i)} - \theta^Tx^{(i)})^2}{2\sigma^2}\big)$$

This function can be viewed as the funciton and y and maybe x as well since it might be some randomness on feature vectors as well with fixed parameter $\theta$. Thus, we can call it likelihood function:

$$L(\theta) = \prod_{i=1}^{m} p(y^{(i)} \lvert x^{(i)};\theta)$$

We need to find such $\theta$ so that with the chosen $\theta$ the probability of y given a x is maximized. We call it **maximum likelihood**. To simplify, we find the max of **log likelihood**:


$$\begin{align}
\ell &= \log L(\theta)\\
&= \log \prod_{i=1}^{m} \frac{1}{\sqrt{2\pi \sigma}}\exp\big(-\frac{(y^{(i)} - \theta^Tx^{(i)})^2}{2\sigma^2}\big)\\
&= \sum\limits_{i=1}^{m} \log \frac{1}{\sqrt{2\pi \sigma}}\exp\big(-\frac{(y^{(i)} - \theta^Tx^{(i)})^2}{2\sigma^2}\big)\\
&= m\log\frac{1}{\sqrt{2\pi\sigma}} - \frac{1}{\sigma^2}\frac{1}{2}\sum\limits_{i=1}^{m}(y^{(i)} - \theta^T x^{(i)})^2
\end{align}$$

Maximizing this with respect to $\theta$ will give the same answer as minimizing J. That means we can justify what we have done in LMS in probabilitic point of view. 

## 4 Locally Weighted Linear Rgression

In the regression method discussed above, we treat the cost resulted from training samples equally in the process. However, this might not be proper since some outliers should placed less weights. We implement this idea by placing weights to each sample with respect to the querying point. For example, such a weight can be:

$$w^{(i)} = \exp\big(-\frac{(x^{(i)} - x)^2}{2r^2}\big)$$

Although this is similar to Gaussian, it has nothing to do with it. And x is the querying point. We need to keep all the training data for new prediction. 

## 5 Classification and Logistic regression

We can imagine the clssification as a special regression problem where we only regress to a set of binary values, 0 and 1. Sometimes, we use -1 and 1 notation as well. We call it negative class and positive class, respectively.

However, it does not make sense that we predict any values other than 0 and 1. Therefore, we modify the hypothese function to be:

$$h_{\theta}(x) = g(\theta^T x) = \frac{1}{1+\exp(-\theta^Tx)}$$

where g is called **logistic function or sigmoid function**. A plot of logistic function can be found below:

![Logistic Function](/images/cs229_lec1_logistic.png)

It ranges from 0 to 1 as output. 

let's look at what it looks like when we take derivative of logistic funciton:

$$\begin{align}
\frac{d}{dz} g(z) &= \frac{1}{(1+\exp(-z))^2}\big(\exp(-z)\big)\\
&= \frac{1}{(1+\exp(-z))}\Big(1 - \frac{1}{(1+\exp(-z))^2}\Big)\\
&= g(z)(1-g(z))
\end{align}$$

With this prior knowledge, the question is how are we supposed to find $\theta$. So we know least square regression can be derived from maximum likelihood algorithm, which is where we should start from. 

We assume:

$$P(y \lvert x;\theta) = (h_{\theta}(x))^y (1 - h_{\theta}(x))^{1-y}$$

where y should be either one or zero. Assuming that samples are iid, we have likelihood as:

$$\begin{align}
L(\theta) &= \prod_{i=1}^{m} p(y^{(i)}\lvert x^{(i)};\theta)\\
&= \prod_{i=1}^{m} (h_{\theta}(x^{(i)}))^{y^{(i)}} (1 - h_{\theta}(x^{(i)}))^{1-y^{(i)}}
\end{align}$$

using log likelihood, we can have:

$$\log L(\theta) = \sum\limits_{i=1}^m y^{(i)}\log h(x^{(i)}) + (1-y^{(i)}\log(1-h(x^{(i)})$$

Then, we can use graident descent to optimize the likelihood. In updating, we should have $\theta = \theta + \alpha\triangledown_{\theta}L(\theta)$. Note we have plus sign instead of minus sign since we are finding max not min. To find the derivative, 

$$\begin{align}
\frac{\partial}{\partial\theta_j}L(\theta) &= \bigg(y\frac{1}{g(\theta^Tx)}-(1-y)\frac{1}{1-g(^Tx)}\bigg)\frac{\partial}{\partial\theta_j}g(\theta^Tx)\\
&= \bigg(y\frac{1}{g(\theta^Tx)}-(1-y)\frac{1}{1-g(^Tx)}\bigg) g(\theta^Tx)(1 - g(\theta^Tx))\frac{\partial}{\partial\theta_j}\theta^Tx\\
&= (y - h_{\theta}(x))x_j
\end{align}$$

From the fisrt line to second line, we use the derivative of logistic function derived above. This gives us the update rule for each dimension on feature vector. Although we have same algorithm as LMS in this case, the hypothesis in this cases is different. It is not surprising to have the same equation when we talk about Generalized Linearized Model. 

## 6 Digression: The Perceptron Learning Algortihm

We will talk about this in Learning Theory in more detials. In short, we change our hypothesis function to be:

$$g\theta^Tx) = \begin{cases} 1  \text{, if } \theta^Tx geq 0 \\ 0  \text{, otherwise} \\ \end{cases}$$

The updating equation remains the same. This is called **perceptron learning algorithm**.\

## 7 Newton's Method for Maximizing

So imagine that we want to find the root of a function f. Newton's method allows us to do this task in quadratic speed. The idea is to initialize $x_0$ randomly and find the tangent line of $f(x_0)$,dentoed $f^{\prime}(x_0)$. We use the root of $f^{\prime}(x_0)$ as new x. We also define the distance between new x and old x as $\Delta$. An example of this can be shown as:

![Newton's Method](/images/cs229_lec1_newton.png)

So we now have:

$$f^{\prime}(x_0) = \frac{f(x_0)}{\Delta} \Rightarrow \Delta = \frac{f(x_0)}{f^{\prime}(x_0)}$$

Derived from this idea, we can let $f(x) = L^{\prime}(\theta)$. Going this way, we can find max of objective function faster. For finding min, it is similar. 

If $\theta$ is vector-valued, we need to use Hessian in the updating. More details about Hession can be found in [the other post](https://wei2624.github.io/Useful-Formulas-for-Math/). In short, to update, we have:

$$\theta = \theta - H^{-1}\triangledown_{\theta}L(\theta)$$

Alhtough it converges in quadratic, each updating is more costly than gradient descent. 

## 8 Generalized Linear Models and Exponential Family

Remeber that we have "coincidence" where the updating of logistic regression and least mean square regress ends up with same form. They are special cases in the big family called GLM. The reason why it is called linear is because every distribution in this family places a linear relationship between varaibles and their weights. 

Before going to GML, we fisrt talk about exponential family distributions as the foundation to GLM. We define that a class of distribution is in the exponential family if it can be written in the form:

$$p(y;\eta) = b(y)\exp(\eta^T T(y) - a(\eta))$$

where $\eta$ is called **natural parameter**, $T(y)$ is called **sufficient statistic** and $a(\eta)$ is called **log partition fucntion**. Usually, $T(y) = y$ is our case. the term $-a(\eta)$ is the normalizing constant. 

T,a and b are fixed parameters with which we can vary $\eta$ to establish different distribution in a class of distributuion. Now, we can show that Bernoulli and Gaussian belong to exponential family. 

Bernoulli:

$$\begin{align}
p(y;\phi) &= \phi^y(1-\phi)^{1-y}\\
&= \exp(y\log\phi + (1-y)\log(1-\phi))\\
&= \exp\bigg(\bigg(\log\bigg(\frac{\phi}{1-\phi}\bigg)\bigg)y+\log(1-\phi)\bigg)
\end{align}$$

where:

$$\eta = \log(\phi/(1-\phi))$$

$$T(y) = y$$

$$a(\eta) = -\log(1-\phi) = \log(1+e^{\eta})$$

$$b(y) = 1$$

Gaussian:

$$p(y;\mu) = \frac{1}{\sqrt{2\pi}}\exp\bigg(-\frac{1}{2}y^2\bigg)\exp\bigg(\mu y - \frac{1}{2}\mu^2\bigg)$$

where $\sigma$ is 1 in this case(we can still do the same thing with varying $\sigma$ and :

$$\eta = \mu$$

$$T(y) = y$$

$$a(\eta) = \mu^2/2 = \eta^2/2$$

$$b(y) = (1/\sqrt{2\pi})\exp(-y^2/2)$$

Other exponential distribution: Multinomial, Possion, gamma and exponential, beta and Dirichlet. Since they are all in exponential family, what we can do is to study exponential family in general form and vary $\eta$ to model differently. 

## 9 Constructing GLM

As discussed, once we know T,a and b, the family of distribution is already determined. We only need to find $\eta$ to determine the exact distribution. 

For example, assume that we want to predict y given x. Before moving on deriving GLM of this regression problem, we make three major assumption about this:

**(1)** We always assume $y \lvert x;\theta \thicksim \text{ExponentialFamily}(\eta)$. 

**(2)** In general, we want to predict the expected vaye of T(y) given x. Most likely, we have $T(y) = y$. Formally, we have $h(x) = \mathbb{E}[y\lvert x]$, which is true for both logistic regression and linear regression. Note that in logistic regression, we always have $\mathbb{E}[y\lvert x] = p(y=1\lvert x;\theta)$.

**(3)** The input and natural parameter are related as:$\eta = \theta^Tx$

### 9.1 Ordinary Least Squares

In this case, we have $y\thicksim \mathcal{N}(\mu,\sigma^2)$. Previoulsy, we discussed about Gaussian as exponential family. In particular, we have:

$$\begin{align}
h_{\theta}(x) &= \mathbb{E}[y\lvert x;\theta]\\
&= \mu\\
&= \eta \\
&=\theta^Tx
\end{align}$$

where the first equation is from assumption (2); the second is by definition; the third is from early derivation; the last is from assumption (3). 

### 9.2 Logistic Regression

In this setting, we predict either 1 or 0 for class label. Recall that, in Bernoulli, we had $\phi=1/(1+e^{\eta}$. Thus, we can derive the following equation as:

$$\begin{align}
h_{\theta}(x) = \mathbb{E}[y\lvert x;\theta]\\
&= \phi\\
&= 1/(1+e^{-\eta}) \\
&= 1/(1+e^{=-\theta^Tx})
\end{align}$$

This partially explains why we came up with the form like sigmoid function. Because we assume that y follows from Bernoulli given x, it is natural to have sigmoid function resulted from exponential family. To predict, we think that expected value of $T(y)$ with respect to $\eta$ is a reasonable guess, namely **canonical response function or inverse of link function**. In general, response function is the function of $\eta$ and gives the relationships between $\eta$ and distribution parameters, while link function produces $\eta$ as a function of distribution parameter. The inversion means to express one in terms of the other, which has nothing to do with mathematical meaning of inversion.  From the derivation above, we know that the canonical response function of Bernoulli is logistic function and that of Gaussian is mean function. 

### 9.3 Softmax Regression

In a broader case, we can have multiple classes instead of binary classes above. It is natural to model it as Multinomial distribution, which also belongs to exponential family that can be derived from GLM. 

In multinomial, we can define $\phi_1,\phi_2,\dots,\phi_{k-1}$ to be the corresponding probability of $k-1$ classes. We do not need all k classes since last is determined once the previous $k-1$ are set. So we can write $\phi_k = 1-\sum_{i=1}^{k-1}\phi_i$.

We first define $T(y) \in \mathbb{R}^{k-1}$ and : 

$$T(1) = \begin{bmatrix} 1\\ 0 \\ \vdots  \\ 0 \end{bmatrix}, T(2) = \begin{bmatrix} 0\\ 1 \\ \vdots  \\ 0 \end{bmatrix},\dots,T(k) = \begin{bmatrix} 0\\ 0 \\ \vdots  \\ 0 \end{bmatrix}$$

Note that for $T(k)$, we just have all zeros in the vector since the length of vector is k-1. We let $T(y)_i$ define i-th element in the vector. The definition of indicator is also introduced in course notes, which I am not talking in details here. 

Now, we show the steps to derive Multinomial as exponential family:

$$\begin{align}
p(y;\phi) &= \phi_1^{\mathbb{1}[y=1]}\phi_2^{\mathbb{1}[y=2]}\dots\phi_k^{\mathbb{1}[y=k]}\\
&= \phi_1^{\mathbb{1}[y=1]}\phi_2^{\mathbb{1}[y=2]}\dots\phi_k^{1 - \sum_{i=1}^{k-1}\mathbb{1}[y=i]}\\
&= \phi_1^{T(y)_1}\phi_2^{T(y)_2}\dots\phi_k^{1 - \sum_{i=1}^{k-1}T(y)_i} \\
&= \exp\Big(T(y)_1\log(\phi_1/\phi_k)+T(y)_2\log(\phi_2/\phi_k) + \dots + T(y)_{k-1}\log(\phi_{k-1}/\phi_k)+ \log(\phi_k)\Big) \\
&= b(y)\exp(\eta^TT(y) - a(\eta))
\end{align}$$

where

$$\eta = \begin{bmatrix} \log(\phi_1/\phi_k)\\ \log(\phi_2/\phi_k) \\ \vdots  \\ \log(\phi_{k-1}/\phi_k) \end{bmatrix}$$

and $a(\eta) = -\log(\phi_k)$ and $b(y) = 1$. 

This formulates multinomial as exponenital family. We can now have the link function as:

$$\eta_i = \log(\frac{\phi_i}{\phi_k})$$

To get the response function, we need to invert the link function:

$$e^{\eta_i} = \frac{\phi_i}{\phi_k}$$

$$\phi_k e^{\eta_i} = \phi_i$$

$$\phi_k \sum\limits_{i=1}^{k}e^{\eta_i} = \sum\limits_{i=1}^{k} \phi_i$$

Then, we have the response function:

$$\phi_i = \frac{e^{\eta_i}}{\sum_{j=1}^{k}e^{\eta_j}}$$

This response function is called **softmax function**. 

From the assumption (3) in GLM, we know that $\eta_i = \theta_i^Tx$ for $i=1,2,\dots,k-1$ and $\theta_i \in \mathbb{R}^{n+1}$ is the parameters of our GLM model and $\theta_k$ is just 0 so that $\eta_k = 0$. Now, we have the model based on x:

$$p(y=i\lvert x;\theta) = \phi_i = \frac{e^{\eta_i}}{\sum_{j=1}^{k}e^{\eta_j}} = \frac{e^{\theta_i^T x}}{\sum_{j=1}^{k}e^{\theta_j^Tx}}$$

This model is called **softmax regression**, which is a generalization of logistic regression. Thus, the hypothesis will be:

$$\begin{align}
h_{\theta}(x) &= \mathbb{E}[T(y)\lvert x;\theta]\\
&=\begin{bmatrix} \phi_1\\ \phi_2 \\ \vdots  \\ \phi_{k-1} \end{bmatrix} \\
&= \begin{bmatrix} \frac{\exp(\theta_1^Tx)}{\sum_{j=1}^k\exp(\theta_j^Tx)}\\ \frac{\exp(\theta_2^Tx)}{\sum_{j=1}^k\exp(\theta_j^Tx)} \\ \vdots  \\ \frac{\exp(\theta_{k-1}^Tx)}{\sum_{j=1}^k\exp(\theta_j^Tx)} \end{bmatrix}
\end{align}$$

Now, we need to fit $\theta$ such that we can max the log likelihood. by definition, we can write it out:

$$\begin{align}
L(\theta) &= \sum\limits_{i=1}^m \log(p(y^{(i)}\lvert x^{(i)};\theta)\\
&=\sum\limits_{i=1}^m \log\prod_{l=1}^k\bigg(\frac{\exp(\theta_l^Tx)}{\sum_{j=1}^k\exp(\theta_j^Tx)}\bigg)^{\mathbb{1}{y^{(i)}=l}}
\end{align}$$

We can use gradient descent or Newton's method to find the max of it. 

# Lecture 2 Generative Learning Algorithm

## 1 Discriminative Model

Algorithms try to directly classify a label for input such logstic regression and perceptron algorithm. The discriminative model does not have a concept of what the object might look like. They just classify. It cannot generate a new image based on the boundary.

## 2 Generative Model

Models fisrt try to learn each object might look like such as Bayesian method. Then, based on input, it gives a probability of the input being this class. It has the concepts on what the object might look like. It can generate a new image based on the past knowledge. 

With class prior, we can use Bayes rule to calculate the probability of being each class and then take the one with a bigger value. 

## 3 Gaussian Discriminant Analysis 

For a vector-values random variable Z:

$$Cov(Z) = E[(Z-E[Z])(Z-E[Z])^T] = E[ZZ^T - 2ZE[Z]^T + E[Z]E[Z]^T] $$

$$= E[ZZ^T] - 2E[Z]E[Z]^T + E[Z]E[Z]^T = E[ZZ^T] - E[Z]E[Z]^T$$

## 4 GDA and logistic regression

If $P(x\lvert y)$ is multivariate gaussian with shared covariance, then $P(y\lvert x)$ follows a logistic function. It means that GDA requires a strong assumption that data of each class can be modeled with a gaussian with shared covariance. However, GDA will fit better and train faster if assumptions are correct. 

On the other side, if assumption cannot be made, logistic regression is less sensitive. For example, Poisson can replace gaussian also leading to logistic regression. 

## 5 Naive Bayes

This is for learning discrete valued random variables like text classification. In text classification, a word vector is used for training. However, if we have 50000 words and try to model it as multinominal, then the dimension of parameter is $2^50000-1$, which is too large. Thus, we make **Naive Bayes Assumption:**

Each word is conditionally independent to each other based on given class. 

Then, we have:

$$P(x_1,...,x_50000\lvert y) = P(x_1\lvert y)P(x_2\lvert y,x_1)...P(x_50000\lvert y,x_1,x_2,...,x_49999) $$

$$= \prod\limits_{i=1}^{n} P(x_i\lvert y)$$

We apply **probability law of chain rule** for the first step and naive basyes assumption for the second step. 

After finding the max of **log joint likelihood**, which is:

$$\mathcal{L}(\phi_y,\phi_{j\lvert y=0},\phi_{j\lvert y=1}) = \prod\limits_{i=1}^{m} P(x^{(i)},y^{(i)}) $$

where $\phi_{j\lvert y=1} = P(x_j = 1 \lvert y = 1)$.

Then, we can use **Bayes Rule** to calculate $P(y=1\lvert x)$ and compare which is higher. 

**Ext**: In this case, we model $P(x_i\lvert y)$ as Bernouli since it is binary valued. That is, it can be either 'have that word' or 'not have that word'. Bernouli takes class label as input and models its probability but it has to binary. To deal with non-binary valued $x_i$, we can model it as Multinomial distribution, which can be parameterized with multiple classes. 

**Summary:** Naive Bayes is for discrete space. GDA is for continous space. We can alsway discretize it. 

## 6 Laplace smoothing

The above shwon example is generally good but will possibly fail where a new word which does exist in the past training samples appear in the coming email. In such case, it would cause $\phi$ for both classes to become zero because the models never see the word before. The model will fail to make prediction. 

This motivates a solution called **Laplace Smoothing**, which sets each parameter as:

$$\phi_j = \frac{\sum_{i=1}^{m} \mathbb{1}[z^{(i)}] + 1}{m+k}$$

where k is the number of classes. In reality, the Laplace smoothing does not make too much difference since it usually has all the words but it is good to have it here. 

## 7 Event Models

In generative setting, we should have class prior and likelihood for each class. For the likelihood model, it can be Bernoulli if it is binary or it can be Multinomial if it is multi-class. 

# Lecture 3 Support Vector Machine (SVM)

## 1 Intuition Notation

![SVM Intuition](/images/svm_intuition.png)

From the figure, we have A, B and C point in the space. A is the safest point since it is far from the **boundary line**, while C is the most dangerous point since it is close to the **hyperplane**. The distance between the boundary line and the point is called **margin**.

We also denote $x$ as feature vector, $y$ as label and $h$ as classifier. Thus, the classifier an be shown as:

$$h_{w,b}(x) = g(w^Tx + b)$$

Note, we have w, b instead $\theta$ here. And the label only takes the value 1 and -1 instead of 0 and 1. The classifier predicts directly as 1 or -1 like **perceptron algorithm** without calculating the probability like what logistic did. **However, this does not mean SVM cannot output its corresponding probability.**

## 2 Functional and Geometric Margins

**functional margine** with respect to training example:

$$\overset{\wedge}{\gamma^{(i)}} = y^{(i)}(w^Tx^{(i)} + b)$$

We want $(w^Tx^{(i)} + b)$ to be a large positive number if label is positive or large negative number if label is negative. Thus, it means that **functional margin should be positive to be correct. And the larger the margin, the more confident we are.** However, this might not be meaningful when we replace w and b with 2w and 2b without changing anything else. Thus, this leads to the deifnition **geometric margine** coming next. Furthermore, we denote the function margin as:

$$\overset{\wedge}{\gamma} = \min_{i=1,\dots,m} \overset{\wedge}{\gamma^{(i)}} $$

where m is the number of training samples. 

**Geometric Margins:** In functional margin, We need to normalize w and b **with respect to the norm of w** since magnitude of w and b should not affect the scale of the margin.A figure for geometric margin can be shown:

![SVM Geometric Margins](/images/svm_gm.png)

It shows a vector w also called **support vector** which is perpendicular to the boundary line, which is always true. To prove this, you just need to take two points on boundary line to get a parallel vector and prove that the dot product is 0. 

Similarily, to find the margin of point A, we take point B as the projected point of A. Formally, $x^{(i)} - \gamma^{(i)} w/\lvert\lvert w \rvert\rvert$. The point is on boundary, meaning that"

$$w^T(x^{(i)} - \gamma^{(i)} w/\lvert\lvert w \rvert\rvert) + b = 0$$

Solve:

$$\gamma^{(i)} = (w/\lvert\lvert w \rvert\rvert)^T x^{(i)} + b/\lvert\lvert w \rvert\rvert$$

Thus, **geometric margin** with respect to a training sample is defined as:

$$\gamma^{(i)} = y^{(i)}((w/\lvert\lvert w \rvert\rvert)^T x^{(i)} + b/\lvert\lvert w \rvert\rvert)$$

If $\lvert\lvert w \rvert\rvert = 1$, the functional margin is equal to geometric margin. Similarily, the geometric margin for all samples is:

$$\gamma = \min_{i=1,\dots,m}\gamma^{(i)}$$

## 3 Optimal Margin Classifier

The goal is to maximize the geometric margin.

For now, we assume that data is linearly separable. The optimization problem can be defined as :

$$\max_{\gamma,w,b} \gamma$$

$$ \text{s.t. } y^{(i)}(w^Tx^{(i)} + b) \geq \gamma, i = 1,\dots,m$$

$$\lvert\lvert w \rvert\rvert = 1$$

The nasty point is $\lvert\lvert w \rvert\rvert = 1$ constraint, which makes it is non-convex. 

We can then transform it to:

$$\max_{\overset{\wedge}{\gamma},w,b} \frac{\overset{\wedge}{\gamma}}{\lvert\lvert w \rvert\rvert}$$

$$ \text{s.t. } y^{(i)}(w^Tx^{(i)} + b) \geq \overset{\wedge}{\gamma}, i = 1,\dots,m$$

Basically, we relate geometric margin with function margine. Instead of geometric margin, we subject to a functional margin. **By doing this, we eliminate $\lvert\lvert w \rvert\rvert = 1$.** However, it is still bad. 

By scaling constraint on w and b, we do not change anything. We use this fact to make $\overset{\wedge}{\gamma} = 1$.And then, the max problem becomes a min problem now. That is,

$$\min_{\gamma,w,b} \frac{1}{2} \lvert\lvert w \rvert\rvert^2$$

$$ \text{s.t. } y^{(i)}(w^Tx^{(i)} + b) \geq 1, i = 1,\dots,m$$

The problem can be solved by using quadratic programming software. We can still go further to simplify this but it requires the knowledge of **Lagrange Duality**

## 4 Lagrange Duality

Let's take a side step on how to solve general **constrained optimizing problem.** 

Consider a problem such as :

$$\min_w f(w)$$

$$\text{s.t. } h_i(w) = 0,i = 1,\dots,l$$

Now, we can define **Lagrangian** to be:

$$\mathcal{L}(w,\beta) = f(w) + \sum\limits_{i=1}^l \beta_i h_i(w)$$

where $\beta_i$ is called **Lagrange multiplier.** Now, we can use partial derivative to set to zero and find out w and $\beta$

We can generalize to inequality and equality constraint. So we can define **primal** problem to be:

$$\min_w f(w)$$

$$\text{s.t. } g_i(w) \leq 0,i = 1,\dots,k$$

$$h_i(w) = 0,i = 1,\dots,l$$

We define **generalized Lagrangian** as:

$$\mathcal{L} = f(w) + \sum\limits_{i=1}^k \alpha_i g_i(w) + \sum\limits_{i=1}^l \beta_i h_i(w)$$

where all $\alpha$ and $\beta$ are Lagrangian multiplier. 

Let's define a quantity for primal problem as :

$$\theta_{\mathcal{P}}(w) = \max_{\alpha,\beta:\alpha_i\geq 0} \mathcal{L}(w,\alpha,\beta)$$

If some constraints are violated, then $\theta_{\mathcal{P}}(w) = \infty$

Thus, we have:

$$\theta_{\mathcal{P}}(w) = \begin{cases} f(w)  \text{, if w satisfy primal constraints} \\ \infty  \text{, otherwise} \\ \end{cases}$$

To match to our primal problem, w define the min problem as:

$$\min_w \theta_{\mathcal{P}}(w) = \min_w \max_{\alpha,\beta:\alpha_i\geq 0} \mathcal{L}(w,\alpha,\beta)$$

This is the same as the primal problem if all constrain are satisfied. We define the value of primal problem to be: $p^{\ast} = \min_w \theta_{\mathcal{P}(w)}$. Then, we define:

$$\theta_{\mathcal{D}}(\alpha,\beta) = \min_w \mathcal{L}(w,\alpha,\beta)$$

to be the dual part. To again match the primal problem, we define the **dual optimization problem** to be:

$$\max_{\beta,\alpha:\alpha_i\geq 0} = \max_{\alpha,\beta:\alpha_i\geq 0} \min_w \mathcal{L}(w,\alpha,\beta)$$

Similarily, the value of dual problem is $d^{\ast} = \max_{\alpha,\beta:\alpha_i\geq 0} \theta_{\mathcal{D}}(\alpha,\beta)$

The primal and dual problem is related by:

$$d^{\ast} = \max_{\alpha,\beta:\alpha_i\geq 0} \theta_{\mathcal{D}}(\alpha,\beta) \leq p^{\ast} = \min_w \theta_{\mathcal{P}(w)}$$

This is always true. The proof can be found online. The key is that under certain condition, they are equal. If they are equal, we can focus on dual problem instead of primal problem. 

We assume that f and g are all convex and h are affine(**When f has a Hessian, it is convex iff Hessian is positive semi-definite. All affine are convex. Affine means linear.**) and g are all less than 0 for some w. Wtih these assumptions, there must exist $w^{\ast}$ for primal solution and $\alpha^{\ast},\beta^{\ast}$ for dual solution and $p^{\ast} = d^{\ast}$. And $w^{\ast}$,$\alpha^{\ast}$ and $\beta^{\ast}$ satisfy **Karush-Kuhn-Tucker (KKT) conditions**, which says:

$$\frac{\partial}{\partial w_i}\mathcal{L}(w^{\ast},\alpha^{\ast},\beta_{\ast}) = 0. i = 1,\dots,n$$


$$\frac{\partial}{\partial \beta_i}\mathcal{L}(w^{\ast},\alpha^{\ast},\beta_{\ast}) = 0. i = 1,\dots,l$$

$$\alpha_i^{\ast}g_i(w^{\ast}) = 0,i = 1,\dots,k$$

$$g_i(w^{\ast}) \leq 0,i = 1,\dots,k$$

$$\alpha_i^{\ast} \geq 0,i = 1,\dots,k$$

Third euqaiton is called **KKT dual complementarity condition**. It means if $\alpha_i^{\ast} > 0$, then $g_i(w^{\ast}) = 0$.

## 5 Optimal Margin Classifier

Let's revisit the primal problem:

$$\min_{\gamma,w,b} \frac{1}{2} \lvert\lvert w \rvert\rvert^2$$

$$ \text{s.t. } y^{(i)}(w^Tx^{(i)} + b) \geq 1, i = 1,\dots,m$$

we can re-arrange the constraint to be:

$$g_i(w) = -y^{(i)}(w^Tx^{(i)} + b) + 1 \leq 0$$

where i spans all training samples. From KKT dual complementarity condition, we have $\alpha_i > 0$ only when the functional margin is 1 ($g_i(w) = 0$).

We can vistualize this in the picture below. The three points on the dash line are the ones with the smallest geometric margin which is 1. Thus, those points are the ones with positve $\alpha_i$ and are called **support vector**. 

![SVM Boundary](/images/svm_bound.png)

The Lagranian with only inequality constraint is:

$$\mathcal{L}(w,b,\alpha) = \frac{1}{2}\lvert \lvert w\rvert \rvert^2 - \sum\limits_{i=1}^m \alpha_i [y^{(i)}(w^Tx^{(i)} + b) - 1] \tag{1}$$

To find the dual form of this problem, we first find min of loss with respect to w and b for a fixed $\alpha$. To do that, we have:

$$\triangledown_{w}\mathcal{L}(w,b,\alpha) = w - \sum\limits_{i=1}^m \alpha_i y^{(i)}x^{(i)} = 0\tag{2}$$

$$w = \sum\limits_{i=1}^m \alpha_i y^{(i)}x^{(i)}tag{3}$$

$$\frac{\partial}{\partial b}\mathcal{L}(w,b,\alpha) = \sum\limits_{i=1}^m \alpha_i y^{(i)} = 0 \tag{4}$$

We take equation (3) back to equation (1) we have:

$$\mathcal{L} = (w,b,\alpha) = \sum\limits_{i=1}^m \alpha_i - \frac{1}{2}\sum\limits_{i,j}^m y^{(i)}y^{(j)}\alpha_i\alpha_j (x^{(i)})^Tx^{(j)} - b\sum\limits_{i=1}^m\alpha_i y^{(i)}$$

$$= \sum\limits_{i=1}^m \alpha_i - \frac{1}{2}\sum\limits_{i,j}^m y^{(i)}y^{(j)}\alpha_i\alpha_j (x^{(i)})^Tx^{(j)}$$

Thus, we have the dual problem as :

$$\max_{\alpha} W(\alpha) = \sum\limits_{i=1}^m \alpha_i - \frac{1}{2}\sum\limits_{i,j}^m y^{(i)}y^{(j)}\alpha_i\alpha_j (x^{(i)})^Tx^{(j)}$$

$$\text{s.t.} \alpha_i \geq 0, i = 1,\dots,m$$

$$\sum\limits_{i=1}^m \alpha_i y^{(i)} = 0$$

which satisfies KKT condition. It means we found out the dual problem to solve instead of primal problem. If we can find $\alpha$ from this dual problem, we can use equation (3) to find $w^{\ast}$. With optimal $w^{\ast}$, we can find $b^{\ast}$:

$$b^{\ast} = -\frac{\max_{i:y^{(i)}=-1}w^{\ast T}x^{(i)} + \min_{i:y^{(i)}=1}w^{\ast T}x^{(i)}}{2}$$

This is easy to verify. The optimal w and b will make the geometric margin of cloest negative and positive sample to be equal. 

The equation (3) says that the optimal w is based on the optimal $\alpha$. To make prediction, we have:

$$w^Tx + b = (\sum\limits_{i=1}^m \alpha_i y^{(i)} x^{(i)})^Tx + b = \sum\limits_{i=1}^m \alpha_i y^{(i)} <x^{(i)},x> + b$$

If it is bigger than zero, we predict one. We also know that $\alpha$ will be all zeros except for the support vectors. That means **we only cares about the inner product between x and support vector**. This makes the prediction faster and brings the **Kernel funciton** into the sight, which is for high dimensional space. 


## 6 Kernels

In the example of living area of house, we can use the feature $x.x^2,x^3$ to get cubic function. X is called **input attribute** and $x.x^2,x^3$ is called **features**. We dentoe $\phi (x)$ the feature mapping from attribute to features. 

Thus, we might want to learn inthe new feature space $\phi (x)$. In last section,we only need to calculate inner product $<x,z>$ and now we can replace it with $<\phi(x),\phi(z)>$. 

Formally, given a mapping, we denote **Kernel** to be:

$$K(x,z) = \phi(x)^T\phi(z)$$

We can use Kernel for the replacement instead of mapping itself. The reason is that Kernel is less expensive computationally. So we can learn in high dimensuional space without calculating mapping $\phi$.

An example of how effective it is can be shown in the notes. It should be noted that calculating mapping is exponential time complexity whereas Kernel is linear time. 

In another way, Kernel is a measurement of how close or how far it is between x and z. It indicates the similarity. One of the popular Kernel is called **Gaussian Kernel** defined as: 

$$K(x,z) = \exp(-\frac{\lvert\lvert x-z \rvert\rvert^2}{2\sigma^2})$$

We can use this as learning SVM and it corresponds to infinite dimensional feature mapping $\phi$. It also shows that it is impossible to calculate infinite dimensional mapping but we can use Kernel instead. 

Next, we are insterested in telling if a Kernel is valid or not. 

We define **Kernel Matrix** as $K_{ij} = K(x^{(i)},x^{(j)})$ for m points(i.e. K is m-by-m). Now, if K is valid, it means:

(1)Symmetric: $K_{ij} = K(x^{(i)},x^{(j)}) = \phi(x^{(i)})^T\phi(x^{(j)}) = \phi(x^{(j)})^T\phi(x^{(i)}) = K_{ji}$

(2)Positive semi-definite: $z^TKz \geq 0$ proof is easy. 

**Mercer Theorem: Let $K:\mathbb{R}^n \times \mathbb{R}^n \mapsto \mathbb{R}$ be given. Then for a Kernel to be valid, it is necessary and sufficient that for any $\{x^{(1)},\dots,x^{(m)}\}$, the corresponding kernel matrix is symmetric and postive semi-definite.**

Kernel method is not only used in SVM but also anywhere that inner product is used. So we can replace the inner product with Kernel so that we can work in a higher dimensional space. 

## 7 Regularization and Non-separable Case

Although mapping x to higher dimensional space increases the chance to be separable, it might not be case. An outlier could also be the cause that we actually don't want to include. An example of such a case can be shown below. 

![SVM outlier](/images/svm_outlier.png)

To make the algorithm work for non-linear case as well, we add **regularization** to it:

$$\min_{\gamma,w,b} \frac{1}{2}\lvert\lvert w\rvert\rvert^2 + C\sum\limits_{i=1}^m \xi_i$$

$$\text{s.t. } y^{(i)}(w^Tx^{(i)} + b) \geq 1-\xi_i,i=1,\dots,m$$

$$\xi_i \geq 0,i=1,\dots,m$$

It will pay the cost for the functional margin that is less than one. C controls how cost that would be. It says that:

(1) We want w to be small so that margine will be large. 

(2) We want most samples to have functional margin that is larger than 1. 

The Lagrangian is :

$$\mathcal{L}(w,b,\xi,\alpha,r) = \frac{1}{2}w^Tw + C\sum\limits_{i=1}^{m}\xi_i - \sum\limits_{i=1}^m \alpha_i[y^{(i)}(x^{(i)T}w + b) - 1 + \xi_i] - \sum\limits_{i=1}^{m}r_i\xi_i$$

where $\alpha$ and r are Lagrangian multipliers which must be non-negative. Now, we can set derivative on w,b and $\xi$ to zero and find w. Keep in mind that we try to take min in the dual problem so we do not want to give it chance to have anything like $-\infty$. Plugging back will produce the dual problem as:

$$\max_{\alpha} W(\alpha) = \sum\limits_{i=1}^{m}\alpha_i - \frac{1}{2}\sum\limits_{i,j=1}^{m}y^{(i)}y^{(j)}\alpha_i\alpha_j<x^{(i)},x^{(j)}>$$

$$\text{s.t. }0\leq \alpha_i \leq C,i=1,\dots,m$$

$$\sum\limits_{i=1}^{m}\alpha_i y^{(i)} = 0$$

Notice that we have an interval for $\alpha$ becuase it has $\sum\limits_{i=1}^{m}(C-\alpha_i-r_i)\xi_i$. We take derivative to $\xi$ set to zero and we can eliminate $\xi$.

Also notice that the optimal b is not the same anymore because the margin for both cloest points have changed. In next section, we will find the algrotihm to figure out the solution. 

## 8 The SMO Algorithm

The SMO(sequential minimal optimization) algorithm by John Platt is to solve the dual problem in SVM. 

### 8.1 Coordinate Ascent

In general, the optimization problem

$$\max_{\alpha}W(\alpha_1,\alpha_2,\dots,\alpha_m)$$

can be solved by gradient ascent and Newton's method. In addition, we can also use coordinate ascent:

{% highlight bash %}
for loop until convergence:
  for i in range(1,m):
    alpha(i) = argmin of alpha(i) W(all alpha)
{% endhighlight %}

Basically, we fix all the $\alpha$ except for $\alpha_i$ and then move to next $\alpha$ for updating. **The assumption is that calculating gradient to $\alpha$ is efficient.** An example can be shown below. 

![SVM coordinate](/images/svm_coordinate.png)

Note that the path of the convergence is always parallel to axis because it is updated one variable at a time. 

### 8.2 SMO

We cannot do the same thing in dual problem in SVM because varying only one variable might violate the constraint:

$$\alpha_1 y^{(1)} = -\sum\limits_{i=2}^m \alpha_i y^{(i)}$$

which says once we determine the rest of $\alpha$, we cannot vary the left $\alpha$ anymore. Thus, we have to vary two $\alpha$ at one time and update them. For exmaple, we can have:

$$\alpha_1 y^{(1)} + \alpha_2 y^{(2)} = -\sum\limits_{i=3}^m \alpha_i y^{(i)}$$

We make right side to be constant:

$$\alpha_1 y^{(1)} + \alpha_2 y^{(2)} = \zeta$$

which can be pioctorially shown as:

![SVM coordinate](/images/svm_two_coord.png)

Note that although it is a square where $\alpha$ can lie but with a straight line, we might have a lower bound and upper bound on them. 

We can rewrite the above equation by multiplying $y^{(1)}$ on both sides:

$$\alpha_1 = (\zeta - \alpha_2 y^{(2)})y^{(1)}$$

Then, W will be :

$$W(\alpha_1,\dots,\alpha_m) = W((\zeta-\alpha_2 y^{(2)})y^{(1)},\alpha_2,\dots,\alpha_m)$$

We treat all other $\alpha$ as constants.Thus, after plugging in, W will become quadratic, which can be written as $a\alpha_2^2 + b\alpha_2 + c$ for some a, b and c. 

Last, we define $\alpha_2^{new, unclipped}$ as the current solution to update $\alpha_2$. Thus, with applying constraints, only for this single variable, we can write:

$$\alpha_2^{new} = \begin{cases} H  \text{, if          }\alpha_2^{new, unclipped}>H \\ \alpha_2^{new, unclipped}  \text{, if } L\leq \alpha_2^{new, unclipped} \leq H \\ L  \text{, if          } \alpha_2^{new, unclipped} < L \\ \end{cases}$$

# Bias-Varaince and Error Analysis

## 1 The Bias-Varaince Tradeoff

Let's denote $\overset{\wedge}{f}$ be the model that is trained on some dataset and $y$ be the ground truth. Then, the mean squared error(MSE) is defined:

$$\mathbb{E}_{(x,w)\sim \text{test set}} \lvert \overset{\wedge}{f}(x) - y \rvert^2$$

We have three explanation for a high MSE:

**Overfitting:** The model does not generalize well and probably only works well in training dataset.

**Underfitting:** The model does not train enough or have enough data for training so does not learn a good representation. 

**Neither:** The noise of data is too high. 

We formulate these into **Bias-Varaince Tradeoff**.

Assume that samples are sampled from similar distribution which can be defined as:

$y_i = f(x_i) + \epsilon_i$ where the noise $\mathbb{E}[\epsilon] = 0$ and $Var(\epsilon) = \sigma^2$.

Whereas our goal is to compute f, we can only obtain an estimate by looking at training samples generated from above distribution. Thus, $\overset{\wedge}{f}(x_i)$ is random since it depends on $\epsilon_i$ which is random and it is also the prediction of $y = f(x_i) + \epsilon_i$. Thus, it makes sense to get $\mathbb{E}(\overset{\wedge}{f}(x)-f(x)$.

We can now calculate the expected MSE:

$$\begin{align}
\mathbb{E}[(y-\overset{\wedge}{f}(x))^2] &= \mathbb{E}[y^2 + (\overset{\wedge}{f})^2 - 2y\overset{\wedge}{f}]\\
&= \mathbb{E}{y^2} + E[(\overset{\wedge}{f})^2] - \mathbb{E}[2y\overset{\wedge}{f}] \\
&= Var(y) + Var(\overset{\wedge}{f}) + (f^2 - 2f\mathbb{E}[\overset{\wedge}{f}] + (\mathbb{E}[\overset{\wedge}{f}])^2\\
&= Var(y) + Var(\overset{\wedge}{f}) + (f - \mathbb{E}[\overset{\wedge}{f}])^2\\
&=\sigma^2 + \text{Bias}(f)^2+ Var(\overset{\wedge}{f})
\end{align}$$

The fisrt term is data noise which we cannot do anything. A high bias term means the model does not learn efficiently and is underfitting. A high variance means that the model does not generalize well and is overfitting. 

## 2 Error Analysis

To analyze a model, we should first build a pipeline of the interests. Then, we start from plugging ground truth for each component and see how much accuracy that change makes on the model. An example can be seen below. 

![Error Analysis](/images/err_ana.png)

## 3 Ablative Analysis

Whereas error analysis tries to recognize the difference between current performance and perfect performance, Ablative Analysis tries to recognize that between baseline and current model. 

For example, assume that we have more add-on features that makes the model perform better. We want to see how much performance it will be reduced by eliminating one add-on feature at a time. An example can be shown below. 

![Ablative Analysis](/images/ablative_ana.png)

# Learning Theory

## 1 Bias-Varaince tradeoff

This has already been discussed in last section. In additon, we should emphasize that:

1) A simple model with few parameters should have low variance on its prediction but will produce a **high bias** in general. 

2) A complex model with too many parameters to train usually have a **high variance** although it can predict well(low bias).

There is a tradeoff in between. 

## 2 Preliminaries

The key idea is to formalize the analysis on a machine learning algorithm. For example, is there a bound on generalization error? Is there condition on that limit? How can we select a model over others? This is what learning theory talks about. 

We start off by introducing two lemmas:

**Lemma** (The Union Bound) Let $A_1,A_2,\dots,A_k$ be k different events, which might not be independent. Then:

$$P(A_1\cup\dots\cup A_k)\leq P(A_1)+ \dots + P(A_k)$$

This, Union Bound, is a comment used axiom in learning theory. The proof can be easily found online. 

**Lemma** (Hoeffding Inequality) Let $Z_1,Z_2,\dots,Z_m$ be m iid random variables drawn from Bernoulli($\phi$) distribution. That is: $P(Z_i = 1) = \phi, P(Z_i = 0) = 1 - \phi$. Let $\hat{\phi} = \frac{1}{m}\sum_{i=1}^m Z_i$ be the mean of random variable and let $\gamma > 0$ be fixed. Then:

$$P(\lvert \phi - \hat{\phi}\rvert >\gamma)\leq 2\exp(-2\gamma^2m)$$

This is a.k.a **Chernoff bound**. Think about what it says. If we take the mean of r.v. as the estimation of future input, then the probability of the discrepance between truth and the estimation larger than a threshold is less than a value related with the number of training samples and the threshold as well. This is frequently used in learning theory. 

This lemma can be generalized to multi-class classification as well but we just focus on binary case for now. Assume that we have a training set S with m pairs of sample. Each $(x^{(i)},y^{(i)}$ pair are drawn iid from some distribution $\mathcal{D}$. For a hypothesis h, we define the **training error** or **empirical risk** to be:

$$\hat{\varepsilon}(h) = \frac{1}{m}\sum\limits_{i=1}^m \mathbb{1}\{h(x^{(i)}\neq y^{(i)}\}$$

We also define the **generalization error** to be:

$$\varepsilon(h) = P_{(x,y)\thicksim \mathcal{D}}(h(x)\neq y)$$

This quantity shows how much probability of misclassification will be if we sample one pair from the distribution $\mathcal{D}$. The above concepts are often related to **probably approximately correct(PAC)** problem, which has two most important assumption:(1)training and testing samples are from the same distribution. (2) each pair of sample is iid. **In short, the empirical risk is the error resulted from training data that we are currently holding, whereas the generalization error is the error resulted from samples drawn from the same distribution as training dataset.**

Think about linear classification again. We can let $h_{\theta}(x) = \mathbb{1}\{\theta^Tx\geq 0\}$ to be our hypothesis. The goal is to find $\theta$ which can minimize the training error. Formally,

$$\hat{\theta} = \arg\min_{\theta}\hat{\varepsilon}(h_{\theta})$$

This is called **empirical risk minimization(ERM)**. The output hypothesis is $h_{\hat{\theta}}$. The ERM is the core idea of learning algorithm. Logisitic regression problem can also be analog to this algorithm. 

In learning theory, we do not want to restrict the hypothesis to a linear classifier or so. We want it to point out a general hypothesis form. Thus, we define the **hypothesis class $\mathcal{H}$** to be the set of all classifiers in the case. In this set, all valid classifiers are considered against to a evaluation scheme. Concequently, ERM can be treated as the minimization over the class of functions $\mathcal{H}$. Formally:

$$\hat{h} = \arg\min_{h\in\mathcal{H}}\hat(\varepsilon)(h)$$


## 3 The Case of Finite $\mathcal{H}$

To begin with, we first consider the case where the number of hypothesis classes is finite, dentoed $\mathcal{H} = \{h_1,h_2.\dots,h_k\}$ for k hypotheses. Each hopythosis is just a mapping function which takes $\mathcal{x}$ as input and map to either 1 or 0 and ERM algorithm is just to select the hypothesis which produces minimum training error, namely $\hat{h}$.

So now the question is what we can say about generalization error on $\hat{h}$. For exmaple, can we give a bound on the error? If so, it implies that in any circumstances, the error rate would not exceed the bound we derived. To do so, we need (1) to show $\hat{\varepsilon}(h)$ is a good estimate of $\hat{\varepsilon}(h)$ for all h. (2) to show that this implies an upper-bound on the generalization error of $\hat{h}$.

We pick $h_i$ from $\mathcal{H}$ and denote $Z=\mathbb{1}{h_i(x) \neq y}$ where $(x,y)\thicksim\mathcal{D}$. Basically, Z indicates if $h_i$ misclassifies it. And we also denote $Z_j = \mathbb{1}{h_i(x^{(i)}) \neq y^{(i)}}$. Note that since all samples are drawn from D, thus $Z$ and $Z_i$ have the same distribution. 

We should notice that $\mathbb{E}[Z] = \mathbb{E}[\mathbb{1}{h_i(x) \neq y}] = P_{(x,y)\thicksim \mathcal{D}}(h(x)\neq y) = \varepsilon(h)$ which also applies for $Z_j$. It represents the probability of misclassification on a random sample. Moreover, the training error can be written:

$$\hat{\varepsilon}(h_i) = \frac{1}{m}\sum\limits_{j=1}^m Z_j$$

We can see that $\hat{\varepsilon}(h_i)$ is exactly the mean of m random variables $Z_j$ drawn iid from Bernoulli distribution with mean $\varepsilon(h_i)$. We can apply Hoeffding inequality as:

$$P(\lvert\varepsilon(h_i)-\hat{\varepsilon}(h_i)\rvert > \gamma)\leq 2\exp(-2\gamma^2m)$$

This means that for a particular $h_i$ with high probablity the empirical error will be close to generalization error. which is nice. The more valueable point is to prove this is true for all $h\in\mathcal{H}$. 

To do this, we denote $A_i$ be the event that $\lvert\varepsilon(h_i) - \hat{\varepsilon}(h_i)\rvert>\gamma$. In the above, we have proved that for a particular $A_i$, it is true that $P(A_i)\leq 2\exp(-1\gamma^2m)$. With union bound, we have:

$$\begin{align}
P(\exists h_i\in \mathcal{H}.\lvert \varepsilon(h_i)-\hat{\varepsilon}(h_i)>\gamma) &= P(A_1\cup A_2\cup\dots\cup A_k)\\
&\leq \sum\limits_{i=1}^k P(A_i) \\
&\leq \sum\limits_{i=1}^k 2\exp(-2\gamma^2m)\\
&= 2k\exp(-2\gamma^2m)
\end{align}$$

If we substract 1 from both sides, we have:

$$\begin{align}
P(\neg\exists h_i\in \mathcal{H}.\lvert \varepsilon(h_i)-\hat{\varepsilon}(h_i)>\gamma) &= P(\forall h_i\in \mathcal{H}.\lvert \varepsilon(h_i)-\hat{\varepsilon}(h_i)\leq\gamma)\\
&\geq 1-2k\exp(-2\gamma^2m)
\end{align}$$

This simply says that with probability at least $1-2k\exp(-2\gamma^2m)$, we have generalization error to be within the bound of empirical error for all $h\in \mathcal{H}$. It is called **uniform convergence result**.  

In this case, we are really interested in 3 quantities:$m,\gamma$ and probability of error, denoted as $\delta$. The reason that we are interested in these three variables is because they are correlated in some way. For example, given $\gamma$ and some $\delta>0$, we can find m by solving $\delta = \2k\exp(-2\gamma^2m):

$$m\geq\frac{1}{2\gamma^2}\log\frac{2k}{\delta}$$

This quantity says about how many training samples are required to make the bound valid, which is only logarithmic in k. It is also called **sample complexity**.

Similarly, given m and $\delta$, we can solve for $\gamma$ and we will get:

$$\lvert\hat{\varepsilon}(h) - \varepsilon(h)\rvert\leq\sqrt{\frac{1}{2m}\log\frac{2k}{\delta}}$$

Assume that the uniform convergence holds for all hypotheses, can we also bound the generalization error on $\hat{h}=\arg\min_{h\in\mathcal{H}}\hat{\varepsilon(h)}$?

Define $h^{ast} = \arg\min_{h\in\mathcal{h}}\varepsilon(h)$ to be the best possible hypothesis. We are trying to compare the hypothesis which achieves the best in training data and that which does the best in generalization error theorectically. We have:

$$\begin{align}
\varepsilon(\hat{h}) &\leq \hat{\varepsilon}(hat{h}) + \gamma\\
&\leq \hat{\varepsilon}(h^{\ast}) + \gamma\\
&\leq \varepsilon(h^{\ast}) + 2\gamma
\end{align}$$

The first line is by definition $\lvert \varepsilon(\hat{h}) -\hat{\varepsilon}(\hat{h})\rvert \leq gamma$, which is similar for the third line as well. From this proof, we have shown that if uniform convergence occurs, the generalization error of empirically selected h is at most $2\gamma$ worse than per generalization-error selected hypothesis in $\mathcal{H}$.

**Theorem** Let $\lvert \mathcal{H} \rvert = k$,and let $m,\delta$ be fixed. Then with probability at least $1-\delta$, we have:

$$\varepsilon(\hat{h})\leq \bigg(\min_{h\in\mathcal{H}}\varepsilon(h)\bigg)+2\sqrt{\frac{1}{2m}\log\frac{2k}{\delta}}$$

This is related to bias-variance tradeoff as well. Assum that we have a larger hypothesis class $\mathcal{H}^{\prime}$ where $\mathcal{H} \supseteq \mathcal{H}^{\prime}$. If we learn on the new hypothesis class, we have a bigger k. Thus, the second term above will be larger. That is the variance will be larger. However, the the first term is smaller. That is the bias will go down. 

**Corollary** Let $\lvert\mathcal{H}\rvert=k$ and given $\delta,gamma$, then for $\varepsilon(\hat{h})\leq\min_{h\in\mathcal{H}}\varepsilon(h) + 2\gamma$ to hold with probability at least $1-\gamma$, it suffices that:

$$m\geq\frac{1}{2\gamma^2}\log\frac{2k}{\delta} = O\bigg(\frac{1}{\gamma^2}\log\frac{k}{\delta}\bigg)$$

## 4 The Case of Infinite $\mathcal{H}$

In section 3, it shows that finite hypothesis class owns several convenient theorems we can directly bound the generalization error. However, many hypothesis class such as linear regression parameterized by real numbers contain an infinite number of functions since real numbers lie in continuous space. So can we also give the bound for such a case?

For intuition, imagine that we want to parameterize the model with d parameters with double type. That is for a single hypothesis we need 64d bits to represent. Totally, we can have $2^{64d}$ hypotheses. From last Corollary, we know that for uniform convergence to hold, we need $m\geq O\bigg(\frac{1}{\gamma^2}\log\frac{2^{64f}}{\delta}\bigg) = O_{\gamma,\delta}(d)$. This means the samples that we need is in linear relationship with d which is the number of model parameters. 

However, this intuition is not technically correct. We could also have 2d parameters for the same hypothesis class which has the set of linear classifer in n dimension. Thus,we need to seek for more technical definition. 

Let's define $S=\{x^{(1)},x^{(2)},\dots,x^{(d)}\}$ to be a set of points in any dimension. We say that $\mathcal{H}$ **shatters** S if $\mathcal{H}$ can realize any labeling on S. That is, for any possible set of $\{y^{(1)},y^{(2)},\dots,y^{(d)}\}$, there exists some $h\in\mathcal{H}$ so that $h(x^{(i)})=y^{(i)}$.

Given a hypothesis class, we define its **Vapnik-Chervonenkis dimension(VC-dimension)** to be the largest set that is shattered by $\mathcal{H}$. $VC(\mathcal{H}) = \infty$ means it can shatter any arbitrarily large sets.

For instance, we consider the case of three points:

![Three Points in 2D](/images/cs229_learningtheory_vc1.png)

So assume that we have a hypothesis class in 2D, can this hypothesis class classify any possible labeling for these three points or shatter them? Given that $h(x) = \mathcal{1}\{\theta_0+\theta_1x_1+\theta_2x_2\}$, we can enumerate all possible labeling for these three points and draw a straight line in 2D to perfectly classify them. That is:

![All Possible Classifier](/images/cs229_learningtheory_vc2.png)

Moreover, we can prove that there is no set of 4 points that can be shattered by this hypothesis class. Thus, $VC(\mathcal{H})$=3$. 

Note that not all possible of set of 3 points can be shattered by this hypothesis class even if VC dimension is 3. The below is an example for such as case where this hypothesis class failed to classify them. 

![Special Case](/images/cs229_learningtheory_vc3.png)

Thus, to prove that some hypothesis class has VC dimension at least d, we just need to it can shatter at least one set of points with size d. 

**Theorem** Let $\mathcal{H}$ be given and $d = VC(\mathcal{H}$. Then with porbability at least $1-\delta$, we have that for all $h\in\mathcal{H}$:

$$\lvert\varepsilon(h) - \lvert\hat{\varepsilon}(h)\rvert\leq O\bigg(\sqrt{\frac{d}{m}\log\frac{m}{d}+\frac{1}{m}\log\frac{1}{\delta}}\bigg)$$

Thus, with probabilyt at least $1-\delta$, we also have that:

$$\varepsilon(\hat{h}) \leq \varepsilon(h^{\ast}) +  O\bigg(\sqrt{\frac{d}{m}\log\frac{m}{d}+\frac{1}{m}\log\frac{1}{\delta}}\bigg)$$

It means that if a hypothesis class has a finite VC dimension, the uniform convergence happens as m goes large. We use similar derivation from finite hypothesis class to give a bound on $\varepsilon(\hat{h})$ in terms of $\varepsilon(h^{\ast})$.

**Corollary** For $\lvert\varepsilon(h) - \hat{\lvert\varepsilon(h)}\rvert\leq\gamma$ to hold for all $h\in\mathcal{H}$ with probability at least $1-\delta$, it suffices that $m=O_{\gamma,\delta}(d)$ where d is the VC dimension. 

In short, the number of training samples we need to train well using a particular $\mathcal{H}$ is linear in the VC dimension of $\mathcal{H}$. In general, VC dimension for most hypothesis classes is approximately linear in the number of parameters. So the number of needed traning samples is also related to the number of parameters. 

# Regularization and Model Selection

In model selection, if we have k parameters in the model, the quesiton is what k should be?0,1,or 10?Which does one of them give the best bias-varaince tradeoff. In particular, we use a finite set of models $\mathcal{M} = \{M_1,M_2,\dots,M_d\}$ from which we try to select the best. Each model in the set contains either different parameterization of a particular model or different models. 

## 1 Cross Validation

Imagine that given a dataset S and a set of models, it is easy to think to select a model out of the set by:

1 Training each model $M_i$  from S and get the hypothesis $h_i$.

2 Pick the hypothesis with the smallest training error. 

This pipeline does not work simply because the higher order of the polynomial you choose, the better it will fit for the training set. However, the model you select will have a high generalizaton error in a new dataset. That is, it will be high variance.

In this scenario, **hold-out cross validation** will do a better work as:

1 Randomly split S into training set $S_{tr}$ and validation set $S_{cv}$ with 70% and 30% respectively

2 Train each $M_i$ on $S_{tr}$ to get hypothesis $h_i$

3 Select the hypothesis which has the smallest epirical error on the $S_{cv}$, which denotes $\hat{\varepsilon}\_{S_{cv}}(h_i)$

By doing the above, we try to estimate the real generalization error by testing the model on validation set. In step 3, after selecting the best model, we can retrain the model on the entire dataset again to generate the best hypothesis. However, even though that's the case, we still select the model based on 70% dataset. This is bad when data is scarce. 

Thus, we introduce the K-fold corss validation as:

1 Randomly split S into k disjoint subsets of m/k samples each. Denote $S_1,S_2,\dots,S_k$

2 For each model $M_i$, we unselect one of subsets dentoed j, and train the model on the rest of data to get the hypothesis $H_{ij}$. We test the hypothesis on $S_j$ and get $\varepsilon_{S_j}(h_{ij})$. We do this for all j. And lastly, we take average the generalization error over j.

3 We select the model with the smallest averaged generalization error. 

A typical choice for k is 10. This is computationally expensive although it gives the best performance. If the data is scarce, we might set k=m. In this case, we leave one sample at a time. We call it **leave-ont-out cross validation**. 

## 2 Feature Selection

If we have n features and m samples where $n \gg m$ (VC dimension is O(n)), we might have overfitting. In this case, you might want to select some of features which might be the most important. In brute force algorithm, we can have $2^n$ different combinations of feature setting. We can perform model selection over all $2^n$ possible models. This is too expensive to deal with. Thus, we have an option called **forward search** algorithm:

1 We initialize $\mathcal{F} = \emptyset$

2 Repeat: (a)for $i =1,\dots,n$ if $i\notin\mathcal{F}$, let $\mathcal{F}_i = \mathcal{F}\cup\{i\}$ and use some corss validation algorithm to evaluate $\mathcal{F}_i$. (b)Set $\mathcal{F}$ to be the best feature subset from (a)

3 Select the best feature subset from the above. 

You can terminate the loop by setting the number of features you like to have. In contrast, we can also have **backward search** in for feature selection, which is similar wtih the section of **Ablative Analysis**. However, both of them are computationally expensive since it requires $O(n^2)$ in time complexity. 

Instead, we can use **Filter feature selection** heristically. The idea is to give a score to how informative each feature is with respect to labels y. Then we pick the best out of it. 

One intuitive option of the sorce is to compute the correlation between each feature $x_i$ and y. In practice, we set the score to be **mutual information** as:

$$MI(x_i,y) = \sum\limits_{x_i\in\{0,1\}}\sum\limits_{y\in\{0,1\}} p(x_i,y)\log\frac{p(x_i,y)}{p(x_i)p(y)}$$

where we assume each feature and label is binary-valued and the summation is over the domain of the varaibles. Each probability can be calculated empirically from the training dataset. To understand this, we know that:

$$MI(x_i,y) = KL(p(x_i,y)\lvert\lvert p(x_i)p(y))$$

where KL is **Kullback-Leibler divergence**. It simply measures how different the probability distributions from both sides of the two bars are. If $x_i$ and $y$ are independent, then KL is 0. That means there is no relationship between this feature and labels. In contrast, if we have a high score of MI, then such a feature is strongly correlated with labels. 

## 3 Bayesian Statistics and regularization

In the previous sectiosn, we talk about the maximum likelihood (ML) algorithm to fit model parameters as:

$$\theta_{ML} = \arg\max\prod_{i=1}^m p(y^{(i)}\lvert x^{(i)},\theta)$$

In this case, we viewed $\theta$ as a unknown parameter. It already exists there and just happens to be unknown. So our job is to find the unknown or estimate it. 

On the other hand, we can have a Bayesian view of this goal. We think the unknown parameter $\theta$ is also random. Thus, we place our prior belief on this parameter. We call it **prior distribution**. Given the prior distribution, we can calculate the posterior with dataset S as :

$$p(\theta\lvert S) = \frac{p(S\lvert\theta)p(\theta)}{p(S)} = \frac{\prod_{i=1}^m p(y^{(i)}\lvert x^{(i)},\theta)(p(\theta)}{\int_{\theta}\prod_{i=1}^m p(y^{(i)}\lvert x^{(i)},\theta)(p(\theta)d\theta}$$

For prediciton inference by using the posterior, we have:

$$p(y\lvert x,S) = \int_{\theta}p(y\lvert x,\theta)p(\theta\lvert S)d\theta$$

At this point, we can calculate the conditional expected value y. However, it is really hard to calculate the posterior in closed form since the intergral in the denominator cannot be solve in closed form. Thus, alternatively, we seek for a point estimate for the posterior at which it will give us one best $\theta$ for the posterior. The **MAP(maximum a posteriori)** can estimate it by:

$$\theta_{MAP} = \arg\max_{\theta} = \prod_{i=1}^m p(y^{(i)}\lvert x^{(i)},\theta)p(\theta)$$

In general, the prior is usually zero mean and unit variance. This will make MAP less susceptiable overfitting than the ML estimate of the parameters. 

# Online Learning and Perceptron Algorithm

We have talked about the learning paradigm where we feed a batch of training data to train a model. This is called **batch learning**. In this section, we think about the scenario where the model has to make prediction while it is continously learning on the go. This is called **online learning**.

In this scenario, we have a sequnce of examples $(x^{(1)},y^2{(1)}),(x^{(2)},y^2{(2)}),\dots,(x^{(n)},y^2{(n)})$. What online learning does is to first feed $x^{(1)}$ to the model and ask model to predict, and then show $y^{(1)}$ to the model to let the model perform learning process on it. We do this for one pair of training samples at a time. Eventually, we can come up with a model which has gone through the training dataset. What we are interested in is how many errors this model makes while in online learning process. This is heavily related to the knowledge from learning theory we have discussed before. 

Now, we can take perceptron algorithm as an example. We define $y\in\{-1,1\}$ for the label classes. Perceptron algorithm makes prediction based on:

$$h_{\theta}(x) = g(\theta^{T}x)$$

where:

$$g(z) = \begin{cases} 1  \text{, if } z \geq 0 \\ -1  \text{, otherwise} \\ \end{cases}$$

Then the model makes the update to its parameters as:

$$\theta_t = \theta_{t-1} + (h_{\theta}-y)x$$

We can see that if the prediction is correct, we make no change to the parameters. Then, we have the following theorem for the bound on the number of errors made in the online process. 

**Theorem** Let a sequence of examples $(x^{(1)},y^2{(1)}),(x^{(2)},y^2{(2)}),\dots,(x^{(n)},y^2{(n)})$ be given. Suppose that $\lvert\lvert x^{(i)}\rvert\rvert\leq D$ for all i, and further that there exists a unit-length vector u ($\lvert\lvert u\rvert\rvert_2=2$) such that $y^{(i)}(u^Tx^{(i)}\geq \gamma$ for all examples in the sequence(i.e., $u^Tx^{(i)}\geq \gamma$ if $y^{(i)}=1$ and $u^Tx^{(i)}\leq -\gamma$ if $y^{(i)}=-1$ so that u separates the data with the margin at least $\gamma$). Then the total number of mistakes that the perceptron algorithm makes on this sequnece is at most $O(D/\gamma)^2$.

**Proof**. Perceptron is an online learning algorithm. That means it will feed one pair of samples at a time. We also know that perceptron algorithm only updates its parameters when it makes a mistake. Thus, let $\theta^k$ be the weights that were being used for k-th mistake. We initialize from zero vector. Thus, $\theta^1 = \overrightarrow{0}$. In addition, when we make a mistake on i-th iteration, then $g((x^{(i)})^T\theta^k)\neq y^{(i)}$. This is saying:

$$(x^{(i)})^T\theta^k y^{(i)} \leq 0$$

The update rule is $\theta^{k+1} = \theta^k + y^{(i)}x^{(i)}$. We can multiply it by u to have:

$$(\theta^{k+1})^Tu = (\theta^k)^Tu + y^{(i)}(x^{(i)})^Tu \geq (\theta^k)^Tu + \gamma$$

This triggers inductive calculation, which says:

$$(\theta^{k+1})^Tu \geq k\gamma$$

On the other hand, we have:

$$\begin{align}
\lvert\lvert \theta^{k+1}\rvert\rvert^2 &= \lvert\lvert \theta^k + y^{(i)}x^{(i)}\rvert\rvert^2\\
&=  \lvert\lvert\theta^k\rvert\rvert^2 + 2y^{(i)}(x^{(i)})^T\theta^k + \lvert\lvert x^{(i)}\rvert\rvert^2\\
&\leq \lvert\lvert\theta^k\rvert\rvert^2 + \lvert\lvert x^{(i)}\rvert\rvert^2 \\
&\leq \lvert\lvert\theta^k\rvert\rvert^2 + D^2
\end{align}$$

The third step is because last term in step 2 is a negative. Similarly, we can apply induction here to get:

$$\lvert\lvert \theta^{k+1}\rvert\rvert^2 \leq kD^2$$

Now, we combine everything to get:

$$\begin{align}
\sqrt{k}D &geq \lvert\lvert \theta^{k+1}\rvert\rvert\\
&geq  (\theta^{k+1})^Tu\\
&\leq k\gamma
\end{align}$$

We have second step because u is unit length vector so the product of the norms is greater than the dot product of the two. This means $k\leq (\frac{D}{\gamma})^2$. Note that this bound does not involve in the number of training samples. So the number of mistakes perceptron made is only bounded by D and $\gamma$. 

# Deep Learning

## 1 Nerual Networks

Recall that in the housing price prediction example. We take the size of the house as input and make predictions on price by fitting a straight line. The problem in this model is that a straight line has mathematical meaning in negative domain, which does not make sense in predicting house values. Thus, we need to perform some link function to get a plot like the one below. 

![Link Function](/images/cs229_deeplearning_link.png)

Mathematically, we want $f:x\rightarrow y$. To prevent a negative prediction, we can have a single neuron where $f(x) = \max(ax+b,0)$ for some a and b from training process. This is called ReLU (rectified linear unit) function. This is essentially the simplest neuron that we can have. We can also stack multiple neurons where the output of a neuron can serve as the input of the other. This can give us a more complex struture. 

In the housing price prediction example, we can have multiple input such as the size of house, the number of bedrooms, the zip code and the wealth of neighborhood. We can take these features as input to the neural network. In addition, we might also find out that the size of house and the number of bedrooms are related to family size, the zip code is related to the walkable distance to stores and the wealth of neighborhoods are related to the quality of life around. Thus, we can futher say that the price of house depends more directly on these three factors. Such an idea can be realized by stacking several neurons together as:

![Neuron Networks](/images/cs229_deeplearning_nn.png)

Part of magic of neural networks is that we only need to feed the network with input features x and output prediction y. Everything else is called hidden units and figured out by the neural network itself. They are called hidden layers since we do not have ground truth for those nuerons and we ask network to solve for us. We cal this **end-to-end learning**. The last thing required is a large amount of training samples. The model will figure out the latent features that are helpful on prediction. Since human cannot understand the features it has produced, this renders the neural network as **black box** technology.

Before going to details, let's denote $x_i$ as i-th input feature, $a_j^{[\ell]}$ as the activition output at j-th unit in layer $\ell$, $foo^{\ell}$ as everything associated with layer $\ell$ and $z=\theta^Tx$. We can draw a diagram for a single neuron for illustration as:

![Neuron Networks](/images/cs229_deeplearning_neuron.png)

For the choice of activition functions, we can logistic function as before:

$$g(x) = \frac{1}{1 + \exp(-w^Tx)}$$

In addition, we can have more:

$$g(z) = \frac{1}{1+\exp(-z)}\quad\text{sigmoid}$$

$$g(z) = \max(z,0)\quad\text{RelU}$$

$$g(z) = \frac{\exp(z) - \exp(-z)}{\exp(z) + \exp(-z)}\quad\text{tanh}$$

Back to neuron network of price prediction, what it does for first hidden unit at first hidden layer is:

$$z_1^{[1]} = W_1^{[1]}x + b_1^{[1]} \quad \text{and} \quad a_1^{[1]} = g(z_1^{[1]})$$

where W is parameter matrix and $W_1$ is first row of it and b is a scalar. Similarly, we can have:

$$z_2^{[1]} = W_2^{[1]}x + b_2^{[1]} \quad \text{and} \quad a_2^{[1]} = g(z_3^{[1]})$$

$$z_3^{[1]} = W_3^{[1]}x + b_3^{[1]} \quad \text{and} \quad a_3^{[1]} = g(z_3^{[1]})$$

So the output of first layer from activition function can be defined as:

$$a^{[1]} = \begin{bmatrix} a_1^{[1]}\\ a_2^{[1]} \\ a_3^{[1]}  \\ a_4^{[1]} \end{bmatrix}$$

For some of tasks, we might not want to use ReLU although it is really popular in research simply becuase it is not always correct that we should have non-negative value for prediction. 

## 2 Vectorization

Now, a natural question to ask is that what the activation does and what if I remove it. Intuitively, activition functions are the key part of making deep learning work and making it possible to model non-linear relationships. Without it, what neural network does simply becomes linear combinations between weights and its input. Let's see how mathematically we can prove this. 

In the previous section, we calculate each $z_i^{[1]}$ and apply activation function for each of them. We can put all of them into a matrix and take advantage of matrix calculations to speed up this process. 

### 2.1 Vectorizaing the Output Computation

So for the first layer, we can have:

$$\underbrace{\begin{bmatrix} z_1^{[1]}\\ z_2^{[1]} \\ z_3^{[1]}  \\ z_4^{[1]} \end{bmatrix}}_{z^{[1]}\in\mathcal{R}^{4\times 1}} = \underbrace{\begin{bmatrix} -(W_1^{[1]})^T-\\ -(W_2^{[1]})^T- \\ -(W_3^{[1]})^T-  \\ -(W_4^{[1]})^T- \end{bmatrix}}_{W^{[1]}\in\mathcal{R}^{4\times 3}}\underbrace{\begin{bmatrix} x_!\\ x_2  \\ x_3 \end{bmatrix}}_{x\in\mathcal{R}^{3\times 1}} + \underbrace{\begin{bmatrix} b_1^{[1]}\\ b_2^{[1]} \\ b_3^{[1]}  \\ b_4^{[1]} \end{bmatrix}}_{b^{[1]}\in \mathcal{R}^{4\times 1}} $$

The dimenion of each matrix is labelled below. In short, it is $z^{[1]} = W^{[1]}x + b^{[1]}$, which is linear relationship. Then, we can apply activition function on z vector like sigmoid function for example. Similarly, we can use matrix to represent the propagation from first layer to secon layer. **As you can see here, without non-linear activition function, we simply do linear regression here, which cannot model many complicated non-linear relationship.**

### 2.2 Vectorization over Training Examples

Now, we want to do this thing for all the training samples that going to be fed into neural network. We want to do it at one time. So we define a sample matrix:

$$X = \begin{bmatrix} \lvert & \lvert & \lvert\\ x^{(1)} & x^{(2)} & x^{(3)} \\ \lvert & \lvert & \lvert \end{bmatrix}$$

So we can get the outpout as :

$$Z^{[1]} = \begin{bmatrix} \lvert & \lvert & \lvert\\ z^{[1](1)} & z^{[1](2)} & z^{[1](3)} \\ \lvert & \lvert & \lvert \end{bmatrix} = W^{[1]}X + b^{[1]}$$

Meanwhile, we also (as always) need to define the objective function that we want to maximize. For binary class, we can have the objective function as :

$$\sum\limits_{i=1}^m \big( y^{(i)}\log a^{[2] (i)} + (1 - y^{(i)})\log (1 - a^{[2] (i)})\big)$$

where $a^{[2] (i)}$ is the output from second layer (also the final layer) for i-th training sample. Remember that we are trying to model a binary problem, which is usually a Bernoulli. Thus, the output from neural network should be in class 1 with probability $a^{[2] (i)}$. We take log for this Bernoulli and you will get the above with math manipulation. 

We can use gradient ascent for updating. 

## 3 Backpropagation

We have defined and learned how neural network propagates forwards, which is called prediction stage. Now, we want to know how neural network propagates backwards, which is called learning stage. 

For example, assume that we want to predict if an image contains a ball or not, which is a binary problem. As an image, we have RGB values, which means we deal with a three dimensional matrix. We first flatten it to a one-dimensional vector, and then feed it into the neural network to get the output. It can be illustrated figuratively as below. 

![Example for BP](/images/cs229_deeplearning_bp_1.png)

So next, let's talk about how to update its parameters. 

### 3.1 Parameter Initialization

Keep in mind that the input is flattened although it is image. With two layers of neural network, we can draw it as:

![Example for BP](/images/cs229_deeplearning_bp_2.png)

Note how each node in each layer is connected. This is called fully connected. We can now use the method discussed in last section to figure out what output will be for each node in each layer by using matrix notation. In addition, with matrix notation, we can calculate the number of parameters that we are trying to update. I would not repeat the calculation step but the answer is $3n+14$.

Before updating, we need to initialize these parameters. We CANNOT initialize them to zero since this will cause the output of first layer to be zero and further problem when we update them (gradient will be same). The workaround is to initialize them by unit Gaussian. 

After initialization and one single input, we then have the prediction $\hat{y}$. We can use this value to back-propagate so that network can learn from it. If $\hat{y} = y$, then we can nothing to learn. The network does well. However, if not, we have something to ask for network to update its parameters so that it can do better next time. Is it like a human, isn't?

Let's define the loss function as :

$$\mathcal{L}(\hat{y},y) = -\Big[(1-y)\log (1 - \hat{y}) + y\log (\hat{y})\Big]$$

The loss function can basically tell the network about what we really care about. So the network knows what the evaluation scheme is during the training. 

Given a layer index $\ell$, we can update them:

$$W^{[\ell]} = W^{[\ell]} - \alpha\frac{\partial \mathcal{L}}{\partial W^{[\ell]}}$$

$$b^{[\ell]} = b^{[\ell]} - \alpha\frac{\partial \mathcal{L}}{\partial b^{[\ell]}}$$

where $\alpha$ is the learning rate. 

There are two cases that I want to discuss. 

(1) What will happen if we initialize all the parameters to zeros? In this case, we can plug it back to matrix calculation, which will zero as output, which is also the input sigmoid function leading to 0.5 ALWAYS. f

(2) What will happen if we initialize all the parameters to the same values? In this case, from matrix calculation, we can see that this can cause that output from each node in that layer will have all the same values. This will occur to each layer. When we calculate the gradient, this will give us the same gradient in each node in a layer. It will learn the same thing for each neuron. 

Instead, we have something better than Gaussian, called Xavier/He initialization. We initialize it as:

$$w^{[\ell]} \sim \mathcal{N}(0,\sqrt{\frac{2}{n^{[\ell]} + n^{[\ell-1]}}})$$

where $n^{[\ell]}$ is the number of neurons in layer $\ell$. 

### 3.2 Optimization

In the simple neural network above, we have several parameters to update, namely $W^{[1]},b^{[1]},W^{[2]},b^{[2]},W^{[3]},b^{[3]}$. We can use stochastic gradident descent to optimize. That is, we find the derivative with respect to each variable and take a step of it. Let's look at $W^{[3]}$.

$$\begin{align}
\frac{\partial \mathcal{L}}{\partial W^{[3]}} &= -\frac{\partial}{\partial W^{[3]}}\frac{\partial \mathcal{L}}{\partial W^{[3]}}\frac{\partial \mathcal{L}}{\partial W^{[3]}}\bigg((1-y)\log(1-\hat{y}) + y\log\hat{y}\bigg)\\
&= -(1-y)\frac{\partial}{\partial W^{[3]}}\log\bigg(1-g(W^{[3]}a^{[2]}+b^{[3]})\bigg) \\
& - y\frac{\partial}{\partial W^{[3]}}\log\bigg(g(W^{[3]}a^{[2]}+b^{[3]})\bigg) \\
&= -(1-y)\frac{1}{1-g(W^{[3]}a^{[2]}+b^{[3]})}(-1)g^{\prime}(W^{[3]}a^{[2]}+b^{[3]})a^{[2] T}\\
& -y\frac{1}{1-g(W^{[3]}a^{[2]}+b^{[3]})}g^{\prime}(W^{[3]}a^{[2]}+b^{[3]})a^{[2] T}\\
& = (a^{[3]}-y)a^{[2] T}
\end{align}$$

where g is sigmoid function. 

In order to compute the gradient for $W^{[2]}$, we have to use chain rule from calculus, which will give us as:

$$\frac{\partial \mathcal{L}}{\partial W^{[2]}} = \frac{\partial \mathcal{L}}{\partial a^{[3]}}\frac{\partial a^{[3]}}{\partial z^{[3]}}\frac{\partial z^{[3]}}{\partial a^{[2]}}\frac{\partial a^{[2]}}{\partial z^{[2]}}\frac{\partial z^{[2]}}{\partial W^{[2]}}$$

Note that each fraction shows the dependence between numerator and denominator. 

Now, we can plug in each one:

$$\frac{\partial \mathcal{L}}{\partial W^{[2]}} = \underbrace{\frac{\partial \mathcal{L}}{\partial a^{[3]}}\frac{\partial a^{[3]}}{\partial z^{[3]}}}_{a^{[3]} - y}\underbrace{\frac{\partial z^{[3]}}{\partial a^{[2]}}}_{W^{[3]}}\underbrace{\frac{\partial a^{[2]}}{\partial z^{[2]}}}_{g^{\prime}(z^{[2]})}\underbrace{\frac{\partial z^{[2]}}{\partial W^{[2]}}}_{a^{[1]}}$$

Traditionally, we need to use generalized Jacobian matrix for this calculation. If you are not familiar with this, you can check [my post on math](https://wei2624.github.io/math/Useful-Formulas-for-Math/). However, we won't do this here since generalized Jacobian matrix calculation will require a lot of memory. We have to work around it. 

I do suggest to take a look at [this post](http://cs231n.stanford.edu/handouts/derivatives.pdf) and [this post](http://cs231n.stanford.edu/handouts/linear-backprop.pdf) for detailed explanation. Here, I just keep it simple to get:

$$\frac{\partial \mathcal{L}}{\partial W^{[2]}} = \underbrace{(a^{[3]}- y)}_{1\times 1}\underbrace{W^{[3]^T}}_{2\times 1}\odot\underbrace{g^{\prime}(z^{[2]})}_{2\times 1}\underbrace{a^{[1]}}_{1\times 3}$$

where $\odot$ denotes element-wise product. What happens here, in short, is that the first term is scalar but $W^{[3]^T}\odot g^{\prime}(z^{[2]})$ this part is originally a generalized Jacobian matrix multiplication. However, since the activition function is per element, the generalized Jacobian matrix for $\frac{\partial a^{[2]}}{\partial z^{[2]}}$ is a 2 by 2 diagnoal matrix. And $\frac{\partial z^{[3]}}{\partial a^{[2]}}$ is actually a 1 by 2 vector. The matrix multiplication of the two can be calculated in another way which is element-wise product. 

For the last term, the reason that it is not a generalized Jacobian is that we can work around it by just getting the matrix as a result. More details can be found the linked posts above. 

Now, we can use the gradient descent for updating:

$$W^{[\ell]} = W^{[\ell]} - \alpha\frac{J}{W^{[\ell]}}$$

where J is the cost function defined as $J=\frac{1}{m}\sum\limits_{i=1}^m\mathcal{L}^i$. 

Another popular optimization algorithm is called **momentum**. The update rule is:

$$\begin{cases} v_{dW^{[\ell]}} = \beta dW^{[\ell]} + (1-\beta)\frac{\partial J}{dW^{[\ell]}} \\ W^{[\ell]}  = W^{[\ell]} - \alpha v_{dW^{[\ell]}}\\ \end{cases}$$

This rule happens in two stages. The first one is to get the speed and the second is to use the speed to update it. This algorithm basically keeps track of all the past gradient and will help escape from saddle point. 

### 3.3 Analyzing the Parameters

We have done all the components in the training process. If we have trained model which performs 94% on training dataset but only 60% in testing dataset, then there is an overfitting. The possible solutions are: collecting more data, employing regularization or making the model simpler/shallower. In this section, I am going to talk about regularization. 

#### L2 Regularization

Let W donote all the parameters in the model. The L2 regularization adds another term to the cost function, which is called reluarizer:

$$\begin{align}
J_{new} &= J_{old} + \frac{\lambda}{2}\lvert\lvert W \rvert\rvert^2 \\
&=J_{old} + \frac{\lambda}{2}\sum\limits_{i,j}\lvert W_{ij}\rvert^2\\
&=J_{old} + \frac{\lambda}{2}W^TW
\end{align}$$

where $\lambda$ is an arbitrary value. If it is large, it means a parge penalty and large regularization. Then, the update rule has also changed to:

$$\begin{align}
W &= W - \alpha\frac{\partial J}{\partial W} - \alpha\frac{\lambda}{2}\frac{\partial W^TW}{\partial W}\\
&= (1-\alpha\lambda)W - \alpha\frac{\partial J}{\partial W}
\end{align}$$

This means that in updating, some penalties might be included in order to optimize the new J overall. Note that this penalty encourages parameters to be small in l2 magnitude. This is becuase larger magnitude of parameters results in larger varaince. 

### Parameter Sharing

Recall that logistic regression train each parameter for each pixel. However, for ball detection task, if the ball always appears in the center pixels in the training dataset, this might be a problem if a ball appears in a cornor in testing phase. This is because the wieghts on the cornor have never been trained with a ball in there so that the weights do not have that concepts in them. 

To solve this, we have a new type of network structure called **comvolutional neural networks**. Instead of a vector of parameters, we use a matrix of vector, say size of 4 by 4. We take this matrix and slide it over the image. This can be shown below.

![Example for CNN](/images/cs229_deeplearning_cnn_1.png)

This matrix of parameters will take inner product with corresponding pixels in the image, which is a scalar. Then we slide matrix to the right and the bottom, which can be shown as:

![Example for CNN](/images/cs229_deeplearning_cnn_2.png)

Note that each matrix share the same weighs across the entire image. 

# Backpropagation



