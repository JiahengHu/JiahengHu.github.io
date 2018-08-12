---
layout: post
mathjax: true
title: Useful Formulas for Math
---

I am opening a new post for formulas in Learning Theory and general Machine Learning area. It is purely mathamatically based. It is suggested to use it as reference instead of studying them one by one. 

* This will become a table of contents (this text will be scraped).
{:toc}

# Probability Theory and Expectation
## Law of Total Expectection (a.k.a. Tower Rule)

Let random variable X and Y defined in the same probability space. Then, $E_X (X) = E_Y(E_X(X\lvert Y))$.

Proof: 

$$\begin{align}
E_Y (E_X (X\lvert Y)) &= E_Y(\sum_{x} x * P(X\lvert Y))\\
&= \sum_{y} \big[ \sum_{x} x* P(X=x\lvert Y)\big] P(y) \\
&=\sum_{x}  \sum_{y} x* P(Y)* P(X=x\lvert Y)\\
&=\sum_{x} x* \sum_{y} P(Y)* P(X=x\lvert Y)\\
&=\sum_{x} x*P(x)\\
&=E_X (X) 
\end{align}$$

# Learning Theory
## L' Hospital Rule

L' Hospital Rule uses derivatives to help evaluate limits involving inderterminate forms. It states that for functions $f$ and $g$ which are **differentiable** on an open interval $I$ except possibly at a point $c$ contained in $I$,if 

$$\lim\limits_{x\to c} f(x) = \lim\limits_{x\to c} g(x) = 0$$ or $$\pm \infty$$

$g^{\prime}(x)\neq 0$ for all $x$ in $I$ with $x\neq c$ and $\lim\limits_{x\to c} \frac{f^{\prime}(x)}{g^{\prime}(x)}$ exists,

then $\lim\limits_{x\to c} \frac{f(x)}{g(x)} = \lim\limits_{x\to c} \frac{f^{\prime}(x)}{g^{\prime}(x)}$

## Markov Inequality

For a positive random variable $X \leq 0$, 

$$Pr[X \geq b] \geq \frac{E[X]}{b}$$

Prove: $E[X] = \sum_x xPr(X=x) \geq \sum_{x\leq b} bPr(X=x) = bPr(X\geq b)$

# Math
## Log properties

$a^{log_b n} = (b^{log_b a})^{log_b n} = (b^{log_b n})^{log_b a} = n^{log_b a} $

## Geometric Series

$1 + x + x^2 + \dots + x^n = \frac{1-x^{n+1}}{1-x}$ for $x\neq 1$

$1 + x + x^2 +\dots = \frac{1}{1-x}$ for $\lvert x\rvert <1$

## Harmonic Series Series

$\sum\limits_{i=1}^{n} \frac{1}{i} = \ln{n} + \gamma + \frac{1}{2n} - \frac{1}{12n^2}$ is the best approximation for the series. 

## Stirling's approximation

$$\log_2 n! = n\log_2 n - (\log_2 e)n + \mathcal{O}(\log_2 n)$$

## Trace Properties

$tr(AB) = tr(BA)$

$tr(ABC) = tr(CAB) = tr(BCA)$ which is called cyclic property of trace.

$\nabla_{A} tr(AB) = b^{T}$

$\nabla_{A^T} f(A) = (\nabla_A f(A))^T$

$\nabla_A tr(ABA^TC) = CAB + C^TAB^T$

$\nabla_A \lvert A\rvert = \lvert A\rvert (A^{-1})^T$

## Woodbury Matrix Identity

It says that for any given A with n by n and U with n by k and C with k by k and V with k by n such that A and C are nonsingular, we have :

$$(A + UCV)^{-1} = A{^-1} - A^{-1}U(C^{-1} + VA^{-1}U)^{-1}VA$$

Proof can be found on Wiki easily. 

## Vector Calculus

### Gradient

Let's denote a function $f:\mathbb{R}^n\mapsto\mathbb{R}$. The gradient of the function is defined as:

$$\triangledown_x f = \frac{\partial}{\partial x}f = \begin{bmatrix}\frac{\partial f}{\partial x_1} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix}$$

It simply says that for a function which takes a vector as input and a scaler as output, the gradient of the function is a column vector which each element is the derivative of f with respect to a single component of x. 

For example, $y = x^Tz$ where $x \in \mathbb{R}^n$ should be a good practice to work. 

### Jacobian Matrix

For Jacobian, the case is more complicated. Let's denote a function $f:\mathbb{R}^n\mapsto\mathbb{R}^m$. The gradient of the function is defined as:







