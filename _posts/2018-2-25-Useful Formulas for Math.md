---
layout: post
mathjax: true
title: Useful Formulas for Math
---

I am opening a new post for formulas in Learning Theory and general Machine Learning area. It is purely mathamatically based. It is suggested to use it as reference instead of studying them one by one. 

## Probability Theory and Expectation
**Law of Total Expectection (a.k.a. Tower Rule)**

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

## Learning Theory
**L' Hospital Rule**

L' Hospital Rule uses derivatives to help evaluate limits involving inderterminate forms. It states that for functions $f$ and $g$ which are **differentiable** on an open interval $I$ except possibly at a point $c$ contained in $I$,if 

$$\lim\limits_{x\to c} f(x) = \lim\limits_{x\to c} g(x) = 0$$ or $$\pm \infty$$

$g^{\prime}(x)\neq 0$ for all $x$ in $I$ with $x\neq c$ and $\lim\limits_{x\to c} \frac{f^{\prime}(x)}{g^{\prime}(x)}$ exists,

then $\lim\limits_{x\to c} \frac{f(x)}{g(x)} = \lim\limits_{x\to c} \frac{f^{\prime}(x)}{g^{\prime}(x)}$

**Markov Inequality**

For a positive random variable $X \leq 0$, 

$$Pr[X \geq b] \geq \frac{E[X]}{b}$$

Prove: $E[X] = \sum_x xPr(X=x) \geq \sum_{x\leq b} bPr(X=x) = bPr(X\geq b)$

## Math
**Log properties**

$a^{log_b n} = (b^{log_b a})^{log_b n} = (b^{log_b n})^{log_b a} = n^{log_b a} $

**Geometric Series**

$1 + x + x^2 + \dots + x^n = \frac{1-x^{n+1}}{1-x}$ for $x\neq 1$

$1 + x + x^2 +\dots = \frac{1}{1-x}$ for $\lvert x\rvert <1$

**Harmonic Series Series**

$\sum\limits_{i=1}^{n} \frac{1}{i} = \ln{n} + \gamma + \frac{1}{2n} - \frac{1}{12n^2}$ is the best approximation for the series. 

**Stirling's approximation**

$$\log_2 n! = n\log_2 n - (\log_2 e)n + \mathcal{O}(\log_2 n)$$

**Trace Properties** 

$tr(AB) = tr(BA)$

$tr(ABC) = tr(CAB) = tr(BCA)$ which is called cyclic property of trace.

$\nabla_{A} tr(AB) = b^{T}$

$\nabla_{A^T} f(A) = (\nabla_A f(A))^T$

$\nabla_A tr(ABA^TC) = CAB + C^TAB^T$

$\nabla_A \lvert A\rvert = \lvert A\rvert (A^{-1})^T$

