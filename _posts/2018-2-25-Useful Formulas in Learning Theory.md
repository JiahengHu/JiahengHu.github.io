---
layout: post
mathjax: true
title: Useful Formulas in Learning Theory
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
&=\sum_{x} x*P(x)
&=E_X (X) \qed
\end{align}$$
