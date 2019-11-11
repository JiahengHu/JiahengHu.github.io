---
layout: single
mathjax: true
toc: true
toc_sticky: true
author_profile: true
title: Modular Robot Hardware Optimization
share: false
permalink: /research/hardware_opt/
---

This work is advised by Professor Matt Travers and Professor Howie Choset (CMU). This is an ongoing work and therefore results are not shown. 

# Overview

In this project, we designed a pipeline that can generate a desired modular robots design and the corresponding policy given a task.  

# Current Progress

1. We reimplemented the "Jointly Learning to Construct and Control Agents using Deep Reinforcement Learning" by M. Walter on a snake robot model, and get the corresponding optimal design for snake navigation.
2. We expanded the optimization to include not only the physical parameters of the robot (such as link length and mass), but also the morphological structure (such as link connection).
3. We expanded the scheme to modular robot (Eigen Bot), and used Graph Neural Network as the control policy.
4. We are working on generating the robot distribution with a generator network conditioning on the terrain / task.



|:-----:|:-------:|
|Platform|Ubuntu 18.04|
|Programming Language|Python|
|Deep Learning Framework|Tensorflow,PyTorch|
|Research Area|Deep Reinforcement Learning, Graph Neural Network, Hardware Optimization|