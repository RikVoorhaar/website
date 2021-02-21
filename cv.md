---
layout: single
title: Curriculum Vitae
permalink: /cv/
toc: true
teaching: /teaching.md
---

## Research

Research interests:  
- Numerical linear algebra
- Low-rank tensor decompositions
- Riemannian optimization
- Machine learning

Currently I'm working with Bart Vandereycken on low rank methods in tensor networks. Specifically
I'm exploring the relationship between Tensor Train / MPS networks and random forests. It seems that
tensor train networks can be used to define a general-purpose regressor that has similar properties
to random forests. I'm using Riemannian methods to optimize them, and I'm trying to keep all my code
backend agnostic so anyone can use it. 

On the weekends I like to study topics in datascience, bioinformatics and scientific computing to
broaden my knowledge. I do this by either taking online courses, reading text books, or doing
small programming projects. For the latter I usually write blog posts on this website.

I am also working with Nicolas Hemelsoet on an algorithm to do explicit computations with the BGG
resolution. I wrote a package in Sagemath/Python that can compute the BGG resolution and its 
cohomology for a large class of Lie algebra modules. This can be used to for example compute
the Hochschild cohomology of flag varieties, or to do computations related to the center of the
small quantum group. 

Before that, I was trying to develop methods to compress MERA tensor networks using Riemannian
optimization methods. Unfortunately this didn't work as well as expected. It turns out to be very
hard to compress tensor networks with loops.

Still before that, I worked with Pavol Ševera on deformation quantization of Poisson-Lie principal
bundles. The idea there is to combine the quantization of Poisson-Lie groups and of Poisson
bivectors to quantize a bundle over a Poisson manifold with Poisson-Lie structure group into a
Hopf-Galois extension of the quantized Lie bialgebra.

I also dabbled in 2-categories and higher gauge theory in my master thesis, and at the beginning of
my PhD. Under supervision of Anton Alekseev and with some help of Eugene Lerman I worked on a "toy
model" of 2-gauge theory and principal 2-bundles. I translated a number of facts from the
differential geometry of ordinary principal bundles to this setting.


## Publications

[A computer algorithm for the BGG resolution](https://www.sciencedirect.com/science/article/abs/pii/S0021869320305135) November 2019, joint work with Nicolas Hemelsoet  
_Published in the Journal of Algebra in 2021_
<ul style="list-style-type:none;">
  <li><sup> In this work we descibe an algorithm to compute the BGG resolution for modules over a simple Lie algebra. This is then used to compute various thing like the Hochschild cohomology of some flag varietes.</sup></li>
</ul>

[Parallel 2-transport and 2-group torsors](https://arxiv.org/abs/1811.10060) October 2018
<ul style="list-style-type:none;">
  <li><sup>This work is a continuation of my masters thesis. The idea is to study
  a toy model of principal 2-bundles and 2-transport by restricting to a stricter notion, where the fibers are all strict 2-groups.
  This allows to get some nice generalizations of the classical theory, which would
  be harder to proof with the more useful notions weaker notions.</sup></li>
</ul>

[Higher Gauge Theory](https://dspace.library.uu.nl/handle/1874/361953) February 2018 (Master thesis)

## Open source contributions

[bgg-cohomology sage package](https://github.com/RikVoorhaar/bgg-cohomology)
<ul style="list-style-type:none;">
  <li><sup>I wrote a Sagemath package used for computing the BGG resolution of simple Lie algebra modules and the associated cohomology. </sup></li>
</ul>

[geoopt](https://github.com/geoopt/geoopt)
<ul style="list-style-type:none;">
  <li><sup>This is a package for Riemannian optimization using PyTorch. I added a Riemannian line search and conjugate gradient optimizer to this project. </sup></li>
</ul>

[autoray](https://github.com/jcmgray/autoray)
<ul style="list-style-type:none;">
  <li><sup>This is a package meant to help writing backend agnostic code, i.e. code that can manipulate objects from different numerical libraries like numpy, tensorflow or pytorch (among others). I added a few extra translations from the numpy API to other libraries, and I improved the functionality
  to infer the backend of the arguments of some functions. </sup></li>
</ul>

## Work experience

**2018-202X**  
-- PhD student at _University of Geneva_. 
<ul style="list-style-type:none;">
<li><sup>I was working in pure mathematics from 2018 until early
2020, when I switched research direction to applied math. Over the past few years a significant
fraction of my time is spent writing research code in Python, both numerical code and code for
computer algebra. I spend about 20% of my time teaching. I also spend about 20% of my time studying 
to broaden my knowledge about data science and scientific computing, either by doing online courses,
reading text books, or doing small programming projects. </sup></li>
</ul>

**2014-2016**  
-- Teaching assistant at _Utrecht University_.
<ul style="list-style-type:none;">
<li><sup>I was a teaching assistant for four different courses
during my time as a student at Utrecht.  </sup></li>
</ul>

**2012-2014**  
-- Mathematics tutor. 
<ul style="list-style-type:none;">
<li><sup> I tutored two students in 2012-2013, and one student 2013-2014. The students studied 
the International School Hilversum, taking IB Higher Level Mathematics. </sup></li>
</ul>

## Education

**2016-2017**  
-- Masterclass Geometry, Toplogy and Physics at _University of Geneva_

**2015-2018**  
-- Masters degree Mathematical Sciences at _Utrecht University_ _(cum laude, GPA 4.00)_  
-- Honors degree "Utrecht Geometry Center" at _Utrecht University_

**2012-2015**  
-- Bachelor degree Mathematics at _Utrecht University_ _(cum laude, GPA 4.00)_  
-- Bachelor degree Physics and Astronomy at _Utrecht University_ _(cum laude, GPA 4.00)_

### Online courses completed
2019/08 (Coursera): [Advanced Machine Learning Specialization](https://www.coursera.org/account/accomplishments/specialization/5BM8U5DJJCJN)  
2020/09 (Coursera): [Genomic Data Science Specialization](https://www.coursera.org/account/accomplishments/specialization/NYQNJVCT7XV3)  
2021/02 (Coursera): [Neuroscience and Neuroimaging Specialization](https://www.coursera.org/account/accomplishments/specialization/REWS86DYU496)

## Skills

### Programming languages
**Advanced**  
-- Python

**Intermediate**  
-- $$\LaTeX$$  
-- Mathematica

**Beginner**  
-- C/C++  
-- R

**Wishlist**  
-- Julia  
-- MATLAB

**Tools**  
Armadillo, Cython, Docker, Linux, NumPy, Pandas, PyTorch, Sagemath, SciPy, Sphinx, Tensorflow, Windows

### Languages
**C2 Level**  
-- Dutch  
-- English

**B1 Level**  
-- French

**A2 Level**  
-- Japanese  
-- Russian  
-- Spanish