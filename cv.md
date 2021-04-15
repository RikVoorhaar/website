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
- Tensor networks
- Non-convex and Riemannian optimization
- Machine learning

Currently I'm working with [Bart Vandereycken](https://www.unige.ch/math/vandereycken/) on low rank methods in tensor networks. Specifically
I'm exploring the relationship between Tensor Train / MPS networks and random forests. It seems that
tensor train networks can be used to define a general-purpose machine learning estimator that has
similar properties to random forests. I'm using Riemannian methods to optimize them, and I'm trying
to keep all my code compatible with several numerical Python libraries. 

On the weekends I like to study topics in data science, bioinformatics and scientific computing to
broaden my knowledge. I do this by either taking online courses, reading text books, or doing
small programming projects. For the latter I usually write blog posts on this website.


## Publications and preprints

[Recovering data you have never seen](https://doi.org/10.25250/thescbr.brk513) April 2021
<ul style="list-style-type:none;">
  <li><sup> I wrote an piece in a science outreach journal describing an article about low-rank matrix completion. The goal of this journal is to make the core ideas behind published scientific research accessible to a wide audience. </sup></li>
</ul>

[On certain Hochschild cohomology groups for the small quantum group](https://arxiv.org/abs/2104.05113) April 2021, joint work with Nicolas Hemelsoet
<ul style="list-style-type:none;">
  <li><sup>We apply the algorithm for the BGG resolution developed in the previous paper to compute Hochschild cohomology of blocks of the small quantum group. This allows us to study the center of the small quantum group, and our computations give stronger evidence for several conjectures concerning the small quantum group. My contribution was writing all the code needed for this project. </sup></li>
</ul>


[A computer algorithm for the BGG resolution](https://www.sciencedirect.com/science/article/abs/pii/S0021869320305135) November 2019, joint work with Nicolas Hemelsoet  
_Published in the Journal of Algebra in 2021_
<ul style="list-style-type:none;">
  <li><sup> In this work we describe an algorithm to compute the BGG resolution for modules over a simple Lie algebra. This is then used to compute various thing like the Hochschild cohomology of some flag varieties. My contribution was coding the implementation of the algorithm, and solving several algorithmic problems.</sup></li>
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
  <li><sup>This is a package meant to help writing backend agnostic code, i.e. code that can manipulate objects from different numerical libraries like numpy, tensorflow or pytorch (among others). I added a few extra translations from the numpy API to other libraries, I improved the functionality to infer the backend of the arguments of some functions, and I made data type handling for array creation operations more consistent. </sup></li>
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
Armadillo, Cython, CVXPY, Docker, Linux, NumPy, Pandas, PyTorch, Sagemath, SciPy, Sphinx,
Tensorflow, Windows

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


### Mathematical expertise

I have a wide background in pure and applied mathematics, and I feel comfortable with research-level mathematics in the following areas:

**Applied mathematics:**  
-- Bayesian statistics  
-- Classical machine learning  
-- Computer vision  
-- Neural networks  
-- Nonconvex optimization  
-- Numerical linear algebra  
-- Riemannian optimization  
-- Signal processing  
-- Tensor networks  

**Pure mathematics:**  
-- Category theory  
-- Deformation quantization  
-- Differential geometry  
-- Fiber bundles  
-- Lie theory  
-- Lie groupoids  
-- Operadic algebra  
-- Poisson geometry  
-- Tensor / monoidal categories  