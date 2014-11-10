[TOC]

<!-- 
Clive's advice
* 3-6 visually distinct chunks of less than 100 words with eye catching, informative headings
* All diagrams/photos need captions so they stand alone. Include large eye catcher
 -->


Findings (conclusions/discussion/take-home message)
-------------------

Problem 
---------------------------------
(In plain English, clearly say what the problem/issue is and how your research might improve this. )

3 Categories of Surrogate Models
-------------------------
1. Data-driven surrogates involving statistical approximations of the complex model output calibrated on a set of inputs and outputs of the complex model (snapshots). 
2. Projection based methods, where the governing equations are projected onto a reduced dimension subspace characterized by a basis of orthonormal vectors. Typically divided into SVD and Krylov based methods. 
3. Multi-fidelity based surrogates, built by simplifying the underlying physics or reducing numerical resolution.

Polynomial Chaos
---------------------------
Let $(\Omega, \sigma, P)$ be a probability space where $\sigma$ is a $\sigma$-algebra on $\Omega$ and $P$ is a probability measure on $(\sigma, P)$.

Any random variable $X: \Omega \rightarrow \mathbb{R}$ with finite variance (square integrable, $X \in L^2(\Omega)$) may be written as the polynomial chaos (PC) expansion of orthogonal polynomials $\Gamma_p$ 

$$\begin{align}
X(\omega) &= a_0 \Gamma_0 + \sum_{i_1=1}^\infty a_{i_1} \Gamma_1(\xi_{i_1}) + \sum_{i_1=1}^\infty \sum_{i_2=1}^{i_1} a_{i_1i_2} \Gamma_2(\xi_{i_1},\xi_{i_2})+ \cdots \notag \newline
&= \sum_{k=0}^\infty \alpha_k \Psi_k(\xi_1,\xi_2,\cdots) 
\end{align}$$

Since $\{\xi_i\}_{i=1}^\infty$ are Gaussian, this orthogonality condition requires $\Gamma_p$ (or equivalently $\Psi_k$ ) to be multivariate Hermite polynomials.

For random fields (processes) including spatial $x$ and temporal $t$ 
$$\begin{equation}
X(x,t,\omega) = \sum_{k=0}^\infty \alpha_k(x,t) \Psi_k(\xi_{i_1},\xi_{i_2},\cdots). 
\end{equation}$$

<!-- \approx \sum_{k=0}^P \alpha_k \Psi_k(\boldsymbol\xi) -->

Uncertainty
--------------

Structural Uncertainty
------------------

Diagnostics
------------------

PC emulator of Groundwater model (MODFLOW)
------------------------------------------
* automated process using freely available, continental available datasets in variety of formats (shp, gdb, WFS)
* scripts to convert from formats to modflow inputs with examples of where to use parametes
* PC emultor of these inputs to head or flux (common outputs)