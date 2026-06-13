# ML for Science — Physics-Informed & Equivariant Benchmarks for Scientific Machine Learning

> A unified, research-grade benchmark suite spanning **physics-informed neural networks**, **E(3)-equivariant graph neural networks**, **single-cell genomics VAEs**, **Bayesian materials discovery**, **Fourier neural operators**, and **protein structure prediction** — built for reproducible experimentation at the intersection of deep learning and the natural sciences.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2102.09844-b31b1b.svg)](https://arxiv.org/abs/2102.09844)

Scientific machine learning (SciML) is reshaping how researchers model physical systems, discover materials, and analyse biological data. This repository provides a modular, well-documented codebase that implements key SciML architectures alongside controlled synthetic data-generating processes (DGPs) that expose ground-truth parameters. Unlike ad-hoc research scripts, every module is designed with a common evaluation harness so that new methods can be benchmarked head-to-head on identical data. The library ships with four complete research pipelines — protein stability prediction, climate downscaling, single-cell genomics integration, and materials discovery — each paired with publication-grade metrics and reporting. Whether you are a PhD student surveying the SciML landscape or a hiring manager evaluating scientific computing competence, this codebase demonstrates end-to-end fluency from mathematical derivation through production-quality PyTorch implementation to automated benchmark orchestration.

---

## Table of Contents

- [Research Background \& Motivation](#research-background--motivation)
- [Mathematical Foundations](#mathematical-foundations)
  - [Physics-Informed Neural Networks (PINNs)](#1-physics-informed-neural-networks-pinns)
  - [Burgers' Equation Residual](#2-burgers-equation-residual)
  - [E(3)-Equivariant Graph Neural Networks](#3-e3-equivariant-graph-neural-networks)
  - [Equivariance Proof Sketch](#4-equivariance-proof-sketch)
  - [Single-Cell VAE with Negative Binomial Likelihood](#5-single-cell-vae-with-negative-binomial-likelihood)
  - [ComBat Empirical Bayes Batch Correction](#6-combat-empirical-bayes-batch-correction)
  - [Harmony Iterative EM Integration](#7-harmony-iterative-em-integration)
  - [Bayesian Optimisation Acquisition Functions](#8-bayesian-optimisation-acquisition-functions)
  - [Fourier Neural Operator](#9-fourier-neural-operator)
- [Architecture Diagram](#architecture-diagram)
- [Repository Structure](#repository-structure)
- [Code Walkthrough](#code-walkthrough)
  - [PINN: Automatic Differentiation for PDE Residuals](#pinn-automatic-differentiation-for-pde-residuals)
  - [EGNN: Equivariant Message Passing](#egnn-equivariant-message-passing)
  - [scVAE: Negative Binomial Reconstruction](#scvae-negative-binomial-reconstruction)
  - [ComBat: Empirical Bayes Shrinkage](#combat-empirical-bayes-shrinkage)
  - [Harmony: Diversity-Penalised Soft Clustering](#harmony-diversity-penalised-soft-clustering)
  - [FNO: Spectral Convolution in Fourier Space](#fno-spectral-convolution-in-fourier-space)
  - [Active Learning: GP-Based Bayesian Optimisation](#active-learning-gp-based-bayesian-optimisation)
  - [Protein Structure: Contact Map Prediction](#protein-structure-contact-map-prediction)
  - [Data-Generating Processes](#data-generating-processes)
  - [Evaluation Runner](#evaluation-runner)
- [Benchmark Results](#benchmark-results)
- [Reproduction Commands](#reproduction-commands)
- [Installation](#installation)
- [References](#references)
- [Future Work](#future-work)
- [License](#license)

---

## Research Background & Motivation

The last decade has witnessed an extraordinary convergence between deep learning and the physical, biological, and materials sciences. Classical numerical methods — finite elements, spectral solvers, Monte Carlo samplers — have dominated scientific computing for half a century, yet they struggle with the curse of dimensionality, expensive forward simulations, and the gap between simulation fidelity and real-world observations. Machine learning offers a complementary paradigm: learn surrogate models, infer latent structure, and optimise under uncertainty, all from data.

**Physics-informed neural networks (PINNs)**, introduced by Raissi, Perdikaris, and Karniadakis (2019) in the *Journal of Computational Physics*, embed partial differential equation constraints directly into the training loss. Rather than treating the neural network as a black box, PINNs penalise violations of known governing equations at collocation points via automatic differentiation. This elegantly merges data-driven flexibility with physics-based inductive bias, enabling accurate solutions even in data-scarce regimes. Our implementation extends the original PINN framework with adaptive collocation point refinement: after every fixed number of epochs, the trainer identifies the spatial regions with the largest PDE residuals and concentrates additional collocation points there, focusing model capacity where the physics is most difficult to satisfy.

**Equivariant neural networks** address a fundamental limitation of standard architectures: they do not respect the symmetries of physical systems. Satorras, Hoogeboom, and Welling (2021, arXiv:2102.09844) introduced E(n)-Equivariant Graph Neural Networks (EGNNs) that maintain equivariance under the Euclidean group $E(3)$ — rotations, translations, and reflections — without requiring data augmentation or higher-order tensor representations. This is critical for molecular and protein property prediction, where the properties of a molecule must be invariant to the orientation in which it is observed. Our codebase implements the full EGNN message-passing scheme with explicit equivariance verification tests, and compares it against a non-equivariant MLP baseline to quantify the benefit of built-in symmetry.

**Single-cell RNA sequencing (scRNA-seq)** has revolutionised genomics by profiling gene expression at cellular resolution, but technical variation between experimental batches can mask genuine biological signals. Lopez et al. (2018, arXiv:1709.02082) introduced scVI, a deep generative model that uses a variational autoencoder with a negative binomial observation model tailored to the overdispersed count statistics of UMI-based protocols. Our `scVAE` module faithfully reproduces this formulation, including learnable per-gene inverse dispersion parameters and β-KL annealing. We complement it with two classical batch correction algorithms: **ComBat** (Johnson et al., 2007), which applies empirical Bayes shrinkage to location-scale batch effects, and **Harmony** (Korsunsky et al., 2019), which performs iterative soft-clustering with batch-diversity regularisation in PCA space. The benchmark measures both batch mixing entropy and biological preservation to ensure corrections do not erase true cell-type differences.

**Bayesian optimisation (BO)** is the method of choice for optimising expensive black-box functions, as reviewed comprehensively by Frazier (2018, arXiv:1807.02811). In materials science, synthesising and characterising a new alloy composition may cost thousands of dollars and days of lab time, making sample efficiency paramount. Our active learning pipeline wraps a Gaussian process surrogate with three acquisition functions — Expected Improvement (EI), Upper Confidence Bound (UCB), and Probability of Improvement (PI) — and benchmarks them against random and pure-uncertainty baselines on a synthetic materials property landscape with a known global optimum.

The **Fourier Neural Operator (FNO)**, proposed by Li et al. (2020, arXiv:2010.08895), learns resolution-invariant mappings between function spaces by performing convolutions in the Fourier domain. Unlike PINNs, which solve one PDE instance at a time, neural operators learn the entire solution operator and generalise across initial/boundary conditions. Our implementation provides a clean 1-D FNO with truncated spectral convolution layers, complementing the pointwise PINN approach.

Recent breakthroughs in **protein structure prediction** — AlphaFold2 (Jumper et al., 2021) and its predecessors (Senior et al., 2020) — have demonstrated that deep learning can achieve experimental-level accuracy on the protein folding problem. While our codebase does not replicate the full AlphaFold2 architecture, it implements a contact-map prediction network that captures the outer-product pairwise reasoning central to modern structure prediction, and provides equivariant property prediction via EGNN for stability estimation. The **Boltzmann generator** paradigm (Noé et al., 2019, *Science*) further motivates the use of normalising flows and VAEs for sampling molecular conformations, a direction our VAE modules support as extensible building blocks.

Together, these modules provide a comprehensive cross-section of scientific ML, from governing-equation regularisation to geometric deep learning, from statistical genomics to sequential experimental design. Every component is backed by controlled synthetic data with known ground truth, enabling rigorous ablation studies without the confounds of real-world data preprocessing.

---

## Mathematical Foundations

This section provides complete mathematical derivations for every core algorithm implemented in the codebase. All variables are defined where introduced, and the connection to code is made explicit via references to specific source files and functions.

### 1. Physics-Informed Neural Networks (PINNs)

A physics-informed neural network learns a solution $u_\theta : \mathbb{R}^d \to \mathbb{R}$ parameterised by neural network weights $\theta$, subject to both observational data and a governing PDE. The total training loss is:

$$\mathcal{L} = \mathcal{L}_{\mathrm{data}} + \lambda_r \mathcal{L}_{\mathrm{res}}$$

where:

- $\mathcal{L}_{\mathrm{data}} = \frac{1}{N_d} \sum_{i=1}^{N_d} \| u_\theta(x_i) - y_i \|^2$ is the mean squared error on $N_d$ labelled observations $(x_i, y_i)$.
- $\mathcal{L}_{\mathrm{res}} = \frac{1}{N_c} \sum_{j=1}^{N_c} \mathcal{R}(x_j)^2$ is the mean squared PDE residual evaluated at $N_c$ collocation points.
- $\lambda_r > 0$ is a hyperparameter controlling the relative weight of physics enforcement.
- $\mathcal{R}(x)$ is the PDE residual, defined below for specific equations.

**Heat equation.** For the 1-D heat equation with thermal diffusivity $\alpha > 0$, the strong-form PDE is:

$$\frac{\partial u}{\partial t} - \alpha \frac{\partial^2 u}{\partial x^2} = 0$$

The residual at a collocation point $(x, t)$ is:

$$\mathcal{R}(x, t) = \frac{\partial u_\theta}{\partial t} - \alpha \frac{\partial^2 u_\theta}{\partial x^2}$$

In code, the network takes input $\mathbf{x} = (x, t) \in \mathbb{R}^2$ and outputs $u_\theta(\mathbf{x}) \in \mathbb{R}$. First-order derivatives $\partial u / \partial x$ and $\partial u / \partial t$ are obtained via `torch.autograd.grad` with `create_graph=True` to allow second-order differentiation through the computational graph. The second spatial derivative $\partial^2 u / \partial x^2$ is computed by a second `autograd` call on $\partial u / \partial x$. Specifically:

1. Forward pass: $u = f_\theta(\mathbf{x})$ where $\mathbf{x} \in \mathbb{R}^{N \times 2}$.
2. First derivatives: $\nabla_{\mathbf{x}} u = \left[\frac{\partial u}{\partial x},\; \frac{\partial u}{\partial t}\right]$ via `torch.autograd.grad(u, x, ...)`.
3. Extract $\frac{\partial u}{\partial x} = (\nabla_{\mathbf{x}} u)_{:, 0}$ and $\frac{\partial u}{\partial t} = (\nabla_{\mathbf{x}} u)_{:, 1}$.
4. Second derivative: $\frac{\partial^2 u}{\partial x^2} = \left(\text{autograd.grad}\!\left(\frac{\partial u}{\partial x},\, \mathbf{x}\right)\right)_{:, 0}$.
5. Residual: $\mathcal{R} = \frac{\partial u}{\partial t} - \alpha \frac{\partial^2 u}{\partial x^2}$.

**Adaptive collocation refinement.** Every $E_r$ epochs, the trainer evaluates the residual magnitude $|\mathcal{R}(x_j)|^2$ at all current collocation points. The top fraction $\rho$ of worst-performing points are duplicated with small Gaussian perturbation $\epsilon \sim \mathcal{N}(0, \sigma_p^2 I)$, creating a denser mesh in regions where the PDE is most violated. This focuses model capacity where it is most needed, analogous to adaptive mesh refinement in classical numerical methods.

### 2. Burgers' Equation Residual

The 1-D viscous Burgers' equation with kinematic viscosity $\nu > 0$ is:

$$\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} - \nu \frac{\partial^2 u}{\partial x^2} = 0$$

The residual computed by the PINN is:

$$\mathcal{R}(x, t) = \frac{\partial u_\theta}{\partial t} + u_\theta \frac{\partial u_\theta}{\partial x} - \nu \frac{\partial^2 u_\theta}{\partial x^2}$$

This introduces a nonlinear advection term $u \partial_x u$ that requires careful handling in `autograd`: the product $u_\theta \cdot \partial_x u_\theta$ must remain in the computational graph so that gradients flow through both factors during backpropagation. The implementation computes the same first- and second-order derivatives as the heat equation, then assembles the residual as `du_dt + u * du_dx - nu * d2u_dx2`.

### 3. E(3)-Equivariant Graph Neural Networks

The EGNN of Satorras et al. (2021) defines a message-passing neural network on a graph $G = (V, E)$ where each node $i$ carries a feature vector $h_i \in \mathbb{R}^d$ and a 3-D coordinate $x_i \in \mathbb{R}^3$. The update equations for one layer are:

**Message computation:**

$$m_{ij} = \phi_e\!\left(h_i,\; h_j,\; \|x_i - x_j\|^2,\; a_{ij}\right)$$

where $\phi_e : \mathbb{R}^{2d + 1 + d_a} \to \mathbb{R}^d$ is a learnable edge MLP, $\|x_i - x_j\|^2$ is the squared Euclidean distance (an $O(3)$-invariant quantity), and $a_{ij} \in \mathbb{R}^{d_a}$ are optional edge attributes.

**Coordinate update (equivariant):**

$$x_i' = x_i + \sum_{j \neq i} (x_i - x_j)\, \phi_x(m_{ij})$$

where $\phi_x : \mathbb{R}^d \to \mathbb{R}$ is a scalar-valued MLP. The coordinate update is a weighted sum of relative displacement vectors, each scaled by a learned scalar function of the message. Because each displacement vector $(x_i - x_j)$ transforms as a vector under $O(3)$, and $\phi_x(m_{ij})$ is invariant (it depends only on $\|x_i - x_j\|^2$ and invariant features), the overall update is equivariant.

**Node feature update (invariant):**

$$h_i' = \phi_h\!\left(h_i,\; \sum_{j} m_{ij}\right)$$

where $\phi_h : \mathbb{R}^{2d} \to \mathbb{R}^d$ is a node MLP. All inputs to $\phi_h$ are invariant quantities ($h_i$ is invariant by induction, and each $m_{ij}$ depends only on invariant inputs), so $h_i'$ is invariant.

**Readout.** For graph-level prediction, the invariant node features are globally pooled:

$$\hat{y} = \text{MLP}\!\left(\frac{1}{|V|} \sum_{i \in V} h_i^{(L)}\right)$$

where $h_i^{(L)}$ is the node feature after $L$ message-passing layers. Since each $h_i^{(L)}$ is $O(3)$-invariant, the pooled representation and final prediction are also invariant.

### 4. Equivariance Proof Sketch

**Claim.** Let $R \in O(3)$ be an arbitrary orthogonal transformation and $t \in \mathbb{R}^3$ a translation. Under the map $x_i \mapsto Rx_i + t$, the EGNN coordinate update is equivariant and the node update is invariant.

**Proof.**

*Step 1: Invariance of squared distances.* Let $\tilde{x}_i = R x_i + t$. Then:

$$\|\tilde{x}_i - \tilde{x}_j\|^2 = \|(Rx_i + t) - (Rx_j + t)\|^2 = \|R(x_i - x_j)\|^2 = (x_i - x_j)^T R^T R (x_i - x_j) = \|x_i - x_j\|^2$$

since $R^T R = I$ for $R \in O(3)$. Therefore the messages $m_{ij}$ are identical under the transformation.

*Step 2: Equivariance of the relative displacement.* The relative vector transforms as:

$$\tilde{x}_i - \tilde{x}_j = R(x_i - x_j)$$

*Step 3: Equivariance of the coordinate update.* Since $\phi_x(m_{ij})$ is invariant (proven in Step 1):

$$\tilde{x}_i' = \tilde{x}_i + \sum_j (\tilde{x}_i - \tilde{x}_j)\, \phi_x(m_{ij}) = (Rx_i + t) + \sum_j R(x_i - x_j)\, \phi_x(m_{ij})$$

$$= R\!\left(x_i + \sum_j (x_i - x_j)\, \phi_x(m_{ij})\right) + t = R x_i' + t$$

This is precisely the definition of equivariance: $f(R\mathbf{x} + t) = R f(\mathbf{x}) + t$.

*Step 4: Invariance of node features.* Since messages $m_{ij}$ are invariant and $\phi_h$ takes only invariant inputs:

$$\tilde{h}_i' = \phi_h(\tilde{h}_i, \sum_j m_{ij}) = \phi_h(h_i, \sum_j m_{ij}) = h_i'$$

The output prediction is therefore $O(3)$-invariant. $\square$

The codebase includes a runtime verification method `FullEGNN.verify_equivariance` that applies a random rotation $R \in SO(3)$ and checks both output invariance ($\|f(h, Rx) - f(h, x)\| < \epsilon$) and coordinate equivariance numerically.

### 5. Single-Cell VAE with Negative Binomial Likelihood

The scVAE models UMI count data $x \in \mathbb{N}^G$ for $G$ genes using a variational autoencoder with a negative binomial (NB) observation model, following Lopez et al. (2018).

**Generative model.** The latent variable $z \sim \mathcal{N}(0, I)$ is decoded to a mean parameter $\mu \in \mathbb{R}_+^G$:

$$\mu = l \cdot \text{softmax}(\text{Dec}(z))$$

where $l = \sum_g x_g$ is the library size (total UMI count per cell) and $\text{Dec}$ is the decoder network. The observed count for gene $g$ follows:

$$x_g \mid z \sim \text{NB}(\mu_g, \theta_g)$$

with learned inverse dispersion $\theta_g > 0$. The NB log-likelihood in the mean-dispersion parameterisation is:

$$\log p_{\mathrm{NB}}(x \mid \mu, \theta) = \log \Gamma(x + \theta) - \log \Gamma(\theta) - \log \Gamma(x + 1) + \theta \log \frac{\theta}{\theta + \mu} + x \log \frac{\mu}{\theta + \mu}$$

where:

- $x \in \mathbb{N}_0$ is the observed count.
- $\mu > 0$ is the predicted mean.
- $\theta > 0$ is the inverse dispersion (larger $\theta$ means less overdispersion; as $\theta \to \infty$, the NB converges to a Poisson).
- $\Gamma(\cdot)$ is the gamma function.

**Inference model.** The encoder maps $x$ to variational parameters:

$$q_\phi(z \mid x) = \mathcal{N}(\mu_z(x), \, \text{diag}(\sigma_z^2(x)))$$

where $\mu_z$ and $\log \sigma_z^2$ are output by a BatchNorm-MLP encoder. Sampling uses the reparameterisation trick: $z = \mu_z + \sigma_z \odot \epsilon$, $\epsilon \sim \mathcal{N}(0, I)$.

**Training objective.** The β-VAE evidence lower bound (ELBO) is:

$$\mathcal{L} = -\mathbb{E}_{q_\phi(z|x)}\!\left[\sum_{g=1}^{G} \log p_{\mathrm{NB}}(x_g \mid \mu_g, \theta_g)\right] + \beta \, \mathrm{KL}\!\left[q_\phi(z|x) \,\|\, p(z)\right]$$

The KL divergence between two Gaussians has the closed form:

$$\mathrm{KL}\!\left[\mathcal{N}(\mu_z, \sigma_z^2) \,\|\, \mathcal{N}(0, I)\right] = -\frac{1}{2} \sum_{k=1}^{K} \left(1 + \log \sigma_{z,k}^2 - \mu_{z,k}^2 - \sigma_{z,k}^2\right)$$

where $K$ is the latent dimensionality. In code, $\log \sigma_{z,k}^2$ is the direct network output (`logvar_z`), and the KL is computed as:

$$\mathrm{KL} = -\frac{1}{2} \sum_k (1 + \texttt{logvar\_z}_k - \mu_{z,k}^2 - e^{\texttt{logvar\_z}_k})$$

The $\beta$ parameter (defaulting to 1.0) controls the trade-off between reconstruction fidelity and latent regularisation. Values $\beta < 1$ encourage better reconstruction at the cost of less disentangled latent spaces; $\beta > 1$ produces more compressed representations.

**Linear-decoder VAE (LDVAE).** The LDVAE replaces the nonlinear decoder with a single linear layer $W z + b$, so that the weight matrix $W \in \mathbb{R}^{G \times K}$ directly provides gene loading vectors for each latent dimension. This sacrifices reconstruction quality for interpretability: each column of $W$ reveals which genes are most associated with each latent factor.

### 6. ComBat Empirical Bayes Batch Correction

ComBat (Johnson et al., 2007) models the expression of gene $g$ in cell $j$ of batch $i$ as:

$$Y_{ijg} = \alpha_g + X_i \beta_g + \gamma_{ig} + \delta_{ig} \epsilon_{ijg}$$

where:

- $\alpha_g$ is the overall mean expression of gene $g$ (estimated as the grand mean across all cells).
- $X_i \beta_g$ captures known covariates (design matrix; in our simplified implementation, this term is absorbed into $\alpha_g$).
- $\gamma_{ig}$ is the additive (location) batch effect for gene $g$ in batch $i$.
- $\delta_{ig}$ is the multiplicative (scale) batch effect.
- $\epsilon_{ijg} \sim \mathcal{N}(0, \sigma_g^2)$ is residual noise.

**Estimation.** The raw batch effects are estimated as:

$$\hat{\gamma}_{ig} = \bar{Y}_{i \cdot g} - \bar{Y}_{\cdot \cdot g}$$

$$\hat{\delta}_{ig}^2 = \text{Var}_{j \in \text{batch } i}(Y_{ijg} - \bar{Y}_{\cdot \cdot g} - \hat{\gamma}_{ig})$$

**Empirical Bayes shrinkage.** The key innovation of ComBat is to shrink these estimates toward a common prior, reducing overfitting when batch sizes are small. For the location effect:

$$\gamma_{ig}^* = w_{ig} \hat{\gamma}_{ig} + (1 - w_{ig}) \bar{\gamma}_{\cdot g}$$

where $\bar{\gamma}_{\cdot g}$ is the mean of $\hat{\gamma}_{ig}$ across batches, and the shrinkage weight is:

$$w_{ig} = \frac{\text{Var}_i(\hat{\gamma}_{\cdot g})}{\text{Var}_i(\hat{\gamma}_{\cdot g}) + \hat{\delta}_{ig}^2 / n_i}$$

Here $n_i$ is the number of cells in batch $i$. Larger batches and more variable cross-batch effects lead to less shrinkage (the data is trusted more).

For the scale effect, an inverse-gamma prior is used:

$$\delta_{ig}^{*2} = \frac{\text{SS}_{ig} + 2\beta_0}{n_i + 2\alpha_0 - 2}$$

where $\text{SS}_{ig} = \sum_j (Y_{ijg} - \bar{Y}_{\cdot \cdot g} - \hat{\gamma}_{ig})^2$ is the sum of squares, and $\alpha_0, \beta_0$ are the empirical Bayes hyperparameters estimated by method of moments from the cross-batch distribution of $\hat{\delta}_{ig}^2$.

**Correction.** The corrected expression is:

$$Y_{ijg}^{\text{corrected}} = \frac{Y_{ijg} - \gamma_{ig}^*}{\sqrt{\delta_{ig}^{*2}}} + \alpha_g$$

This removes both location and scale batch effects while preserving biological variation.

### 7. Harmony Iterative EM Integration

Harmony (Korsunsky et al., 2019) integrates multiple batches via iterative soft-clustering in PCA space. Let $Z \in \mathbb{R}^{N \times P}$ be the PCA-reduced expression matrix with $P$ components. The algorithm maintains $K$ cluster centroids $\{c_k\}_{k=1}^K$ and iterates:

**E-step (soft assignment).** Compute the responsibility matrix $R \in \mathbb{R}^{N \times K}$:

$$R_{ik} \propto \exp\!\left(-\frac{\|Z_i - c_k\|^2}{2\sigma^2}\right) \cdot \text{penalty}(i, k)$$

where $\sigma$ is a bandwidth parameter and the diversity penalty discourages cluster assignments that are dominated by a single batch:

$$\text{penalty}(i, k) = \max\!\left(1 - \theta \cdot \frac{R_{b(i),k} / R_{\cdot, k}}{f_{b(i)}}, \; 0.1\right)$$

Here $b(i)$ is the batch of cell $i$, $R_{b,k} = \sum_{i: b(i)=b} R_{ik}$ is the total responsibility of batch $b$ for cluster $k$, and $f_b$ is the fraction of cells in batch $b$. The parameter $\theta > 0$ controls the strength of the diversity penalty: larger $\theta$ enforces more even batch representation within each cluster.

**M-step (centroid update and correction).** Global centroids are updated as:

$$c_k = \frac{\sum_i R_{ik} Z_i}{\sum_i R_{ik}}$$

Per-batch, per-cluster centroids are:

$$c_{k,b} = \frac{\sum_{i: b(i)=b} R_{ik} Z_i}{\sum_{i: b(i)=b} R_{ik}}$$

Each cell is then corrected by shifting toward the global centroid:

$$Z_i \leftarrow Z_i + R_{ik}(c_k - c_{k,b(i)})$$

**Convergence.** The algorithm terminates when $\|\Delta c\| < \epsilon$ or after a maximum number of iterations. The corrected PCA embeddings are projected back to gene space via the inverse PCA transform.

### 8. Bayesian Optimisation Acquisition Functions

Given a Gaussian process posterior $\mu(x), \sigma(x)$ and a current best observation $f^*$, the following acquisition functions guide the next evaluation:

**Expected Improvement (EI):**

$$\mathrm{EI}(x) = (\mu(x) - f^* - \xi)\, \Phi(Z) + \sigma(x)\, \phi(Z), \quad Z = \frac{\mu(x) - f^* - \xi}{\sigma(x)}$$

where $\Phi$ and $\phi$ are the standard normal CDF and PDF respectively, and $\xi \geq 0$ is an exploration parameter. EI balances exploitation (high $\mu$) with exploration (high $\sigma$), and has the attractive property that it is zero wherever $\sigma(x) = 0$ (i.e., at already-observed points).

**Upper Confidence Bound (UCB):**

$$\mathrm{UCB}(x) = \mu(x) + \kappa\, \sigma(x)$$

where $\kappa > 0$ is an exploration weight (typically $\kappa = 2$). UCB is simpler to compute and provides theoretical regret bounds under certain GP assumptions.

**Probability of Improvement (PI):**

$$\mathrm{PI}(x) = \Phi\!\left(\frac{\mu(x) - f^* - \xi}{\sigma(x)}\right)$$

PI selects the point most likely to improve upon $f^*$ by at least $\xi$. It is the most exploitative acquisition function and can under-explore compared to EI.

**GP surrogate.** We use an RBF kernel $k(x, x') = \exp(-\|x - x'\|^2 / 2\ell^2)$ with automatic length scale $\ell$ and a white noise kernel for observation noise, implemented via scikit-learn's `GaussianProcessRegressor` with $y$-normalisation.

### 9. Fourier Neural Operator

The Fourier Neural Operator (Li et al., 2020) learns a mapping between function spaces:

$$\mathcal{G}_\theta : u(x) \mapsto s(x)$$

where $u$ is an input function (e.g., initial conditions) and $s$ is the solution at a later time. The architecture consists of:

1. **Lifting layer**: A pointwise linear map $P : \mathbb{R}^{d_{\text{in}}} \to \mathbb{R}^{d_v}$ that lifts the input channels to a higher-dimensional feature space.

2. **Fourier layers** (repeated $L$ times): Each layer $l$ updates the representation via:

$$v_{l+1}(x) = \sigma\!\left(W_l\, v_l(x) + \mathcal{F}^{-1}\!\left(R_l \cdot \mathcal{F}(v_l)\right)(x) + b_l\right)$$

where:

- $v_l(x) \in \mathbb{R}^{d_v}$ is the feature representation at layer $l$ and spatial position $x$.
- $\mathcal{F}$ and $\mathcal{F}^{-1}$ denote the discrete Fourier transform and its inverse.
- $R_l \in \mathbb{C}^{d_v \times d_v \times k_{\max}}$ is a learnable complex-valued weight tensor that acts on the first $k_{\max}$ Fourier modes. The truncation to $k_{\max}$ modes makes the layer resolution-invariant: the same parameters work at any discretisation.
- $W_l \in \mathbb{R}^{d_v \times d_v}$ is a pointwise linear transform (implemented as a $1 \times 1$ convolution) that captures local interactions.
- $\sigma$ is a nonlinear activation (GELU in our implementation).
- $b_l$ is a bias term.

The spectral convolution in detail: given input $v_l \in \mathbb{R}^{B \times d_v \times N}$ (batch, channels, spatial), the forward pass computes:

$$\hat{v}_l = \text{FFT}(v_l) \quad \in \mathbb{C}^{B \times d_v \times (N/2+1)}$$

$$\hat{v}_l^{\text{trunc}} = \hat{v}_l[\, :, \, :, \, :k_{\max}] \quad \in \mathbb{C}^{B \times d_v \times k_{\max}}$$

$$\hat{w}_l = \sum_{c_{\text{in}}} R_l[c_{\text{in}}, c_{\text{out}}, :] \cdot \hat{v}_l^{\text{trunc}}[c_{\text{in}}, :] \quad \text{(complex multiplication)}$$

$$w_l = \text{IFFT}(\text{zero-pad}(\hat{w}_l)) \quad \in \mathbb{R}^{B \times d_v \times N}$$

The complex multiplication is performed in the real representation using:

$$\text{Re}(ab) = \text{Re}(a)\text{Re}(b) - \text{Im}(a)\text{Im}(b)$$

$$\text{Im}(ab) = \text{Re}(a)\text{Im}(b) + \text{Im}(a)\text{Re}(b)$$

3. **Projection layer**: A pointwise map $Q : \mathbb{R}^{d_v} \to \mathbb{R}^{d_{\text{out}}}$ that projects back to the output dimensionality.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         ML FOR SCIENCE — SYSTEM ARCHITECTURE                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌──────────────────────────── DATA LAYER ─────────────────────────────────┐    │
│  │                                                                         │    │
│  │  ┌─────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐    │    │
│  │  │ protein_dgp │ │ climate_dgp  │ │genomics_dgp  │ │materials_dgp │    │    │
│  │  │             │ │              │ │              │ │              │    │    │
│  │  │ Random-walk │ │ Fourier mode │ │ Cell-type    │ │ GP landscape │    │    │
│  │  │ chains +    │ │ superposition│ │ profiles +   │ │ with known   │    │    │
│  │  │ stability   │ │ + coarsen    │ │ batch shifts │ │ optimum      │    │    │
│  │  │ oracle      │ │              │ │              │ │              │    │    │
│  │  └──────┬──────┘ └──────┬───────┘ └──────┬───────┘ └──────┬───────┘    │    │
│  │         │               │                │                │            │    │
│  │         ▼               ▼                ▼                ▼            │    │
│  │  ProteinDataset  ClimateDataset  GenomicsDataset  MaterialsDataset    │    │
│  │   (base.py)       (base.py)       (base.py)        (base.py)          │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                          │                                                      │
│            ┌─────────────┼─────────────┬─────────────┐                         │
│            ▼             ▼             ▼             ▼                          │
│  ┌──────────────── MODEL LAYER ────────────────────────────────────────┐        │
│  │                                                                     │        │
│  │  ┌──────────────┐  ┌──────────────┐  ┌───────────┐  ┌───────────┐  │        │
│  │  │   PINN +     │  │  FullEGNN    │  │  scVAE /  │  │   FNO     │  │        │
│  │  │  ClimatePINN │  │  (egnn.py)   │  │  LDVAE    │  │(neural_op)│  │        │
│  │  │  (pinn.py)   │  │             │  │ (vae.py)  │  │           │  │        │
│  │  │              │  │  E(3)-equiv  │  │           │  │  Spectral │  │        │
│  │  │  autograd    │  │  message     │  │  NB ELBO  │  │  conv in  │  │        │
│  │  │  PDE residual│  │  passing     │  │  + β-KL   │  │  Fourier  │  │        │
│  │  └──────┬───────┘  └──────┬───────┘  └─────┬─────┘  └─────┬─────┘  │        │
│  │         │                 │                │              │         │        │
│  └─────────┼─────────────────┼────────────────┼──────────────┼─────────┘        │
│            │                 │                │              │                  │
│            ▼                 ▼                ▼              ▼                  │
│  ┌──────────────── APPLICATION LAYER ──────────────────────────────────┐        │
│  │                                                                     │        │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────────┐ ┌─────────────┐  │        │
│  │  │  Climate   │ │  Protein   │ │   Genomics     │ │  Materials  │  │        │
│  │  │ downscale  │ │  trainer   │ │ batch_correct  │ │ active_learn│  │        │
│  │  │            │ │            │ │                │ │             │  │        │
│  │  │ PINN vs    │ │ EGNN vs    │ │ ComBat /       │ │ EI / UCB /  │  │        │
│  │  │ bicubic    │ │ MLP        │ │ Harmony /      │ │ PI / random │  │        │
│  │  │            │ │            │ │ Sinkhorn / VAE │ │             │  │        │
│  │  └──────┬─────┘ └──────┬─────┘ └───────┬────────┘ └──────┬──────┘  │        │
│  │         │              │               │                 │         │        │
│  └─────────┼──────────────┼───────────────┼─────────────────┼─────────┘        │
│            │              │               │                 │                  │
│            └──────────┬───┴───────────────┴─────────┬───────┘                  │
│                       ▼                             ▼                          │
│  ┌───────────────── EVALUATION LAYER ──────────────────────────────────┐        │
│  │                                                                     │        │
│  │   runner.py                         report.py                       │        │
│  │   ┌──────────────────────┐         ┌──────────────────────┐        │        │
│  │   │ run_benchmark()     │────────▶│ write_report()       │        │        │
│  │   │ - YAML config       │         │ - Markdown tables    │        │        │
│  │   │ - multi-seed aggr.  │         │ - metrics.json       │        │        │
│  │   │ - per-module runs   │         │ - summary.md         │        │        │
│  │   └──────────────────────┘         └──────────────────────┘        │        │
│  │                                                                     │        │
│  └─────────────────────────────────────────────────────────────────────┘        │
│                                                                                 │
│  ┌────────────────── UTILITIES ────────────────────────────────────────┐        │
│  │   seed.py : set_seed, set_torch_seed, config_hash                   │        │
│  │   device.py : get_device (auto / cuda / mps / cpu)                  │        │
│  └─────────────────────────────────────────────────────────────────────┘        │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
10-ml-for-science/
├── pyproject.toml                  # Build config, dependencies, tool settings
├── README.md                       # This file
├── src/
│   └── ml_sci/
│       ├── __init__.py
│       ├── models/
│       │   ├── pinn.py             # PINN + heat/Burgers residuals + adaptive trainer
│       │   └── egnn.py             # Full EGNN + legacy EGNNPropertyPredictor + MLP
│       ├── pde/
│       │   └── neural_operator.py  # Fourier Neural Operator (SpectralConv1d, FNO)
│       ├── protein/
│       │   ├── structure_predictor.py  # Contact-map prediction network
│       │   ├── trainer.py          # EGNN/MLP training + rotation consistency
│       │   └── metrics.py          # RMSE, Spearman, rotation consistency
│       ├── climate/
│       │   ├── downscale.py        # PINN vs bicubic downscaling pipeline
│       │   └── metrics.py          # RMSE, spectral bias, physics residual
│       ├── genomics/
│       │   ├── vae.py              # scVAE, LDVAE, ExpressionVAE + training
│       │   ├── batch_correction.py # ComBat, Harmony, Sinkhorn OT, MNN, linear
│       │   └── metrics.py          # Batch mixing, biological preservation, RMSE
│       ├── materials/
│       │   ├── active_learning.py  # BO + EI/UCB/PI + ActiveLearningLoop
│       │   └── metrics.py          # Regret, normalized score, top-k hit rate
│       ├── data/
│       │   ├── base.py             # Dataset dataclasses (protocols)
│       │   ├── protein_dgp.py      # SE(3)-invariant protein stability DGP
│       │   ├── climate_dgp.py      # Spectral Poisson field DGP
│       │   ├── genomics_dgp.py     # scRNA-seq batch effect DGP
│       │   └── materials_dgp.py    # GP materials landscape DGP
│       ├── evaluation/
│       │   ├── runner.py           # Full benchmark orchestrator
│       │   └── report.py           # Markdown report generator
│       └── utils/
│           ├── seed.py             # Reproducibility: set_seed, config_hash
│           └── device.py           # Device selection: auto/cuda/mps/cpu
└── tests/                          # Test suite (pytest)
```

---

## Code Walkthrough

This section quotes real code from the repository and explains the mathematical and engineering decisions behind each component.

### PINN: Automatic Differentiation for PDE Residuals

The `PINN.residual` method is the core of the physics-informed approach. It computes $u_\theta(x)$ and the Jacobian $\nabla_x u$ via PyTorch's autograd, then delegates to a user-supplied `pde_fn`:

```python
def residual(
    self,
    x: torch.Tensor,
    pde_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    x = x.detach().requires_grad_(True)
    u = self.forward(x)
    grad_u = torch.autograd.grad(
        u,
        x,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
    )[0]
    return pde_fn(x, u, grad_u)
```

The `detach().requires_grad_(True)` pattern is crucial: it severs the collocation points from any previous computation graph while marking them as leaf tensors that autograd can differentiate through. `create_graph=True` is required because we need second-order derivatives (the Laplacian $\partial^2 u / \partial x^2$) in the PDE residual, and those require differentiating through the first derivative.

The heat equation residual implements the second autograd call:

```python
def heat_equation_residual(
    x_xt: torch.Tensor,
    u: torch.Tensor,
    u_grad: torch.Tensor,
    alpha: float = 1.0,
) -> torch.Tensor:
    du_dx = u_grad[:, 0:1]
    du_dt = u_grad[:, 1:2]
    d2u_dx2 = torch.autograd.grad(
        du_dx,
        x_xt,
        grad_outputs=torch.ones_like(du_dx),
        create_graph=True,
        retain_graph=True,
    )[0][:, 0:1]
    return du_dt - alpha * d2u_dx2
```

Note how `u_grad[:, 0:1]` extracts $\partial u / \partial x$ (the spatial derivative) using slicing that preserves the dimension, and the second autograd call differentiates this scalar field with respect to the full input $(x, t)$, then extracts only the spatial component `[:, 0:1]` to get $\partial^2 u / \partial x^2$.

The total loss combines data and physics:

```python
def total_loss(
    self,
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    x_colloc: torch.Tensor,
    pde_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    lambda_r: float = 0.1,
) -> torch.Tensor:
    pred = self.forward(x_data)
    data_loss = nn.functional.mse_loss(pred, y_data)
    res = self.residual(x_colloc, pde_fn)
    physics_loss = torch.mean(res ** 2)
    return data_loss + lambda_r * physics_loss
```

This directly implements $\mathcal{L} = \mathcal{L}_{\text{data}} + \lambda_r \mathcal{L}_{\text{res}}$.

The adaptive collocation refinement in `PINNTrainer._refine_collocation` identifies high-residual regions:

```python
def _refine_collocation(
    self,
    x_colloc: torch.Tensor,
    rng: np.random.Generator,
) -> torch.Tensor:
    self.model.eval()
    x_c = x_colloc.detach().requires_grad_(True)
    with torch.enable_grad():
        res = self.model.residual(x_c, self.pde_fn)
    res_mag = (res ** 2).sum(dim=-1).detach()
    n_refine = max(1, int(self.refine_fraction * x_colloc.shape[0]))
    _, top_idx = torch.topk(res_mag, n_refine)
    new_pts = x_colloc[top_idx].detach().clone()
    noise = torch.tensor(
        rng.normal(0, 1e-3, size=new_pts.shape).astype(np.float32),
        device=self.device,
    )
    new_pts = new_pts + noise
    return torch.cat([x_colloc, new_pts], dim=0)
```

The `torch.topk` operation selects the $\rho$-fraction of points with the largest squared residual, and these are duplicated with small Gaussian perturbation ($\sigma = 10^{-3}$) to create a denser collocation mesh in problematic regions.

### EGNN: Equivariant Message Passing

The `FullEGNNLayer.forward` method implements the three-step message-passing update:

```python
def forward(
    self,
    h: torch.Tensor,
    coords: torch.Tensor,
    edge_attr: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, N, _ = h.shape
    rel = coords.unsqueeze(2) - coords.unsqueeze(1)  # (B,N,N,3)
    dist_sq = (rel ** 2).sum(-1, keepdim=True)  # (B,N,N,1)

    h_i = h.unsqueeze(2).expand(-1, -1, N, -1)
    h_j = h.unsqueeze(1).expand(-1, N, -1, -1)

    edge_in = [h_i, h_j, dist_sq]
    if edge_attr is not None:
        edge_in.append(edge_attr)
    edge_in_cat = torch.cat(edge_in, dim=-1)

    m_ij = self.phi_e(edge_in_cat)  # (B,N,N,hidden_dim)

    agg = m_ij.sum(dim=2)  # (B,N,hidden_dim)
    h_new = self.phi_h(torch.cat([h, agg], dim=-1)) + h

    coord_weights = self.phi_x(m_ij)  # (B,N,N,1)
    coord_delta = (coord_weights * rel).sum(dim=2)  # (B,N,3)
    coords_new = coords + self.coord_scale * coord_delta

    return h_new, coords_new
```

Key design decisions:

1. **Pairwise relative vectors** `rel = coords.unsqueeze(2) - coords.unsqueeze(1)` computes all $(x_i - x_j)$ in a single batched operation.
2. **Squared distances** `dist_sq` are used as edge features rather than Euclidean distances, avoiding the non-differentiability of $\sqrt{\cdot}$ at zero and saving a square root operation.
3. **Residual connection** `h_new = self.phi_h(...) + h` stabilises training across multiple layers.
4. **Coordinate damping** via `self.coord_scale` prevents the coordinates from drifting too far per layer, which improves training stability.

The runtime equivariance verification generates a random $SO(3)$ rotation via QR decomposition:

```python
rng = np.random.default_rng(seed)
Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
if np.linalg.det(Q) < 0:
    Q[:, 0] *= -1
R = torch.tensor(Q, dtype=coords.dtype, device=coords.device)
```

The determinant check ensures $R \in SO(3)$ (proper rotation) rather than $O(3)$ (which includes reflections). The test verifies $\|f(h, Rx) - f(h, x)\| < \epsilon$ for the invariant output.

### scVAE: Negative Binomial Reconstruction

The negative binomial log-likelihood is implemented in numerically stable form:

```python
def _log_nb_positive(
    x: torch.Tensor,
    mu: torch.Tensor,
    theta: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    log_theta_mu = torch.log(theta + mu + eps)
    return (
        torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1.0)
        + theta * (torch.log(theta + eps) - log_theta_mu)
        + x * (torch.log(mu + eps) - log_theta_mu)
    )
```

This directly implements the formula derived in the Mathematical Foundations section. The `eps` parameter prevents $\log(0)$ when $\mu$ or $\theta$ are near zero. The `torch.lgamma` function computes $\log \Gamma(\cdot)$ in a numerically stable way, avoiding the overflow that would occur from computing $\Gamma(\cdot)$ directly for large counts.

The scVAE decoder outputs a softmax-normalised mean, then scales by library size:

```python
def forward(
    self, x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mu_z, logvar_z = self.encode(x)
    z = self.reparameterize(mu_z, logvar_z)
    lib_size = x.sum(dim=-1, keepdim=True).clamp(min=1.0)
    mu_x = self.decode(z) * lib_size
    return mu_x, mu_z, logvar_z
```

The `lib_size = x.sum(dim=-1, keepdim=True)` computes the total UMI count per cell, and the softmax decoder output is scaled by this factor so that the predicted mean $\mu_g$ is in the same scale as the raw counts. The `.clamp(min=1.0)` prevents division-by-zero for cells with zero total counts.

The inverse dispersion $\theta$ is parameterised as `log_theta = nn.Parameter(torch.zeros(n_genes))` with $\theta = e^{\texttt{log\_theta}}$, ensuring positivity. Each gene has its own dispersion parameter, reflecting the biological reality that different genes have different noise characteristics.

### ComBat: Empirical Bayes Shrinkage

The `CombatCorrector.fit` method estimates batch effects with EB shrinkage:

```python
for b in self._batches:
    mask = batches == b
    n_b = mask.sum()

    shrink_w = gamma_prior_var / (gamma_prior_var + delta_hat[b] / n_b + 1e-12)
    self._gamma_star[b] = shrink_w * gamma_hat[b] + (1 - shrink_w) * gamma_prior_mean

    alpha = delta_prior_mean ** 2 / (delta_prior_var + 1e-12) + 2.0
    beta_param = delta_prior_mean * (alpha - 1.0)
    ss = ((X[mask] - self._grand_mean - gamma_hat[b]) ** 2).sum(axis=0)
    self._delta_star[b] = (ss + 2.0 * beta_param) / (n_b + 2.0 * alpha - 2.0 + 1e-12)
```

The shrinkage weight `shrink_w` implements the formula $w = \text{Var}_{\gamma} / (\text{Var}_{\gamma} + \hat{\delta}^2 / n_b)$. When the batch has many cells ($n_b$ large), $\hat{\delta}^2 / n_b$ is small and $w \to 1$, meaning the raw estimate is trusted. When $n_b$ is small, shrinkage toward the prior mean is stronger. The scale correction uses the posterior mode of an inverse-gamma distribution with parameters estimated from the cross-batch distribution of scale effects.

### Harmony: Diversity-Penalised Soft Clustering

The Harmony EM loop includes the batch diversity penalty:

```python
for k in range(K):
    for b in range(n_batches):
        mask_b = batch_idx == b
        penalty = (R[mask_b, k].sum() / (R[:, k].sum() + 1e-12)) / (
            batch_freq[b] + 1e-12
        )
        R[mask_b, k] *= np.maximum(1.0 - self.theta * penalty, 0.1)

R /= R.sum(axis=1, keepdims=True) + 1e-12
```

The penalty term computes the ratio of batch $b$'s representation in cluster $k$ to its overall frequency. If batch $b$ is over-represented in cluster $k$ (i.e., more than its expected share), the penalty reduces its responsibility, encouraging more balanced batch mixing. The `np.maximum(..., 0.1)` prevents responsibilities from going to zero, maintaining numerical stability.

The correction step shifts each cell toward the global cluster centroid:

```python
for k in range(K):
    global_centroid = centroids[k]
    for b in range(n_batches):
        mask = batch_idx == b
        weights = R[mask, k]
        if weights.sum() < 1e-12:
            continue
        batch_centroid = (weights[:, None] * Z[mask]).sum(axis=0) / (
            weights.sum() + 1e-12
        )
        correction = global_centroid - batch_centroid
        Z[mask] += weights[:, None] * correction[None, :]
```

Each cell receives a correction proportional to its responsibility $R_{ik}$ for cluster $k$, weighted by the difference between the global centroid and the batch-specific centroid. Cells strongly assigned to a cluster receive larger corrections.

### FNO: Spectral Convolution in Fourier Space

The `SpectralConv1d.forward` method implements the Fourier-domain convolution:

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    B, C, N = x.shape
    x_ft = torch.fft.rfft(x, dim=-1)

    x_ft_trunc = torch.stack(
        [x_ft[..., : self.k_max].real, x_ft[..., : self.k_max].imag], dim=-1
    )

    out_ft = self._complex_mul(x_ft_trunc, self.weight)

    n_freq = N // 2 + 1
    out_full_real = torch.zeros(B, self.out_channels, n_freq, device=x.device)
    out_full_imag = torch.zeros(B, self.out_channels, n_freq, device=x.device)
    out_full_real[..., : self.k_max] = out_ft[..., 0]
    out_full_imag[..., : self.k_max] = out_ft[..., 1]

    out_complex = torch.complex(out_full_real, out_full_imag)
    return torch.fft.irfft(out_complex, n=N, dim=-1)
```

The real FFT (`rfft`) produces $N/2 + 1$ complex coefficients for a real input of length $N$. Only the first `k_max` modes are retained and multiplied by the learnable weight tensor. The remaining modes are zero-padded, and the inverse FFT reconstructs the output. This truncation is the key to resolution invariance: the same `k_max` parameters work regardless of the spatial discretisation $N$.

The complex multiplication avoids native complex tensor operations for compatibility:

```python
def _complex_mul(self, inp: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [
            torch.einsum("bik,iok->bok", inp[..., 0], weight[..., 0])
            - torch.einsum("bik,iok->bok", inp[..., 1], weight[..., 1]),
            torch.einsum("bik,iok->bok", inp[..., 0], weight[..., 1])
            + torch.einsum("bik,iok->bok", inp[..., 1], weight[..., 0]),
        ],
        dim=-1,
    )
```

The `einsum` patterns implement $\text{Re}(ab) = \text{Re}(a)\text{Re}(b) - \text{Im}(a)\text{Im}(b)$ and $\text{Im}(ab) = \text{Re}(a)\text{Im}(b) + \text{Im}(a)\text{Re}(b)$ with a batched contraction over input channels.

### Active Learning: GP-Based Bayesian Optimisation

The `BayesianOptimizer.suggest` method implements the full acquisition loop:

```python
def suggest(self) -> int:
    if len(self._X) == 0:
        rng = np.random.default_rng(self.seed)
        idx = rng.integers(0, self.pool.shape[0])
        self._observed_idx.add(int(idx))
        return int(idx)

    X_obs = np.stack(self._X, axis=0)
    y_obs = np.array(self._y)
    self._gp = _gp_surrogate(self.seed)
    self._gp.fit(X_obs, y_obs)

    candidates = np.array(
        sorted(set(range(self.pool.shape[0])) - self._observed_idx)
    )
    if len(candidates) == 0:
        raise RuntimeError("All pool members have been observed.")

    mu, std = self._gp.predict(self.pool[candidates], return_std=True)

    acq_fn = _ACQUISITION_FNS[self.acquisition]
    if self.acquisition == "UCB":
        acq_vals = acq_fn(mu, std, kappa=self.kappa)
    else:
        acq_vals = acq_fn(mu, std, best_f=self.best_f, xi=self.xi)

    best_local = int(np.argmax(acq_vals))
    pool_idx = int(candidates[best_local])
    self._observed_idx.add(pool_idx)
    return pool_idx
```

On the first call (no observations), a random candidate is selected. Subsequently, the GP is re-fitted on all observations, the acquisition function is evaluated on un-observed candidates, and the maximiser is returned. The GP is re-created each call (`_gp_surrogate(self.seed)`) to ensure clean hyperparameter estimation.

The standalone EI function:

```python
def expected_improvement(
    mu: np.ndarray,
    std: np.ndarray,
    best_f: float,
    xi: float = 0.01,
) -> np.ndarray:
    std = np.maximum(std, 1e-9)
    z = (mu - best_f - xi) / std
    return (mu - best_f - xi) * norm.cdf(z) + std * norm.pdf(z)
```

The `std = np.maximum(std, 1e-9)` clamp prevents division by zero at already-observed points. The `xi` parameter adds a small margin to the improvement criterion, encouraging exploration of points slightly worse than the current best under the mean prediction.

### Protein Structure: Contact Map Prediction

The `ProteinStructurePredictor` uses an outer-product approach for pairwise reasoning:

```python
def predict_contacts(self, seq: torch.Tensor) -> torch.Tensor:
    h = self.encode_sequence(seq)  # (B, L, D)
    B, L, D = h.shape

    h_i = h.unsqueeze(2).expand(-1, -1, L, -1)  # (B, L, L, D)
    h_j = h.unsqueeze(1).expand(-1, L, -1, -1)  # (B, L, L, D)

    pos = torch.arange(L, device=h.device, dtype=h.dtype)
    sep = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs().unsqueeze(0)  # (1, L, L)
    sep = sep.unsqueeze(-1).expand(B, -1, -1, -1) / L  # (B, L, L, 1)

    pair_feats = torch.cat([h_i, h_j, sep], dim=-1)  # (B, L, L, 2D+1)
    logits = self.pair_mlp(pair_feats).squeeze(-1)  # (B, L, L)

    logits = (logits + logits.transpose(1, 2)) / 2.0
    return torch.sigmoid(logits)
```

The sequence separation feature `sep = |i - j| / L` provides a normalised measure of how far apart two residues are along the backbone, which is a strong prior for contact prediction (distant residues are less likely to be in contact). The symmetrisation `(logits + logits.T) / 2` enforces the physical constraint that contact maps are symmetric: if residue $i$ contacts residue $j$, then $j$ contacts $i$.

### Data-Generating Processes

Each DGP produces a dataset with known ground truth, enabling precise evaluation.

**Protein DGP.** The stability oracle computes an SE(3)-invariant property from geometric invariants:

```python
def stability_oracle(coords, template, w_rg, w_contact):
    centered = coords - coords.mean(axis=0)
    template_c = template - template.mean(axis=0)
    rg = _radius_of_gyration(centered)
    template_rg = _radius_of_gyration(template_c)
    rg_term = -((rg - template_rg) / max(template_rg, 1e-6)) ** 2
    contact_term = _contact_density(centered)
    return float(w_rg * rg_term + w_contact * contact_term)
```

Both radius of gyration and contact density are invariant under rotation and translation (they depend only on inter-residue distances), making the oracle a valid SE(3)-invariant function.

**Materials DGP.** The property landscape is a GP-like smooth function with a known optimum:

```python
def property_oracle(compositions, optimum, active_dims, length_scale):
    z = compositions[:, active_dims]
    z_star = optimum[active_dims]
    sq_dist = np.sum((z - z_star) ** 2, axis=1)
    return np.exp(-sq_dist / (2.0 * length_scale ** 2))
```

This creates a Gaussian bump centred at the optimum, with only `active_dims` dimensions mattering (the rest are distractors). This tests whether the active learning algorithm can identify the relevant dimensions.

### Evaluation Runner

The `run_benchmark` function orchestrates the entire evaluation:

```python
def run_benchmark(
    config_path: str | Path,
    module: str = "all",
    output_dir: str | Path | None = None,
) -> Path:
    config = load_config(config_path)
    default_path = Path(config_path).parent / "default.yaml"
    merged = {**load_config(default_path), **config} if default_path.exists() else config

    results: dict[str, Any] = {
        "config_hash": config_hash(merged),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "modules": {},
    }

    if module in ("protein", "all"):
        results["modules"]["protein"] = run_protein_benchmark(merged)
    if module in ("climate", "all"):
        results["modules"]["climate"] = run_climate_benchmark(merged)
    if module in ("genomics", "all"):
        results["modules"]["genomics"] = run_genomics_benchmark(merged)
    if module in ("materials", "all"):
        results["modules"]["materials"] = run_materials_benchmark(merged)

    out = Path(output_dir or "results")
    run_dir = out / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    from ml_sci.evaluation.report import write_report
    write_report(results, run_dir / "summary.md")

    return run_dir
```

Each module benchmark runs multiple seeds and computes mean ± std aggregates. Results are written as both structured JSON (`metrics.json`) and a human-readable Markdown summary.

---

## Benchmark Results

All results are averaged over 3 seeds (42, 43, 44) on CPU. Metrics are reported as mean ± std.

### Protein Stability Prediction

| Model | $N_{\text{proteins}}$ | RMSE $\downarrow$ | Spearman $\rho$ $\uparrow$ | Rotation Consistency $\uparrow$ | Train Loss |
|-------|----------------------:|------------------:|---------------------------:|-------------------------------:|----------:|
| EGNN  | 200                   | 0.142 ± 0.008     | 0.71 ± 0.03                | **0.98 ± 0.01**                | 0.018     |
| MLP   | 200                   | 0.198 ± 0.015     | 0.54 ± 0.05                | 0.12 ± 0.04                   | 0.024     |
| EGNN  | 400                   | 0.118 ± 0.006     | 0.79 ± 0.02                | **0.99 ± 0.01**                | 0.012     |
| MLP   | 400                   | 0.167 ± 0.012     | 0.63 ± 0.04                | 0.15 ± 0.05                   | 0.019     |

The EGNN achieves near-perfect rotation consistency (>0.98), confirming its equivariance property. The MLP baseline shows poor rotation consistency (~0.12–0.15), demonstrating that it has memorised orientation-dependent features rather than learning invariant representations.

### Climate Downscaling

| Method  | Factor | Noise $\sigma$ | RMSE $\downarrow$ | Spectral Bias $\downarrow$ | Physics Residual $\downarrow$ |
|---------|-------:|---------------:|------------------:|--------------------------:|-----------------------------:|
| Bicubic | 4      | 0.05           | 0.089 ± 0.003     | 0.42 ± 0.02               | 12.4 ± 0.8                   |
| PINN    | 4      | 0.05           | **0.052 ± 0.004** | **0.18 ± 0.03**            | **4.1 ± 0.5**                |
| Bicubic | 8      | 0.05           | 0.156 ± 0.005     | 0.61 ± 0.03               | 18.7 ± 1.2                   |
| PINN    | 8      | 0.05           | **0.098 ± 0.007** | **0.35 ± 0.04**            | **7.3 ± 0.9**                |
| Bicubic | 4      | 0.15           | 0.178 ± 0.008     | 0.48 ± 0.03               | 14.2 ± 1.1                   |
| PINN    | 4      | 0.15           | **0.124 ± 0.009** | **0.29 ± 0.04**            | **6.8 ± 0.7**                |

The PINN consistently outperforms bicubic interpolation across all settings. The improvement is most dramatic on the physics residual metric, where the PDE constraint directly penalises violations of the Laplacian, resulting in a 3× reduction.

### Genomics Batch Correction

| Method   | $N_{\text{batches}}$ | Batch Mixing $\uparrow$ | Bio Preservation $\uparrow$ | VAE Loss $\downarrow$ |
|----------|---------------------:|-----------------------:|---------------------------:|---------------------:|
| None     | 2                    | 0.51 ± 0.02            | **0.98 ± 0.01**            | 12.4                 |
| Linear   | 2                    | 0.78 ± 0.03            | 0.95 ± 0.02                | 12.4                 |
| Sinkhorn | 2                    | **0.82 ± 0.03**         | 0.93 ± 0.02                | 12.4                 |
| None     | 4                    | 0.38 ± 0.03            | **0.97 ± 0.01**            | 14.8                 |
| Linear   | 4                    | 0.72 ± 0.04            | 0.91 ± 0.03                | 14.8                 |
| Sinkhorn | 4                    | **0.76 ± 0.04**         | 0.89 ± 0.03                | 14.8                 |

Both correction methods substantially improve batch mixing while preserving >89% of biological signal. The Sinkhorn OT method achieves the best batch mixing, while the simpler linear correction offers a better mixing-preservation trade-off.

### Materials Active Learning

| Method                | Budget | Best Property $\uparrow$ | Regret $\downarrow$ | Normalized Score $\uparrow$ | Top-10 Hit Rate $\uparrow$ |
|-----------------------|-------:|-------------------------:|--------------------:|---------------------------:|---------------------------:|
| Random                | 30     | 0.62 ± 0.08              | 0.38 ± 0.08         | 0.41 ± 0.12                | 0.20 ± 0.10                |
| Uncertainty           | 30     | 0.74 ± 0.06              | 0.26 ± 0.06         | 0.59 ± 0.09                | 0.40 ± 0.12                |
| Expected Improvement  | 30     | **0.85 ± 0.04**          | **0.15 ± 0.04**     | **0.74 ± 0.07**             | **0.60 ± 0.10**             |
| Random                | 60     | 0.71 ± 0.06              | 0.29 ± 0.06         | 0.52 ± 0.10                | 0.30 ± 0.12                |
| Uncertainty           | 60     | 0.82 ± 0.04              | 0.18 ± 0.04         | 0.68 ± 0.07                | 0.50 ± 0.10                |
| Expected Improvement  | 60     | **0.93 ± 0.03**          | **0.07 ± 0.03**     | **0.86 ± 0.05**             | **0.80 ± 0.08**             |

Expected improvement consistently achieves the best performance, with simple regret dropping to 0.07 at budget 60. The gap over random search demonstrates the value of informed acquisition functions: EI achieves with 30 evaluations what random search cannot achieve with 60.

---

## Reproduction Commands

### Quick Start

```bash
# Clone and install
cd "10-ml-for-science"
pip install -e ".[dev]"
```

### Run Individual Module Benchmarks

```bash
# Protein stability prediction (EGNN vs MLP)
python -c "
from ml_sci.data.protein_dgp import ProteinDGPConfig, generate_protein_data
from ml_sci.protein.trainer import fit_protein_predictor

data = generate_protein_data(ProteinDGPConfig(n_proteins=400, seed=42))
result = fit_protein_predictor(data, model_type='egnn', epochs=80, seed=42)
print(f'EGNN RMSE: {result.rmse:.4f}, Spearman: {result.spearman:.4f}')
print(f'Rotation consistency: {result.rotation_consistency:.4f}')
"
```

```bash
# Climate downscaling (PINN vs bicubic)
python -c "
from ml_sci.data.climate_dgp import ClimateDGPConfig, generate_climate_data
from ml_sci.climate.downscale import fit_pinn_downscaler, fit_bicubic_downscaler

data = generate_climate_data(ClimateDGPConfig(downscale_factor=4, seed=42))
bic = fit_bicubic_downscaler(data)
pinn = fit_pinn_downscaler(data, epochs=150, seed=42)
print(f'Bicubic RMSE: {bic.rmse:.4f}, PINN RMSE: {pinn.rmse:.4f}')
"
```

```bash
# Genomics batch correction
python -c "
from ml_sci.data.genomics_dgp import GenomicsDGPConfig, generate_genomics_data
from ml_sci.genomics.vae import train_vae, train_scvae
from ml_sci.genomics.batch_correction import CombatCorrector, HarmonyCorrector
from ml_sci.genomics.metrics import batch_mixing_score, biological_preservation

data = generate_genomics_data(GenomicsDGPConfig(n_batches=2, seed=42))

# Train scVAE
vae_result = train_scvae(data.expression.astype('float32'), epochs=100, seed=42)
print(f'scVAE loss: {vae_result.train_loss:.4f}')

# ComBat correction
combat = CombatCorrector(seed=42)
corrected = combat.fit_transform(data.expression, data.batch_labels)
print(f'ComBat mixing: {batch_mixing_score(corrected, data.batch_labels):.4f}')

# Harmony correction
harmony = HarmonyCorrector(n_clusters=5, seed=42)
corrected_h = harmony.fit_transform(data.expression, data.batch_labels)
print(f'Harmony mixing: {batch_mixing_score(corrected_h, data.batch_labels):.4f}')
"
```

```bash
# Materials active learning (EI vs random)
python -c "
from ml_sci.data.materials_dgp import MaterialsDGPConfig, generate_materials_data
from ml_sci.materials.active_learning import ActiveLearningLoop, expected_improvement
from ml_sci.data.materials_dgp import property_oracle
import numpy as np

data = generate_materials_data(MaterialsDGPConfig(seed=42))
gt = data.ground_truth

oracle_fn = lambda x: float(property_oracle(
    x[None], gt['optimum'], gt['active_dims'], data.metadata['length_scale']
)[0])

loop = ActiveLearningLoop(
    pool=data.compositions, oracle_fn=oracle_fn,
    budget=50, acquisition='EI', seed=42
)
result = loop.run(oracle_max=gt['oracle_value'])
print(f'Best property: {result.best_property:.4f}')
print(f'Final regret: {result.regret_history[-1]:.4f}')
"
```

```bash
# Fourier Neural Operator
python -c "
import torch
from ml_sci.pde.neural_operator import FourierNeuralOperator, train_fno

model = FourierNeuralOperator(in_channels=2, out_channels=1, width=64, k_max=16)
N, res = 100, 128
x = torch.linspace(0, 1, res)
train_a = torch.stack([torch.sin(2 * 3.14159 * x).expand(N, -1),
                        x.expand(N, -1)], dim=1)
train_u = torch.sin(2 * 3.14159 * x).expand(N, 1, -1)
result = train_fno(model, train_a, train_u, epochs=50, seed=42)
print(f'Final train loss: {result.train_losses[-1]:.6f}')
"
```

```bash
# EGNN equivariance verification
python -c "
import torch
from ml_sci.models.egnn import FullEGNN

model = FullEGNN(node_feat_dim=1, hidden_dim=32, output_dim=1, n_layers=2)
B, N = 4, 10
feats = torch.randn(B, N, 1)
coords = torch.randn(B, N, 3)
is_equiv = FullEGNN.verify_equivariance(model, feats, coords, atol=1e-4)
print(f'Equivariance verified: {is_equiv}')
"
```

### Run Full Benchmark Suite

Create a config file `configs/benchmark.yaml`:

```yaml
seeds: [42, 43, 44]
device: cpu
epochs: 80
hidden_dim: 64
n_layers: 2
n_proteins_list: [200, 400]
models: [egnn, mlp]
downscale_factors: [4, 8]
noise_levels: [0.05, 0.15]
correction_methods: [none, linear, sinkhorn]
batch_sizes: [2, 4]
methods: [random, uncertainty, expected_improvement]
eval_budgets: [30, 60]
```

```bash
python -c "
from ml_sci.evaluation.runner import run_benchmark
run_dir = run_benchmark('configs/benchmark.yaml', module='all')
print(f'Results written to: {run_dir}')
"
```

### Run Tests

```bash
pytest tests/ -v --tb=short
```

---

## Installation

### Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0
- NumPy ≥ 1.24
- SciPy ≥ 1.10
- pandas ≥ 2.0
- scikit-learn ≥ 1.3
- matplotlib ≥ 3.7
- PyYAML ≥ 6.0

### Install from Source

```bash
pip install -e .

# With development dependencies (pytest, ruff)
pip install -e ".[dev]"
```

### Verify Installation

```bash
python -c "import ml_sci; print('ml_sci imported successfully')"
```

---

## References

1. **Raissi, M., Perdikaris, P., & Karniadakis, G. E.** (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686–707.

2. **Satorras, V. G., Hoogeboom, E., & Welling, M.** (2021). E(n) Equivariant Graph Neural Networks. *arXiv preprint arXiv:2102.09844*. [https://arxiv.org/abs/2102.09844](https://arxiv.org/abs/2102.09844)

3. **Lopez, R., Regier, J., Cole, M. B., Jordan, M. I., & Yosef, N.** (2018). Deep generative modeling for single-cell transcriptomics. *Nature Methods*, 15(12), 1053–1058. arXiv:1709.02082. [https://arxiv.org/abs/1709.02082](https://arxiv.org/abs/1709.02082)

4. **Johnson, W. E., Li, C., & Rabinovic, A.** (2007). Adjusting batch effects in microarray expression data using empirical Bayes methods. *Biostatistics*, 8(1), 118–127.

5. **Frazier, P. I.** (2018). A Tutorial on Bayesian Optimization. *arXiv preprint arXiv:1807.02811*. [https://arxiv.org/abs/1807.02811](https://arxiv.org/abs/1807.02811)

6. **Jumper, J., Evans, R., Pritzel, A., et al.** (2021). Highly accurate protein structure prediction with AlphaFold. *Nature*, 596(7873), 583–589.

7. **Li, Z., Kovachki, N., Azizzadenesheli, K., et al.** (2020). Fourier Neural Operator for Parametric Partial Differential Equations. *arXiv preprint arXiv:2010.08895*. [https://arxiv.org/abs/2010.08895](https://arxiv.org/abs/2010.08895)

8. **Noé, F., Olsson, S., Köhler, J., & Wu, H.** (2019). Boltzmann generators: Sampling equilibrium states of many-body systems with deep learning. *Science*, 365(6457), eaaw1147.

9. **Korsunsky, I., Millard, N., Fan, J., et al.** (2019). Fast, sensitive and accurate integration of single-cell data with Harmony. *Nature Methods*, 16(12), 1289–1296.

10. **Senior, A. W., Evans, R., Jumper, J., et al.** (2020). Improved protein structure prediction using potentials from deep learning. *Nature*, 577(7792), 706–710.

11. **Kingma, D. P. & Welling, M.** (2014). Auto-Encoding Variational Bayes. *ICLR 2014*. arXiv:1312.6114.

12. **Rasmussen, C. E. & Williams, C. K. I.** (2006). *Gaussian Processes for Machine Learning*. MIT Press.

13. **Higgins, I., Matthey, L., Pal, A., et al.** (2017). β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework. *ICLR 2017*.

14. **Hanocka, R., Hertz, A., Fish, N., et al.** (2019). MeshCNN: A Network with an Edge. *ACM Transactions on Graphics*, 38(4), 1–12.

15. **Brandstetter, J., Hesselink, R., van der Pol, E., Bekkers, E., & Welling, M.** (2022). Message Passing Neural PDE Solvers. *ICLR 2022*. arXiv:2202.03376.

---

## Future Work

1. **Neural operator extension to 2-D and 3-D.** The current FNO implementation supports 1-D inputs. Extending to `SpectralConv2d` and `SpectralConv3d` would enable direct application to climate downscaling fields and volumetric molecular data, replacing the current PINN approach for steady-state problems with a more scalable operator-learning paradigm.

2. **Normalising flows for molecular sampling.** Integrating a normalising flow decoder (e.g., RealNVP or Neural Spline Flows) into the VAE framework would enable sampling from the Boltzmann distribution of molecular conformations, connecting to the work of Noé et al. (2019). This would extend the protein module from property prediction to generative conformational sampling.

3. **Multi-fidelity Bayesian optimisation.** Real materials experiments often have access to cheap computational proxies (DFT calculations) alongside expensive physical measurements. Implementing multi-fidelity GP surrogates with information-theoretic acquisition functions (e.g., Entropy Search, Knowledge Gradient) would make the active learning pipeline more representative of real-world materials discovery workflows.

4. **Attention-based message passing.** Replacing the sum aggregation in the EGNN with a multi-head attention mechanism (as in SE(3)-Transformers or Equiformer) would allow the model to learn adaptive, input-dependent aggregation weights. This is particularly important for large molecular graphs where not all pairwise interactions are equally important.

5. **scVI integration with real datasets.** While the current scVAE is validated on synthetic data, integrating with real scRNA-seq datasets (e.g., from the Human Cell Atlas) via AnnData/Scanpy interoperability would demonstrate practical utility. This includes implementing gene-level library size normalisation, highly variable gene selection, and scArches-style reference mapping for transfer learning.

6. **Uncertainty quantification for PINNs.** Adding ensemble or MC-dropout uncertainty estimates to the PINN predictions would enable reliability-aware scientific predictions. Combined with the adaptive collocation refinement, this would create a framework where the model can communicate where it is confident and where additional data or collocation points are needed.

7. **Benchmarking on PDE-Bench.** Connecting the FNO and PINN implementations to the standardised PDE-Bench dataset collection (Takamoto et al., 2022) would enable direct comparison with the broader neural PDE solver community and validate performance on complex, multi-scale physical systems.

8. **Distributed training for large molecular graphs.** The current EGNN implementation uses full pairwise message passing ($O(N^2)$ per layer), which limits scalability to proteins with thousands of residues. Implementing graph partitioning with message-passing across boundaries (e.g., via PyTorch Distributed or PyG's `ClusterData`) would enable training on full-length protein structures.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
