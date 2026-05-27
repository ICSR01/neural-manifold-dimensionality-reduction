# motor-cortex-pca

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.github.com/github/jsoldadomagraner/TReND-CaMinA/blob/main/notebooks/Zambia25/16-Wed-DimReduction/dimensionality_reduction_pca.ipynb)

An enterprise-grade, open-source Python pipeline showcasing the processing, mapping, and dimensionality reduction of large-scale, in-vivo multi-neuron population dynamics. This framework leverages **Principal Component Analysis (PCA)** to uncover the lower-dimensional neural manifold where movement preparation algorithms organize complex physiological interactions into distinct behavioral states.

---

## 📋 Table of Contents
1. [Project Overview & Rationale](#project-overview--rationale)
2. [Scientific Framework & Dataset Context](#scientific-framework--dataset-context)
3. [Mathematical Implementation Steps](#mathematical-implementation-steps)
4. [Pipeline Process & Methodology](#pipeline-process--methodology)
5. [Key Discovery Milestones & Results](#key-discovery-milestones--results)
6. [Challenges & Troubleshooting](#challenges--troubleshooting)
7. [Limitations & Scope](#limitations--scope)
8. [Quick Start & Installation](#quick-start--installation)
9. [Credits & Institutional Origins](#credits--institutional-origins)

---

## 🔬 Project Overview & Rationale

### How the Project Came About & Motivation
Modern systems neuroscience has transitioned from tracking individual neurons to recording massive neural populations simultaneously. However, analyzing hundreds of interconnected neurons produces highly complex datasets that defy straightforward visualization and analysis. This project utilizes the open-source **TReND-CaMinA** datasets to build a clean, production-ready implementation demonstrating how unsupervised machine learning can extract clear, low-dimensional behavioral trajectories from noisy, high-dimensional biological telemetry.

### What Problem It Solves
When a subject plans a physical action, hundreds of neurons fire concurrently across an interwoven cortical circuit. Attempting to track or decode this movement planning phase variable-by-variable results in an uninterpretable feature space. This pipeline solves the dimensionality challenge by projecting the population spike patterns onto a lower-dimensional subspace (the "neural manifold") where the underlying neural code becomes geometrically separated, interpretable, and visually identifiable.

---

## 🧠 Scientific Framework & Dataset Context

The empirical data evaluates the population dynamics within the **dorsal premotor cortex (PMd)** of a macaque monkey during the preparation and planning phase of a voluntary motor task.

* **Dimensionality Matrix ($X \in \mathbb{R}^{N \times D}$):** Consists of 728 distinct trials ($N$) tracking $D = 97$ simultaneously-recorded individual cortical neurons.
* **Temporal Windowing:** Spikes are integrated over a precise 200 ms tracking frame while the monkey plans a targeted arm reach.
* **Experimental Parameters:** The task tests 8 unique directional reach angles. The matrix contains exactly 91 trials per reaching orientation from the **TReND-CaMinA** course curriculum, ordered sequentially:
  * **Trials 1 to 91:** Reaching Angle 1
  * **Trials 92 to 182:** Reaching Angle 2
  * *...progressing up to Trial 728 for Reaching Angle 8.*

---

## 🛠️ Mathematical Implementation Steps

The processing pipeline drops high-level statistical library dependencies to build the PCA projection directly from foundational linear algebra:

### 1. Centering High-Dimensional Vector Clusters
To satisfy variance constraints, we first compute the empirical sample mean vector ($\boldsymbol{\mu} \in \mathbb{R}^D$) across all recorded neurons:
$$\boldsymbol{\mu} = \frac{1}{N} \sum_{n=1}^{N} \boldsymbol{x}_n$$

Every data point vector is subsequently centered to subtract baseline offsets:
$$\boldsymbol{x}_{n,\text{centered}} = \boldsymbol{x}_n - \boldsymbol{\mu}$$

### 2. Empirical Covariance Matrix Resolution
We construct the global structural dependency metrics via the empirical sample covariance matrix ($S \in \mathbb{R}^{D \times D}$):
$$S = \frac{1}{N} \sum_{n=1}^{N} (\boldsymbol{x}_n - \boldsymbol{\mu})(\boldsymbol{x}_n - \boldsymbol{\mu})^\intercal$$

### 3. Eigendecomposition Optimization
By diagonalizing the symmetric covariance matrix $S$, we isolate the orthogonal eigenvector directions ($\boldsymbol{u}_i$) paired with their corresponding scaling eigenvalues ($\lambda_i$):
$$S\boldsymbol{u}_i = \lambda_i\boldsymbol{u}_i$$

The eigenvalues are sorted in descending order ($\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_D$), and eigenvectors are rearranged accordingly to map the axes of maximum variance.

---

## 💻 Pipeline Process & Methodology

The engineering workspace separates the workflow into clear, production-ready modules to maximize reproducibility and maintainability over standard monolithic notebooks:

### Technology Stack & Tool Rationale
* **Python 3:** Selected as the primary environment due to its robust ecosystem for numerical linear algebra and scientific visualization.
* **NumPy & SciPy (`scipy.io`):** Utilized for fast matrix manipulation and parsing raw binary MATLAB structures (`.mat`) into high-performance memory buffers.
* **Matplotlib & ipympl:** Selected to provide an interactive 3D rendering backend capable of plotting multi-axis spatial clusters.

### Chronological Development Workflow
1. **Data Parsing (`src/data_loader.py`):** Ingests raw `.mat` tables, strips MATLAB-specific array wrappers, and validates input dimensions.
2. **Matrix Modeling (`src/pca_analytics.py`):** Executes centered vector operations, calculates empirical covariance, and computes the sorted eigendecomposition.
3. **Diagnostic Evaluation:** Evaluates the sorted eigenvalue spectrum to identify structural cutoffs ("elbows") in data dimensionality.
4. **Subspace Mapping:** Projects the 97-neuron spike configurations into an optimized 3D coordinate system.
5. **Interactive Visualization (`notebooks/motor_cortex_pca_walkthrough.ipynb`):** Renders the final projected trial scores color-coded by behavioral condition.

---

## 📈 Key Discovery Milestones & Results

### 1. The Population Variance Profile
Evaluating the raw population spike metrics across trials reveals highly heterogeneous firing rates. Certain sub-populations of neurons show heavy baseline firing rates, while others remain completely quiescent during specific target angles, indicating sharp directional tuning.

### 2. The Scree Spectrum "Elbow" Manifold
Plotting the square-rooted sorted eigenvalue spectrum yields a prominent structural **elbow** (a sharp step drop) immediately following the **3rd principal component axis**. 

| Component Group | Cumulative Variance Explained | System Trajectory Impact |
| :--- | :--- | :--- |
| **Top 3 PCs** | **~66.24%** | Captures the core state space; separates movement planning trajectories. |
| **PCs 4–97** | ~33.76% | Represents fine-grained trial-to-trial variance, biological drift, and sensor noise. |

This clear boundary confirms that despite tracking 97 individual variables, the underlying movement-planning program operates within a highly-constrained **three-dimensional space**.

### 3. State-Space Clustering Visualization
Projecting the 728 trials into this 3D PCA coordinate system yields excellent geometric separation. Instead of an uninterpretable cloud of 97-dimensional coordinates, the data projects into 8 distinct, well-isolated clusters. Each cluster maps directly to one of the 8 unique target reaching angles, proving that the monkey's motor planning states can be decoded linearly from population patterns.

---

## ⚡ Challenges & Troubleshooting

* **The Background Ingestion Spike:** Initial passes of raw uncentered covariance matrices caused the first principal component to align blindly with global baseline firing offsets rather than directional tuning variations. This was mitigated by implementing mandatory zero-mean centering ($\boldsymbol{x}_n - \boldsymbol{\mu}$) prior to computing the covariance matrix $S$.
* **Wide Dimensional Scaling:** Because biological spike counts do not have standardized unit constraints, neurons with naturally high firing rates can disproportionately dominate the covariance calculations. In multi-session scaling expansions, applying z-score standardization prevents high-firing baseline channels from overriding subtler directionally-tuned coordinates.

---

## ⚠️ Limitations & Scope

* **Linear Space Constraints:** PCA is restricted to identifying linear relationships between features. If the underlying neural manifold curves nonlinearly through state-space, linear PCA may require additional components to capture structures that non-linear manifold learning tools (e.g., t-SNE, UMAP, or dPCA) could compress more tightly.
* **Temporal Staticism:** This framework processes a single consolidated 200 ms time bin. It models a static snapshot of the planning state and does not track the fluid, time-varying trajectory of the neural population across the entire movement execution phase.

---

## 🚀 Quick Start & Installation

### Prerequisites
* Python 3.8 or higher
* Virtual environment tool (`venv` or `conda`)

### 1. Environment Configuration
Clone this repository and navigate into the root workspace:
```bash
git clone [https://github.com/ICSR01/motor-cortex-pca.git](https://github.com/ICSR01/motor-cortex-pca.git)
cd motor-cortex-pca
