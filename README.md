#  Masked Face Recognition System
### Robust Closed-Set Identification, 1:1 Verification, and 1:N Watchlist Search under Extreme Occlusion

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Backbone ArcFace](https://img.shields.io/badge/backbone-ArcFace%20(buffalo__l)-orange.svg)](https://github.com/deepinsight/insightface)
[![Rank-1 Accuracy](https://img.shields.io/badge/Rank--1%20Accuracy-99.60%25%20(Masked)-brightgreen.svg)]()
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.9955-success.svg)]()
[![License MIT](https://img.shields.io/badge/license-MIT-green.svg)]()

---

##  Executive Summary & Motivation

Standard deep-learning face recognition systems (trained primarily on unoccluded facial imagery) rely heavily on structural geometry across the nose, mouth, chin, and jawline. When individuals wear facial coverings—such as surgical masks, N95 respirators, cloth masks, gas masks, or eye accessories—conventional feature extractors experience severe performance degradation (typically **30% to 50% drop in verification accuracy**).

This repository presents a complete, production-ready **Masked Face Recognition System** powered by InsightFace's `buffalo_l` ArcFace backbone, enhanced with **Multi-View Occlusion-Augmented Embeddings** and a novel **Safe Online Learning Architecture**.

Without requiring expensive model retraining or fine-tuning, the system achieves **99.60% Rank-1 Identification Accuracy** and a **0.9955 ROC-AUC** across **5,768 masked query benchmark tests** spanning 5 distinct mask categories.

---

## ✨ Key Features & Technological Innovations

### 1.  Multi-View Occlusion-Augmented Embeddings ($3 \times 512$ Baseline Matrix)
During identity enrollment (`register_face.py`), the system automatically detects 5 2D facial keypoints (eyes, nose, mouth corners) and synthetically generates occluded variants:
- **Clean Baseline View**: Unoccluded facial image ($1 \times 512$ embedding).
- **Synthetic Eye Occlusion View**: Simulated sunglasses/goggles covering the periocular region ($1 \times 512$ embedding).
- **Synthetic Lower-Face Mask View**: Simulated surgical mask covering from the nose bridge down to the chin ($1 \times 512$ embedding).

These vectors are vertically concatenated into an initial $3 \times 512$ matrix per enrolled identity.

### 2.  Safe Online Learning Architecture (Dynamic In-Memory Memory Adaptation)
During live webcam streams or video playback (`live_recognition.py`, `video_face_recognition.py`), the system dynamically adapts to real-time changes in ambient lighting, facial posture, and camera angles:
- **Recognition Threshold ($\tau_{\text{rec}} = 0.35$)**: Draws green bounding boxes and labels recognized identities. Unknowns below $0.35$ are cleanly suppressed without visual clutter.
- **Safe Online Learning Threshold ($\tau_{\text{learn}} = 0.55$)**: When a live frame scores $\ge 0.55$ (extreme high confidence), the current $1 \times 512$ embedding is appended (`np.vstack`) to the identity's in-memory view matrix.
- **FIFO Buffer (50-View Memory Cap)**: Enforces a strict First-In-First-Out queue capped at 50 views, recycling memory every ~1.6 seconds during continuous streaming.
- **Zero Disk Pollution**: Modifications occur strictly in RAM, preserving original baseline `.npy` gallery files on disk.

### 3. 🎯 Robust Multi-Scale Face Detection Engine
To handle low-resolution input, partial occlusions, or small faces distant from the camera, `register_face.py` employs a 3-stage multi-resolution fallback detection pipeline:
1. Standard $640 \times 640$ @ confidence threshold $0.5$
2. Fallback $320 \times 320$ @ confidence threshold $0.3$
3. Fallback $160 \times 160$ @ confidence threshold $0.2$

### 4. 📊 Empirical Benchmarking & Research Analytics
Includes an automated benchmarking harness (`test.py`) and graph generator (`generate_research_graphs.py`) capable of parsing ground-truth identities across mask flags (`SUR`, `N95`, `K95`, `CLT`, `GAS`), computing ROC-AUC, TAR@FAR log plots, per-mask-type accuracy breakdowns, and score distribution histograms.

---

## 🏗️ System Architecture & Operational Pipeline

### 1. Enrollment & Matrix Synthesis Pipeline

```mermaid
flowchart TD
    A[Input Single Clean Image] --> B[InsightFace Face Analysis]
    B --> C[Extract 5 Facial Landmarks kps]
    
    C --> D1[Extract Raw Vector -> 1x512]
    C --> D2[Synthesize Eye Occlusion Rectangle -> 1x512]
    C --> D3[Synthesize Lower-Face Mask Polygon -> 1x512]
    
    D1 --> E[Stack Vectors into 3x512 Matrix]
    D2 --> E
    D3 --> E
    
    E --> F[Save to faces_db/identity_name.npy]
```

### 2. Live Recognition & Safe Online Learning Pipeline

```mermaid
flowchart TD
    A[Live Camera / Video Frame] --> B[Detect Face & Extract 1x512 Query Vector]
    B --> C[Compute Cosine Similarity against all gallery matrices N x 512]
    C --> D[Select Identity with max score]
    
    D --> E{Score >= 0.35?}
    E -- No --> F[Suppress Output / Reject as Unrecognized]
    E -- Yes --> G[Display Green Box & Name Tag]
    
    G --> H{Score >= 0.55?}
    H -- No --> I[Maintain Current View Matrix]
    H -- Yes --> J[np.vstack: Append 1x512 Vector]
    J --> K{Matrix length > 50?}
    K -- Yes --> L[Slice matrix[-50:] FIFO truncation]
    K -- No --> M[Update In-Memory Database]
    L --> M
```

---

## 📐 Mathematical Foundations

### 1. Cosine Similarity Formula
Given a live query feature vector $\mathbf{q} \in \mathbb{R}^{512}$ and a stored gallery view vector $\mathbf{v}_j \in \mathbb{R}^{512}$:

$$\text{Similarity}(\mathbf{q}, \mathbf{v}_j) = \frac{\mathbf{q} \cdot \mathbf{v}_j}{\|\mathbf{q}\|_2 \|\mathbf{v}_j\|_2}$$

### 2. Multi-View Maximum Similarity Matching
For an enrolled identity $i$ possessing an $N \times 512$ view matrix $\mathbf{M}_i = [\mathbf{v}_{i,1}, \mathbf{v}_{i,2}, \dots, \mathbf{v}_{i,N}]^T$:

$$S_i = \max_{j \in \{1, \dots, N\}} \text{Similarity}(\mathbf{q}, \mathbf{v}_{i,j})$$

$$\text{Predicted Identity} = \arg\max_{i} S_i \quad \text{subject to} \quad \max_i S_i \ge \tau_{\text{rec}} = 0.35$$

### 3. Receiver Operating Characteristic & TAR @ FAR
True Accept Rate (TAR) at a specified False Accept Rate (FAR):

$$\text{TAR}(\tau) = \frac{\text{TP}(\tau)}{\text{TP}(\tau) + \text{FN}(\tau)}, \quad \text{FAR}(\tau) = \frac{\text{FP}(\tau)}{\text{FP}(\tau) + \text{TN}(\tau)}$$

$$\text{AUC} = \int_{0}^{1} \text{TAR}(\text{FAR}) \, d(\text{FAR})$$

---

## 📈 Benchmark Performance & Results

Evaluation conducted across **15,398 inference operations** comparing unoccluded baseline images against 5 masked test categories (1,138 genuine masked queries + 4,636 impostor queries).

### 1. Consolidated Summary Table

| Metric | Original (Unoccluded) | Masked (Occluded) | Performance Delta |
|---|:---:|:---:|:---:|
| **Total Query Benchmark Size** | 9,630 queries | 5,768 queries | — |
| **Rank-1 Identification Accuracy** | **99.83%** | **99.60%** | **−0.23%** |
| **Precision** | **99.68%** | **98.52%** | −1.16% |
| **Recall / True Positive Rate (TPR)** | **100.00%** | **99.47%** | −0.53% |
| **F1-Score** | **0.9984** | **0.9899** | −0.0085 |
| **False Positive Rate (FPR @ $\tau=0.35$)** | **0.35%** | **0.37%** | +0.02% |
| **ROC-AUC (Area Under Curve)** | **0.9990** | **0.9955** | **−0.0035** |
| **TAR @ FAR = 1% (0.01)** | **0.9988** | **0.9947** | −0.0041 |
| **Average Inference Latency (CPU)** | **351.05 ms** | **351.10 ms** | +0.05 ms |

### 2. Breakdown Across Mask Types

| Mask Type | Flag | Genuine Queries | True Positives (TP) | False Positives (FP) | Accuracy / TPR | Mean Cosine Score |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Surgical Mask** | `SUR` | 228 | 228 | 0 | **100.00%** | **0.8805** |
| **N95 Respirator** | `N95` | 228 | 227 | 0 | **99.56%** | 0.8544 |
| **KN95 Respirator** | `K95` | 228 | 227 | 0 | **99.56%** | **0.9016** |
| **Cloth Mask** | `CLT` | 227 | 225 | 1 | **99.56%** | **0.9032** |
| **Gas Mask** | `GAS` | 227 | 224 | 0 | **98.68%** | 0.8606 |
| **Total / Overall** | — | **1,138** | **1,131** | **1** | **99.47%** | **0.8801** |

---

## 📂 Repository Layout & File Responsibilities

```
d:/Masked/
├── faces_db/                            # Storage for identity embedding matrices (.npy)
├── face_database/                       # Raw copy of enrolled facial images
├── Real Images/                         # Enrolled positive gallery images (4,994 images)
├── New_Images/                          # Un-enrolled impostor test set (4,636 images)
├── Masked Images/                       # Occluded test queries (1,138 images across 5 mask types)
│
├── register_face.py                     # Single identity enrollment script (synthesizes 3x512 matrix)
├── batch_register_real_images.py        # Automated bulk enrollment of gallery datasets
├── live_recognition.py                  # Webcam feed recognition + Safe Online Learning
├── recognize_face.py                    # Static single-image query recognition module
├── video_face_recognition.py            # Video file (.mp4) stream processor & renderer
├── reset_embeddings.py                  # Resets memory-expanded matrices back to initial 3 views
├── test.py                              # Master benchmark harness & CSV logger
├── generate_research_graphs.py          # Publication-grade chart & ROC curve generator
│
├── comprehensive_evaluation_report.md   # Full experimental report with dataset metrics
├── online_learning_architecture.md      # Detailed documentation of dynamic memory mechanics
├── requirements.txt                     # Core Python dependencies
└── README.md                            # Main project documentation (this file)
```

---

## ⚙️ Installation & Setup Guide

### 1. Prerequisites
- **Operating System**: Windows 10/11, Ubuntu 20.04+, or macOS
- **Python**: Version `3.8`, `3.9`, `3.10`, or `3.11` (64-bit)
- **Webcam**: Standard USB or built-in camera (for live streaming)

### 2. Environment Setup

Clone the repository and create a clean Python virtual environment:

```bash
# Navigate to project directory
cd d:/Masked

# Create virtual environment
python -m venv face_env

# Activate virtual environment (Windows PowerShell)
.\face_env\Scripts\Activate.ps1

# Activate virtual environment (Linux/macOS)
source face_env/bin/activate
```

### 3. Install Dependencies

Install the locked dependencies from `requirements.txt`:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> [!NOTE]
> `requirements.txt` includes:
> - `insightface==0.7.3`
> - `onnxruntime==1.16.3`
> - `numpy==1.26.4`
> - `opencv-python`

---

## 🚀 Quickstart & Operational Workflows

### Workflow 1: Enrolling a New Person (Single Face)

Run `register_face.py` to register an individual from a single unoccluded image. The script automatically generates eye-occluded and lower-face masked variants and saves a `3 x 512` matrix to `faces_db/<person_name>.npy`.

```bash
python register_face.py
```
*Prompt Input:*
- **Person Name**: `John_Doe`
- **Path to Image**: `path/to/john_photo.jpg`

---

### Workflow 2: Bulk Enrollment of a Dataset Directory

To enroll an entire folder of identity images in batch mode:

```bash
python batch_register_real_images.py
```
*Reads images from `Real Images/` and populates `faces_db/` with baseline matrices.*

---

### Workflow 3: Real-Time Live Webcam Recognition & Online Learning

Launch the interactive webcam recognition system:

```bash
python live_recognition.py
```
- **Display**: Fullscreen window displaying bounding boxes and recognition labels.
- **Controls**: Press **`q`** to quit.
- **Behavior**: Green bounding box drawn for known individuals ($\ge 0.35$). Unknown faces are ignored without visual distraction. High-confidence matches ($\ge 0.55$) trigger safe in-memory embedding learning.

---

### Workflow 4: Processing Video Files (.mp4)

Process an offline video file and render output with bounding boxes:

```bash
python video_face_recognition.py
```
*Processes `input_video.mp4` and exports the recognized video stream to `output_recognized.mp4`.*

---

### Workflow 5: Running Benchmark Evaluation Suite

To run the large-scale benchmark across masked and unmasked query datasets:

```bash
python test.py
```
*Generates per-image evaluation results in `masked_recognition_per_image_results.csv` and summary metrics in `masked_recognition_metrics_by_mask_type.csv`.*

---

### Workflow 6: Generating Research Graphs

Produce publication-grade visualization graphics (ROC curves, TAR@FAR log plots, score distributions):

```bash
python generate_research_graphs.py
```
*Outputs PNG figures to the `graphs/` directory.*

---

### Workflow 7: Maintenance — Resetting Embedding Matrices

If live sessions have expanded memory matrices beyond the baseline views, reset all stored embeddings back to the original $3 \times 512$ matrix shape:

```bash
python reset_embeddings.py
```

---

## 🎛️ Hyperparameter & Threshold Configuration

Key system parameters can be adjusted directly in `live_recognition.py`, `video_face_recognition.py`, and `test.py`:

```python
# Decision Thresholds
RECOGNITION_THRESHOLD = 0.35  # Cosine similarity boundary for identity match
HIGH_CONF_THRESHOLD   = 0.55  # Boundary for triggering Safe Online Learning

# Memory Constraints
MAX_VIEW_BUFFER_SIZE  = 50    # Maximum number of stored face views per identity (FIFO)

# Detection Parameters
DETECTION_SIZE       = (640, 640) # Input resolution for InsightFace detector
```

---

## ❓ Frequently Asked Questions (FAQ)

<details>
<summary><b>Q: Does this system require GPU acceleration?</b></summary>
<br>
No. The system runs efficiently on CPU using <code>onnxruntime</code> CPU execution providers. Average inference latency is <b>~350 ms per frame</b> on standard multi-core x86 CPUs. GPU execution can be enabled by installing <code>onnxruntime-gpu</code> and setting <code>ctx_id=0</code> in <code>FaceAnalysis.prepare()</code>.
</details>

<details>
<summary><b>Q: How does the system prevent impostors from polluting known identity embeddings?</b></summary>
<br>
Online learning requires a similarity score $\ge 0.55$ (significantly higher than the recognition threshold of $0.35$). Impostor scores typically top out around $0.20 - 0.30$, making it mathematically nearly impossible for an unregistered person to trigger matrix expansion.
</details>

<details>
<summary><b>Q: Why are embeddings updated only in memory during live recognition?</b></summary>
<br>
Keeping updates in RAM ensures maximum execution speed ($O(1)$ matrix slicing) while guaranteeing strict zero disk pollution. Restarting the script cleanly reverts all identities back to their baseline 3-view configuration.
</details>

---

## 📄 License & Acknowledgments

- **Model Backbone**: Powered by [InsightFace (DeepInsight)](https://github.com/deepinsight/insightface) using the `buffalo_l` ArcFace pre-trained weights.
- **License**: Distributed under the [MIT License](LICENSE). Free for academic, research, and commercial applications.
