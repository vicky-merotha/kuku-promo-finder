# EE655 Course Project: Cricket Shot Detection and Video Similarity Analysis

## Project Summary

This repository contains the final submission for the `EE655` course project on cricket shot recognition. The system analyzes a cricket batting video, predicts the shot class, and can also compare two videos to estimate how similar their batting mechanics are.

The project combines:

- a Streamlit-based interactive frontend
- a PyTorch inference pipeline
- multiple trained checkpoint variants
- experiment summaries from architecture, sampling, and voting studies
- downloadable PDF and JSON reports

The repository is kept focused on the files needed to understand the project, run the Streamlit app locally, and review the final notebook-backed results.

## Objectives

- classify a cricket batting shot from an uploaded video
- compare two videos and estimate shot-mechanics similarity
- evaluate different model architectures
- evaluate different frame-sampling strategies
- evaluate different prediction-voting strategies
- present the final pipeline through a usable interactive interface

## Supported Shot Classes

- `cover`
- `defense`
- `flick`
- `hook`
- `late_cut`
- `lofted`
- `pull`
- `square_cut`
- `straight`
- `sweep`

## Main Features

- single-video shot classification
- two-video similarity comparison
- browser-camera capture for deployed use, plus live detection from a locally attached webcam or a video file
- checkpoint selection from trained experiment outputs
- multiple sampling strategies: `uniform`, `motion`, `hybrid`
- multiple voting strategies: `single`, `majority`, `weighted`
- probability breakdown for all shot classes
- key-frame preview and timeline summary
- experiment tables and figure viewer inside the app
- PDF report download
- JSON report download

## Method Overview

The final system uses an EfficientNet-B0-based visual encoder and supports three classifier heads:

- `CNN Only`
- `GRU`
- `LSTM`

The workflow is:

1. read the uploaded video
2. extract and resize frames
3. sample a fixed number of frames
4. convert frames into tensors
5. run inference using a selected checkpoint
6. aggregate predictions using the chosen voting strategy
7. show predictions, confidence values, figures, and downloadable reports

## Repository Structure

| Path | Purpose |
| --- | --- |
| `app.py` | Main Streamlit application. Handles UI, model controls, uploads, prediction display, experiment summaries, and report downloads. |
| `cricket_notebook_model.py` | Core inference module. Handles frame extraction, sampling, model definitions, checkpoint loading, prediction logic, similarity computation, and PDF generation. |
| `ee655_final_project_notebook.ipynb` | Final notebook reference used for training, experiments, and result generation. |
| `.streamlit/config.toml` | Streamlit configuration for upload size and theme setup. |
| `requirements.txt` | Python package dependencies needed to run the project. |
| `.gitignore` | Ignores virtual environments, caches, logs, and generated local files. |
| `.gitattributes` | Marks text and binary file types for cleaner repository handling. |
| `LICENSE` | Project license file. |
| `results_current/checkpoints/` | Trained model checkpoints used by the app. |
| `results_current/results/` | Experiment CSV summaries and visualization figures used by the frontend. |

## Code File Explanation

### `app.py`

This is the user-facing application layer. Its responsibilities include:

- loading project assets and available checkpoints
- detecting the best default runtime preset from experiment summaries
- rendering the Streamlit layout and sidebar controls
- accepting uploaded cricket videos
- triggering single-video analysis or two-video comparison
- displaying prediction confidence, key frames, summary cards, and experiment tables
- exporting reports in PDF and JSON formats

Key parts of the file:

- result preparation and checkpoint discovery
- cached loading of model bundles and experiment tables
- UI rendering for analysis, comparison, and report sections
- experiment-summary tabs for architecture, sampling, and voting studies

### `cricket_notebook_model.py`

This file contains the ML and reporting pipeline used by the app. Its responsibilities include:

- frame extraction using OpenCV
- frame sampling using `uniform`, `motion`, and `hybrid` strategies
- model definitions for `CNNOnly`, `CricketGRU`, and `CricketLSTM`
- checkpoint loading with PyTorch
- inference and probability generation
- comparison using feature-vector cosine similarity
- PDF report creation

Important functions and components:

- `find_checkpoint()` locates a usable trained checkpoint
- `sample_frames()` selects frames according to the configured strategy
- `load_bundle()` loads model weights and preprocessing logic
- `analyze_video()` produces the prediction output used by the UI
- `compare_analyses()` compares two analyzed videos
- `create_pdf_report()` generates downloadable reports

## Renamed Artifacts

The result assets were renamed to describe what they actually represent.

### Checkpoints

| Old name | New name |
| --- | --- |
| `p1_cnn_only.pth` | `phase1_cnn_only_uniform_checkpoint.pth` |
| `p1_gru.pth` | `phase1_gru_uniform_checkpoint.pth` |
| `p1_lstm.pth` | `phase1_lstm_uniform_checkpoint.pth` |
| `p2_gru_hybrid.pth` | `phase2_gru_hybrid_checkpoint.pth` |
| `p2_gru_motion.pth` | `phase2_gru_motion_checkpoint.pth` |

### CSV Summaries

| Old name | New name |
| --- | --- |
| `phase1.csv` | `phase1_architecture_comparison.csv` |
| `phase2_corrected.csv` | `phase2_sampling_strategy_comparison.csv` |
| `phase3.csv` | `phase3_voting_strategy_evaluation.csv` |

### Figures

Figure files were renamed from short internal names to readable names such as:

- `confusion_matrix_phase1_cnn_only_uniform_checkpoint.png`
- `training_curves_phase1_gru_uniform_checkpoint.png`
- `confusion_matrix_phase2_gru_motion_checkpoint.png`

## Experiment Assets Included in the Repository

### Checkpoint Files

- `results_current/checkpoints/phase1_cnn_only_uniform_checkpoint.pth`
- `results_current/checkpoints/phase1_gru_uniform_checkpoint.pth`
- `results_current/checkpoints/phase1_lstm_uniform_checkpoint.pth`
- `results_current/checkpoints/phase2_gru_hybrid_checkpoint.pth`
- `results_current/checkpoints/phase2_gru_motion_checkpoint.pth`

### Summary CSV Files

- `results_current/results/phase1_architecture_comparison.csv`
- `results_current/results/phase2_sampling_strategy_comparison.csv`
- `results_current/results/phase3_voting_strategy_evaluation.csv`

### Visualization Files

- training curves for each retained checkpoint
- confusion matrices for each retained checkpoint

These assets are intentionally kept in the repository because the app depends on them for inference, experiment display, and course-project demonstration.

## How to Run the Project

### 1. Create and activate a virtual environment

The project was verified locally with Python `3.10`.

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### 2. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Start the Streamlit app

```powershell
python -m streamlit run app.py
```

### 4. Open the local interface

Streamlit will print a local URL in the terminal, usually:

```text
http://localhost:8501
```

### Webcam note

For deployed apps, use the browser camera option in the live tab. The capture happens in the browser and is uploaded to Streamlit, so it works on hosted deployments too.

The local webcam mode still uses OpenCV on the same machine that runs Streamlit. If you deploy the app to a hosted Linux server, Docker container, WSL session, or another remote environment, that server cannot access your laptop camera. In those cases, use the browser camera option, `Video file` mode, or run the app locally.

## How to Use the App

- choose the architecture, sampling strategy, and voting strategy from the sidebar
- upload one video to classify a shot
- upload two videos to compare predicted results and similarity
- review confidence values, key frames, and experiment figures
- download a PDF or JSON report from the results panel

## Dependencies

Main packages used in this project:

- `streamlit`
- `torch`
- `torchvision`
- `opencv-python-headless`
- `numpy`
- `pandas`
- `Pillow`
- `reportlab`

## Expected Outputs

For a single-video analysis, the app provides:

- predicted shot label
- confidence score
- class probability breakdown
- timeline summary
- key-frame previews
- report downloads

For a two-video comparison, the app provides:

- prediction for video A
- prediction for video B
- similarity score
- summary interpretation of mechanical similarity

## Limitations

- predictions depend on the quality and diversity of the training set
- the app expects cricket batting videos with usable visual clarity
- similarity is based on extracted model features, not a biomechanical ground-truth system
- included checkpoints represent retained experiments, not every intermediate training run

## Final Note

This repository is organized as a final, presentation-ready course-project submission. Only the final reference notebook and the runtime assets required by the app are kept in version control, so teammates and evaluators can clone the repository and run it locally without local logs, cache files, or generated outputs getting in the way.
