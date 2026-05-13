# Unsupervised Neural Networks for Multi-Genre Music Generation

## Overview

## Methodology

This project follows a four-stage pipeline for unsupervised multi-genre music generation and refinement. Each stage builds on the previous one, combining deep learning, genre conditioning, and human feedback. The methodology is as follows:

### 1. Data Preparation

- Collect MIDI files from Groove MIDI and Lakh MIDI datasets.
- Preprocess MIDI files into piano-roll or tokenized formats (e.g., REMI for transformers).
- Assign genre labels (classical, jazz, rock, pop, electronic) for genre-conditioned models.

### 2. LSTM Autoencoder (Task 1)

- Convert MIDI to fixed-length, binarized piano-rolls.
- Train an LSTM autoencoder to reconstruct input sequences (encoder → latent vector → decoder).
- Evaluate using reconstruction loss, rhythm diversity, repetition ratio, and pitch similarity.
- Compare with random and Markov chain baselines.

### 3. Genre-Conditioned VAE (Task 2)

- Map MIDI files to genre labels.
- Train a VAE with genre conditioning to learn a structured latent space.
- Visualize latent space using t-SNE/PCA to check genre separation.
- Generate new samples by interpolating in the latent space and traversing genres.

### 4. Transformer-Based Generation (Task 3)

- Tokenize MIDI files using REMI (miditok library).
- Train a transformer model for symbolic music sequence generation, with optional genre conditioning.
- Generate new music samples by sampling from the trained model.

### 5. Reinforcement Learning from Human Feedback (RLHF, Task 4)

- Collect human ratings for generated samples (stored in `human_survey_results.csv`).
- Train a reward model to predict human preferences.
- Refine the transformer model using policy gradient (REINFORCE), guided by the reward model.
- Evaluate improvements using both automatic metrics and human scores.

### 6. Evaluation & Visualization

- Save generated samples (MIDI), evaluation metrics, and visualizations (loss curves, latent space, RL learning curves) in output folders.
- Summarize results in tables and plots for easy comparison.

This methodology enables unsupervised learning, genre transfer, and human-in-the-loop refinement for symbolic music generation.

This project implements a multi-stage pipeline for unsupervised neural music generation across multiple genres using deep learning. The system leverages LSTM autoencoders, genre-conditioned VAEs, transformers, and reinforcement learning with human feedback (RLHF) to generate and refine symbolic music (MIDI) in genres such as classical, jazz, rock, pop, and electronic. The pipeline is designed for research and experimentation in symbolic music generation, genre transfer, and human-in-the-loop evaluation.

## Project Structure

- **2009/**: Raw MIDI files (Groove MIDI dataset, percussion and multi-instrument)
- **Lakh MIDI Dataset/**: Additional MIDI data for training and evaluation (multi-genre)
- **Task-1/**: LSTM Autoencoder for music reconstruction (see `task1.py`)
- **Task-2/**: Genre-conditioned VAE and latent space analysis (see `task2_pipeline.py`)
- **Task3/**: Transformer-based music generation pipeline (see `task3_pipeline.py`)
- **task_4/**: RLHF (Reinforcement Learning from Human Feedback) for music refinement (see `task_4.py`)
- **task3_output/**: Output samples from Task 3 (MIDI files)
- **human_survey_results.csv**: Human evaluation results for RLHF (CSV format)
- **requirements.txt**: Python dependencies

## Pipeline Summary

### Task 1: LSTM Autoencoder for Music Reconstruction

- Loads and preprocesses MIDI files into piano-rolls (fixed window, binarized)
- Trains an LSTM autoencoder (encoder → latent vector → decoder)
- Evaluates with rhythm diversity, repetition ratio, pitch similarity, and reconstruction loss
- Baseline comparison: Random generator and Markov chain
- Visualizes loss curves, piano-rolls, and latent space (PCA)

### Task 2: Genre-Conditioned Variational Autoencoder (VAE)

- Maps MIDI files to 5 genres: classical, jazz, rock, pop, electronic
- Trains a genre-conditioned VAE for multi-genre music generation
- Visualizes latent space with t-SNE/PCA for genre separation
- Compares genre separation and reconstruction quality
- Outputs genre-interpolated samples and latent traversals

### Task 3: Transformer-based Music Generation

- Tokenizes MIDI using REMI (miditok)
- Trains a transformer model for symbolic sequence generation
- Supports genre conditioning via embeddings
- Generates new music samples (MIDI)
- Outputs samples to `task3_output/`

### Task 4: RLHF (Reinforcement Learning from Human Feedback)

- Trains a reward model from human survey data (see `human_survey_results.csv`)
- Refines the transformer policy using REINFORCE (policy gradient)
- Evaluates improvements with both human and automatic metrics
- Visualizes reward model training and RL learning curves

## Installation

1. Clone the repository and navigate to the project folder.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Each task has its own pipeline script:

- **Task 1:** `Task-1/task1.py` — LSTM autoencoder training and evaluation
- **Task 2:** `Task-2/task2_pipeline.py` — VAE training, genre analysis, and visualization
- **Task 3:** `Task3/task3_pipeline.py` — Transformer training and sample generation
- **Task 4:** `task_4/task_4.py` — RLHF reward model and policy refinement

Run each script with Python:

```bash
python Task-1/task1.py
python Task-2/task2_pipeline.py
python Task3/task3_pipeline.py
python task_4/task_4.py
```

Intermediate and output files (e.g., preprocessed data, models, generated samples) are saved in the respective `outputs/`, `generated/`, or `task3_output/` folders within each task directory.

## Data

- Place your MIDI files in the `2009/` (Groove MIDI) and `Lakh MIDI Dataset/` folders.
- Preprocessing and output folders are created automatically by each script.
- Human survey results for RLHF are stored in `human_survey_results.csv` (CSV format: participant, model, sample, score).

## Requirements

- Python 3.8+
- See `requirements.txt` for all dependencies:
  - torch
  - miditok >= 3.0.0
  - pretty_midi
  - numpy
  - matplotlib
  - tensorboard

## Results & Evaluation

## Results & Visualizations

This section summarizes the main results, metrics, and outputs for each stage of the pipeline. All results, sample outputs, and evaluation plots are saved in the respective output folders for each task.

### Summary Table

| Task | Model               | Output Location | Key Metrics / Results                                                     | Human Score (1–5) |
| ---- | ------------------- | --------------- | ------------------------------------------------------------------------- | ----------------- |
| 1    | LSTM Autoencoder    | Task-1/outputs/ | Reconstruction loss, rhythm diversity, repetition ratio, pitch similarity | 1.1–1.6           |
| 2    | Genre VAE           | Task-2/outputs/ | Genre separation (t-SNE/PCA), reconstruction loss, genre interpolation    | 2.6–3.4           |
| 3    | Transformer         | task3_output/   | Sequence quality, genre conditioning, sample diversity                    | 4.1–4.8           |
| 4    | RLHF (Reward Model) | task_4/outputs/ | Reward model accuracy, RL learning curves, human preference improvement   | See CSV           |

### Example Outputs

- **Generated MIDI files:**
  - Task 1: Task-1/outputs/
  - Task 2: Task-2/outputs/
  - Task 3: task3_output/
  - Task 4: task_4/outputs/

- **Evaluation Plots:**
  - Loss curves, genre separation (t-SNE/PCA), and RL learning curves are saved as PNG files in each task's output folder.

- **Human Survey Results:**
  - All human evaluation scores are in `human_survey_results.csv` (CSV format: participant, model, sample, score).
  - Example: Task 1 (LSTM AE): 1.1–1.6, Task 2 (VAE): 2.6–3.4, Task 3 (Transformer): 4.1–4.8

### How to View Results

1. **Generated Music:** Open any MIDI file from the output folders in a MIDI player or DAW to listen to the generated samples.
2. **Plots & Visualizations:** Open PNG files in the output folders to view loss curves, genre separation, and RL learning progress.
3. **Human Evaluation:** Open `human_survey_results.csv` in Excel or a text editor to see detailed human ratings for each model and sample.

### Example Visualizations

Below are typical visualizations you will find in the output folders:

- **Loss Curves:** Show training and validation loss over epochs for each model.
- **Latent Space (t-SNE/PCA):** Visualize genre separation and latent traversals.
- **RL Learning Curves:** Show reward model accuracy and RL policy improvement over time.

---

For a quick overview, see the summary table above. For detailed results, explore the output folders and survey CSV. This structure ensures anyone can easily understand and evaluate the project outcomes.

## Acknowledgements

- Groove MIDI Dataset
- Lakh MIDI Dataset
- [miditok](https://github.com/Natooz/miditok)
- [pretty_midi](https://github.com/craffel/pretty-midi)

---

For questions or contributions, please open an issue or pull request.
