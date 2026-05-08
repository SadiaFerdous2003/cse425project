# Unsupervised Neural Network for Multi-Genre Music Generation

## Overview

This project implements a multi-stage pipeline for unsupervised neural music generation across multiple genres using deep learning. The system leverages LSTM autoencoders, genre-conditioned transformers, and reinforcement learning with human feedback (RLHF) to generate and refine symbolic music (MIDI) in genres such as classical, jazz, rock, pop, and electronic.

The pipeline is modular, allowing you to train, evaluate, and generate music at each stage. The project is designed for research and experimentation in symbolic music generation, genre conditioning, and human-in-the-loop reinforcement learning.

## Project Structure

- **2009/**: Raw MIDI files (Groove MIDI dataset, drum and piano performances)
- **Lakh MIDI Dataset/**: Additional MIDI data for training and evaluation (multi-genre)
- **Task-1/**: LSTM Autoencoder for music reconstruction (see `task1.py`)
- **Task-2/**: Genre-conditioned LSTM autoencoder and t-SNE/PCA analysis (see `task2_pipeline.py`)
- **Task3/**: Transformer-based music generation pipeline (see `task3_pipeline.py`)
- **task_4/**: RLHF (Reinforcement Learning from Human Feedback) for music refinement (see `task_4.py`)
- **task3_output/**: Output samples from Task 3
- **human_survey_results.csv**: Human evaluation results for RLHF
- **requirements.txt**: Python dependencies

## Pipeline Summary

### Task 1: LSTM Autoencoder for Music Reconstruction

**Goal:** Learn to reconstruct short music segments from MIDI piano-rolls using an LSTM autoencoder.

- **Data:** Loads and preprocesses MIDI files into fixed-length piano-roll windows (128 frames, 8 seconds).
- **Model:** LSTM encoder → latent vector → LSTM decoder.
- **Training:** Focal loss, Adam optimizer, validation monitoring.
- **Evaluation:** Rhythm diversity, repetition ratio, pitch similarity, and comparison to random/Markov baselines.
- **Visualization:** Loss curves, piano-roll plots, latent space PCA.

### Task 2: Genre-Conditioned LSTM Autoencoder

**Goal:** Learn genre-aware latent representations and analyze genre separation.

- **Data:** Maps MIDI files to 5 genres (classical, jazz, rock, pop, electronic).
- **Model:** LSTM autoencoder with genre embedding.
- **Training:** Similar to Task 1, with genre conditioning.
- **Evaluation:** t-SNE/PCA visualization of latent space, genre separation, reconstruction quality.

### Task 3: Transformer-based Music Generation

**Goal:** Generate new music sequences using a transformer model with genre conditioning.

- **Data:** Tokenizes MIDI using REMI (miditok library).
- **Model:** Transformer encoder-decoder (multi-head attention, 4 layers, genre embedding).
- **Training:** Sequence modeling, teacher forcing, early stopping.
- **Evaluation:** Generates new samples, supports genre control, saves outputs to `task3_output/`.

### Task 4: RLHF (Reinforcement Learning from Human Feedback)

**Goal:** Refine music generation using human feedback and reinforcement learning.

- **Reward Model:** Trained on human survey results (`human_survey_results.csv`) using a BiGRU.
- **RL Algorithm:** REINFORCE to update transformer policy.
- **Evaluation:** Human and automatic metrics, improvement over baseline transformer.

## Installation

1. Clone the repository and navigate to the project folder:
   ```bash
   git clone https://github.com/thahamidnabil2002-droid/-Unsupervised-Neural-Networks-for-Multi-Genre-Music-Generation.git
   cd -Unsupervised-Neural-Networks-for-Multi-Genre-Music-Generation
   ```
2. Install dependencies (Python 3.8+ recommended):
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Each task is modular and can be run independently. Example usage:

- **Task 1:**
  ```bash
  python Task-1/task1.py
  ```
- **Task 2:**
  ```bash
  python Task-2/task2_pipeline.py
  ```
- **Task 3:**
  ```bash
  python Task3/task3_pipeline.py
  ```
- **Task 4:**
  ```bash
  python task_4/task_4.py
  ```

**Tips:**

- Place your MIDI files in the `2009/` and `Lakh MIDI Dataset/` folders before running.
- Each script creates its own output and intermediate folders as needed.
- For best results, run tasks in order (1 → 2 → 3 → 4), but you can experiment with any stage.

## Data

- Place your MIDI files in the `2009/` and `Lakh MIDI Dataset/` folders. The more diverse the data, the better the genre modeling.
- Preprocessing and output folders are created automatically by each script.
- Human survey results for RLHF are in `human_survey_results.csv` (CSV: participant, model, sample, score).

## Requirements

- Python 3.8+
- See `requirements.txt` for all dependencies (PyTorch, miditok, pretty_midi, numpy, matplotlib, tensorboard)

## Results & Evaluation

- Generated samples and evaluation metrics are saved in the respective output folders for each task.
- Human survey results for RLHF are in `human_survey_results.csv`.
- Evaluation metrics include:
  - **Reconstruction Loss:** Measures how well the model reconstructs input music.
  - **Rhythm Diversity:** Diversity of rhythmic patterns in generated music.
  - **Repetition Ratio:** Amount of repetition in generated sequences.
  - **Pitch Similarity:** Similarity of pitch distributions to real data.
  - **Genre Separation:** Visualized with t-SNE/PCA in Task 2.
  - **Human Ratings:** Collected for RLHF (Task 4) to guide reward model.

## Datasets

- **Groove MIDI Dataset:** Drum and piano performances for training and evaluation.
- **Lakh MIDI Dataset:** Large-scale multi-genre MIDI collection.
- **Human Survey Results:** CSV file with human ratings for RLHF.

## Acknowledgements

- Groove MIDI Dataset
- Lakh MIDI Dataset
- [miditok](https://github.com/Natooz/miditok)
- [pretty_midi](https://github.com/craffel/pretty-midi)

---

## Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change or add.

## License

This project is for academic and research use. Please cite the datasets and libraries used if you publish work based on this code.

---

For more details, see comments and documentation in each task's script.
