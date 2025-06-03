# Breast Cancer Classification Project

## Overview

This project provides an end-to-end deep learning pipeline for classifying breast cancer images as benign or malignant. It covers data preprocessing, model training, evaluation, and deployment using FastAPI. The modular structure makes it easy to adapt or extend for similar medical imaging tasks.

## Project Structure

```text
Breast-Cancer/
├── config/                # Configuration files
│   └── config.yaml
├── data/                  # Data directory
│   ├── raw/               # Raw images (benign/malignant)
│   └── processed/         # Numpy arrays for training/testing
├── models/
│   └── checkpoints/       # Saved model checkpoints
├── notebooks/             # Jupyter notebooks for EDA and experiments
│   ├── EDA.ipynb
│   ├── exploratory.ipynb
│   └── experiments.ipynb
├── src/                   # Source code
│   ├── api/               # FastAPI app and endpoints
│   ├── data/              # Data processing scripts
│   ├── models/            # Model definition
│   ├── utils/             # Utility functions
│   └── visualization/     # Visualization tools
├── setup.py                 # Package installation script
└── README.md              # Project documentation
```

## Main Components

- **Data Preparation:**
  - Raw images are stored in `data/raw/benign` and `data/raw/malignant`.
  - Preprocessing and augmentation are handled by `src/data/make_data.py`.
  - Processed datasets are saved as NumPy arrays in `data/processed/`.

- **Model:**
  - The model is a deep learning image classifier (see `src/models/model.py`).
  - Training, evaluation, and saving are demonstrated in `notebooks/experiments.ipynb`.

- **API Deployment:**
  - Inference is served via FastAPI (`src/api/`).
  - Endpoints accept image uploads and return predictions (benign/malignant).

- **Notebooks:**
  - `EDA.ipynb`: Exploratory data analysis and visualization.
  - `exploratory.ipynb`: Data processing and class distribution.
  - `experiments.ipynb`: Model training, evaluation, and saving.

## How to Run

1. **Install dependencies:**

   ```bash
   pip install -r ../../requirements.txt
   ```

2. **Prepare data:**

   - Place images in `data/raw/benign` and `data/raw/malignant`.
   - Use the data processing scripts or notebooks to generate processed datasets.

3. **Train the model:**

   - Run the training cells in `notebooks/experiments.ipynb` or use scripts in `src/models/`.

4. **Serve the model:**

   - Start the FastAPI server:

     ```bash
     python src/api/main.py
     ```

   - Access the API at `http://localhost:5000`.

## Configuration

All paths and model settings are managed in `config/config.yaml`.
