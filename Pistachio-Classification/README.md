# Pistachio Classification Project

## Overview

This project provides an end-to-end deep learning pipeline for classifying pistachio images into different categories. It covers data preprocessing, model training, evaluation, and deployment using FastAPI. The modular structure makes it easy to adapt or extend for similar image classification tasks.

## Project Structure

```text
Pistachio-Classification/
├── config/                  # Project configuration files
├── data/                    # Stores raw and processed image data
│   ├── raw/                 # Original, unprocessed images
│   └── processed/           # Prepared data (e.g., NumPy arrays) for model training
├── models/                  # Trained model checkpoints
├── notebooks/               # Jupyter notebooks for exploration and development
│   ├── EDA.ipynb            # Exploratory Data Analysis
│   ├── exploratory.ipynb    # General exploratory work
│   └── experiments.ipynb    # Model experimentation and testing
├── src/                     # Core source code for the project
│   ├── api/                 # FastAPI application and API endpoints
│   ├── constants/           # Global project constants
│   ├── data/                # Data loading and processing scripts
│   ├── model/               # Neural network model definitions
│   ├── processing/          # Data augmentation and preprocessing pipelines
│   ├── utils/               # General utility functions
│   └── visualization/       # Tools for data and model visualization
├── setup.py                 # Package installation script
└── README.md                # Project overview and documentation
```

## Main Components

- **Data Preparation:**
  - Raw images are stored in `data/raw/Pistachio_Image_Dataset`.
  - Preprocessing and augmentation are handled by scripts in `src/processing/`.
  - Processed datasets are saved as NumPy arrays in `data/processed/`.

- **Model:**
  - The model is a deep learning image classifier (see `src/model/model.py`).
  - Training, evaluation, and saving are demonstrated in `notebooks/experiments.ipynb`.

- **API Deployment:**
  - Inference is served via FastAPI (`src/api/`).
  - Endpoints accept image uploads and return predictions.

- **Notebooks:**
  - `EDA.ipynb`: Exploratory data analysis and visualization.
  - `exploratory.ipynb`: Data processing and class distribution.
  - `experiments.ipynb`: Model training, evaluation, and saving.

## How to Run

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare data:**

   - Place images in `data/raw/Pistachio_Image_Dataset`.
   - Use the data processing scripts or notebooks to generate processed datasets.

3. **Train the model:**

   - Run the training cells in `notebooks/experiments.ipynb` or use scripts in `src/model/`.

4. **Serve the model:**

   - Start the FastAPI server:

     ```bash
     python src/api/main.py
     ```

   - Access the API at `http://localhost:5000`.

## Configuration

All paths and model settings are managed in `config/config.yaml`.
