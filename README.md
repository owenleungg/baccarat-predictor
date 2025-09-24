# Baccarat Predictor

A machine learning system for predicting Baccarat game outcomes using 2D CNN.

## File Structure

```
baccarat-predictor/
├── data/
│   ├── raw/              # Original JSON files from games
│   └── processed/        # CSV outcomes files (extracted from JSON)
├── models/               # Trained models and results
│   ├── final_model.pth
│   ├── training_history.png
│   └── evaluation_results.png
├── processed_data/       # Preprocessed training data
│   ├── train_grids.npy
│   ├── train_labels.npy
│   └── label_encoder.pkl
├── src/
│   ├── core/             # Core ML components
│   │   ├── baccarat_model.py
│   │   ├── baccarat_training.py
│   │   ├── baccarat_inference.py
│   │   └── baccarat_data_prep.py
│   ├── apps/             # Streamlit applications
│   │   ├── app.py
│   │   ├── app_manual.py
│   │   └── app_vision.py
│   └── utils/            # Utility scripts
│       ├── extract_baccarat_outcomes.py
│       └── backup_model.py
├── extract_outcomes.py   # Entry point: Extract CSV from JSON
├── prepare_data.py       # Entry point: Prepare training data
├── train.py              # Entry point: Train model
└── README.md
```

## Usage

### 1. Extract Outcomes from JSON Files

```bash
python extract_outcomes.py
```

- Reads JSON files from `data/raw/`
- Outputs CSV files to `data/processed/`

### 2. Prepare Training Data

```bash
python prepare_data.py
```

- Reads CSV files from `data/processed/`
- Creates training datasets in `processed_data/`

### 3. Train Model

```bash
python train.py
```

- Uses data from `processed_data/`
- Saves trained model to `models/`

### 4. Run Streamlit Apps

```bash
# Manual input app
streamlit run src/apps/app_manual.py

# Vision-based app
streamlit run src/apps/app_vision.py

# Combined app
streamlit run src/apps/app.py
```

## Data Flow

```
data/raw/ (JSON files)
    ↓ extract_outcomes.py
data/processed/ (CSV files)
    ↓ prepare_data.py
processed_data/ (NPY files)
    ↓ train.py
models/ (Trained model)
```

## Requirements

- Python 3.8+
- PyTorch
- Streamlit
- OpenCV
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
