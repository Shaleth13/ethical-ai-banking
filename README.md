# Ethical AI in Banking - Building Trust and Transparency

Simple demo of a credit approval system with three views:

- customer view with explanation
- fairness dashboard
- audit log for decisions

## Tech stack

- Python 3
- Streamlit for the UI
- scikit-learn for the model
- pandas / NumPy for data
- joblib for saving the model

## Folder layout

- `train_model.py` – builds a synthetic dataset, trains a model and saves it
- `app.py` – Streamlit app
- `utils_explainability.py` – helper for feature impact
- `data/` – synthetic dataset (created by `train_model.py`)
- `models/` – saved model
- `logs/` – decision logs from the app

## Setup

Install requirements:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python train_model.py
```

Run the app:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal.
