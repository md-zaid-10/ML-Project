# Student Maths Score Predictor: End-to-End ML Project

An end-to-end machine learning project that predicts a student's **maths score (0–100)** from their background and their reading and writing scores. It covers the full workflow: exploratory analysis, a modular training pipeline, model selection across seven regressors, and a Flask web app for live predictions.

**Best model:** Linear Regression · **R² = 0.880** · **Mean absolute error ≈ 4.2 points** (held-out test set)

---

## Features

- **Modular pipeline:** separate components for data ingestion, transformation and model training, with custom logging and exception handling.
- **Automated model selection:** trains 7 regression models, tunes each with `GridSearchCV`, and saves the one with the best test R².
- **Reusable preprocessing:** imputation, one-hot encoding and scaling live in a single scikit-learn `ColumnTransformer`, saved alongside the model so predictions use exactly the same transformations.
- **Web interface:** a responsive dark-mode dashboard (Flask + Tailwind CSS) with inline form validation, a loading state, an animated result gauge, and error handling.
- **Deployment-ready:** includes an AWS Elastic Beanstalk configuration.

## Dataset

[`notebook/data/stud.csv`](notebook/data/stud.csv): 1,000 student records with these columns:

| Feature | Type | Values |
|---|---|---|
| `gender` | Categorical | female, male |
| `race_ethnicity` | Categorical | group A – E |
| `parental_level_of_education` | Categorical | some high school … master's degree |
| `lunch` | Categorical | standard, free/reduced |
| `test_preparation_course` | Categorical | none, completed |
| `reading_score` | Numeric | 0–100 |
| `writing_score` | Numeric | 0–100 |
| **`math_score`** | **Target** | 0–100 |

## How it works

```
stud.csv ──► Data Ingestion ──► Data Transformation ──► Model Trainer ──► artifacts/model.pkl
              (80/20 split)      (impute, encode,        (7 models +         artifacts/preprocessor.pkl
                                  scale)                  GridSearchCV)                │
                                                                                       ▼
                                          Browser form ──► Flask app ──► Predict Pipeline ──► score
```

1. **Ingestion** ([data_ingestion.py](src/components/data_ingestion.py)) reads the dataset and writes `raw.csv`, `train.csv` and `test.csv` to `artifacts/` (80/20 split, `random_state=42`).
2. **Transformation** ([data_transformation.py](src/components/data_transformation.py)) builds the preprocessor:
   - Numeric columns: median imputation, then standard scaling.
   - Categorical columns: most-frequent imputation, one-hot encoding, then scaling.
3. **Training** ([model_trainer.py](src/components/model_trainer.py)) grid-searches each model, compares test R², and saves the best one. Training fails if no model reaches R² ≥ 0.6.
4. **Prediction** ([predict_pipeline.py](src/pipeline/predict_pipeline.py)) loads both pickles and transforms and predicts on new input from the web form.

### Models compared

Linear Regression · Decision Tree · Random Forest · Gradient Boosting · AdaBoost · XGBoost · CatBoost

## Project structure

```
ML Project/
├── app.py                    # Flask app for local use (port 5001)
├── application.py            # Same app, entry point for AWS Elastic Beanstalk
├── requirements.txt
├── setup.py                  # Installs src/ as a package
├── artifacts/                # Trained model, preprocessor and data splits
├── notebook/
│   ├── 1 . EDA STUDENT PERFORMANCE .ipynb
│   ├── 2. MODEL TRAINING.ipynb
│   └── data/stud.csv
├── src/
│   ├── components/           # data_ingestion, data_transformation, model_trainer
│   ├── pipeline/             # predict_pipeline
│   ├── exception.py          # Custom exception with file and line details
│   ├── logger.py             # Timestamped log files in logs/
│   └── utils.py              # save/load objects, model evaluation
├── templates/
│   ├── index.html            # Landing page
│   └── home.html             # Prediction dashboard
└── .ebextensions/            # Elastic Beanstalk config
```

---

## Run it on your own machine

### Prerequisites

- Python 3.10 or newer (developed and tested on **Python 3.11**)
- Git
- An internet connection when you open the web page (it loads Tailwind CSS and fonts from a CDN)

### 1. Clone the repository

```bash
git clone https://github.com/md-zaid-10/ML-Project.git
cd ML-Project
```

### 2. Create and activate a virtual environment

**macOS / Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (PowerShell)**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

<details>
<summary>Using conda instead?</summary>

```bash
conda create -p venv python=3.11 -y
conda activate ./venv
```
</details>

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

This installs pandas, scikit-learn, XGBoost, CatBoost, Flask and the other dependencies. The `-e .` line at the end of `requirements.txt` also installs the `src` package, so imports like `from src.utils import ...` work from anywhere in the project.

### 4. Start the web app

The trained model is already in `artifacts/`, so you can run the app straight away:

```bash
python app.py
```

Then open **http://127.0.0.1:5001** in your browser and click **Start a prediction**, or go straight to **http://127.0.0.1:5001/predictdata**.

Fill in the student's details and scores, then click **Predict maths score**.

> **macOS note:** `app.py` uses port **5001** because macOS's AirPlay Receiver often takes port 5000. `application.py` uses Flask's default port 5000 and is meant for deployment.

### 5. (Optional) Retrain the model

Run the full pipeline (ingestion, transformation, training) from the project root:

```bash
python -m src.components.data_ingestion
```

This takes about 30 seconds, prints the best model's test R², and **overwrites** `artifacts/model.pkl` and `artifacts/preprocessor.pkl`. Restart the Flask app to use the new model. A log of each run is written to `logs/`.

To experiment, edit the models or hyperparameter grids in [model_trainer.py](src/components/model_trainer.py).

### 6. (Optional) Explore the notebooks

```bash
pip install jupyter
jupyter notebook notebook/
```

- **EDA STUDENT PERFORMANCE**: data cleaning, distributions and feature relationships.
- **MODEL TRAINING**: the original model experiments that the pipeline is built from.

---

## Deploying to AWS Elastic Beanstalk

`application.py` exposes a Flask object named `application`, which is what Elastic Beanstalk looks for. `.ebextensions/python.config` points the WSGI server at it:

```yaml
option_settings:
  "aws:elasticbeanstalk:container:python":
    WSGIPath: application:application
```

Create a Python platform environment on Elastic Beanstalk and deploy the repository, for example with the EB CLI (`eb init`, `eb create`) or through AWS CodePipeline connected to GitHub.

## Tech stack

**ML:** pandas · NumPy · scikit-learn · XGBoost · CatBoost · dill
**Web:** Flask · Jinja2 · Tailwind CSS · vanilla JavaScript
**Analysis:** Jupyter · Matplotlib · Seaborn
**Deployment:** AWS Elastic Beanstalk

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | Run commands from the project root, and make sure `pip install -r requirements.txt` completed (it installs `src` via `-e .`). |
| `Address already in use` / 403 on port 5000 | Use `python app.py` (port 5001), or turn off AirPlay Receiver in macOS System Settings. |
| Error loading `model.pkl` | The pickle was created with a different scikit-learn version. Retrain with step 5 to regenerate it for your environment. |
| Page looks unstyled | The page loads Tailwind CSS and fonts from a CDN, so check your internet connection. |

## Author

**Md Zaid** · [GitHub @md-zaid-10](https://github.com/md-zaid-10)
