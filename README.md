# Student Performance Prediction and Flask App

An end-to-end machine learning project that predicts a student's **math score** from demographic details and reading and writing scores. It compares several regression models, packages the best one as a reusable pipeline, and serves predictions through a small Flask web app.

**Best result:** Linear Regression reaches an **R² of 0.88** (RMSE 5.14, MAE 4.03 marks) on a held-out test set of 200 students.

## What's in this project

- **Exploratory data analysis** of exam scores and student background (`notebook/data/EDA.ipynb`).
- **Model comparison** of nine regression algorithms (`notebook/data/MODEL TRAINING.ipynb`).
- **A modular training pipeline** with data ingestion, transformation and model selection (`src/mlproject/`).
- **A Flask web app** with a form that returns a predicted math score.

## Dataset

999 students and 8 columns, with no missing values and no duplicate rows. The file is `notebook/data/raw.csv`.


| Column | Type | Values |
|---|---|---|
| `gender` | categorical | female, male |
| `race_ethnicity` | categorical | group A to group E |
| `parental_level_of_education` | categorical | 6 levels, from some high school to master's degree |
| `lunch` | categorical | standard, free/reduced |
| `test_preparation_course` | categorical | none, completed |
| `reading_score` | numeric | 17 to 100 |
| `writing_score` | numeric | 10 to 100 |
| `math_score` | numeric, **target** | 0 to 100 (mean 66.1) |

## What the data shows

These are averages from the dataset. They describe patterns and do not prove cause and effect.

| Group | Average score | Compared with |
|---|---|---|
| Standard lunch | 70.8 | 62.1 for free/reduced lunch |
| Completed the test preparation course | 72.7 | 65.0 for no course |
| Parents with a master's degree | 73.6 | 63.1 for parents with high school education |
| Group E | 72.8 (highest) | 63.0 for group A (lowest) |

- Female students score higher in reading (72.6 vs 65.5) and writing (72.4 vs 63.3). Male students score higher in math (68.7 vs 63.6).
- Reading and writing scores are very closely related (correlation 0.95), and both are strongly related to math (0.82 and 0.80). This is why they are the strongest inputs for predicting math.

## Models and results

All models were trained on 799 students and tested on 200 (80/20 split, `random_state=42`), after one-hot encoding the categorical columns and scaling the numeric ones (19 input features). The table shows test-set results with default settings, as run in the notebook.

| Model | R² | RMSE | MAE |
|---|---|---|---|
| Ridge | 0.8809 | 5.13 | 4.03 |
| **Linear Regression** | **0.8807** | **5.14** | **4.03** |
| CatBoost | 0.8452 | 5.85 | 4.54 |
| Random Forest | 0.8348 | 6.05 | 4.76 |
| AdaBoost | 0.8215 | 6.28 | 4.91 |
| K-Neighbors | 0.8154 | 6.39 | 4.98 |
| Lasso | 0.8115 | 6.46 | 5.13 |
| XGBoost | 0.7964 | 6.71 | 5.32 |
| Decision Tree | 0.6773 | 8.45 | 6.67 |

Simple linear models did best, which fits the data: math scores rise almost linearly with reading and writing scores. The model saved in `artifacts/model.pkl` is a Linear Regression.

## How the training pipeline works

Run `app.py` to execute the steps in order:

1. **Data ingestion** (`data_ingestion.py`): reads `notebook/data/raw.csv`, saves a copy and splits it into `artifacts/train.csv` and `artifacts/test.csv`.
2. **Data transformation** (`data_transformation.py`): builds a scikit-learn `ColumnTransformer`.
   - Numeric columns: median imputation, then standard scaling.
   - Categorical columns: most-frequent imputation, one-hot encoding, then scaling.
   - The fitted transformer is saved as `artifacts/preprocessor.pkl`.
3. **Model training** (`model_trainer.py`): tunes seven models (Random Forest, Decision Tree, Gradient Boosting, Linear Regression, XGBoost, CatBoost, AdaBoost) with `GridSearchCV` (3-fold), compares them by R² on the test set, and saves the best one as `artifacts/model.pkl`. It refuses to save a model with R² below 0.6.

## Web app

`application.py` is a Flask app with two routes:

- `/` is a simple welcome page.
- `/predictdata` shows a form for gender, ethnicity group, parental education, lunch type, test preparation course, reading score and writing score. On submit, the saved preprocessor and model return the predicted math score.

## Project structure

```
.
├── application.py              # Flask app (entry point for the web app)
├── app.py                      # Runs the training pipeline
├── requirements.txt
├── setup.py
├── notebook/data/
│   ├── raw.csv                 # Dataset
│   ├── EDA.ipynb               # Exploratory analysis
│   └── MODEL TRAINING.ipynb    # Model comparison
├── src/mlproject/
│   ├── components/             # data_ingestion, data_transformation, model_trainer
│   ├── pipelines/              # prediction_pipeline (used by the web app)
│   ├── utils.py                # save/load objects, model evaluation
│   ├── logger.py
│   └── exception.py
├── templates/                  # index.html, home.html
├── artifacts/                  # train/test data, preprocessor.pkl, model.pkl
└── .ebextensions/              # AWS Elastic Beanstalk configuration
```

## Run it locally

Requires Python 3 and pip.

```bash
git clone https://github.com/parmesh1249/Student-Performance-Prediction-and-Flask-project.git
cd Student-Performance-Prediction-and-Flask-project

python -m venv venv
venv\Scripts\activate          # on macOS or Linux: source venv/bin/activate
pip install -r requirements.txt

python app.py                  # trains the models and writes the files in artifacts/
python application.py          # starts the web app
```

Then open `http://127.0.0.1:5000/predictdata` in a browser. Run the commands from the project's root folder. Training tunes several models, so it can take a while.

## Deployment

The repository includes an AWS Elastic Beanstalk configuration (`.ebextensions/python.config`) that points to `application:application`. The raw dataset is tracked with DVC (`artifacts/raw.csv.dvc`).

## Limitations

- **Small dataset.** 999 students, and the results come from a single train/test split.
- **Model selection uses the test set.** The best model is chosen by its test-set score, so the reported score is slightly optimistic. Cross-validated scores on a separate validation set would be more reliable.
- **The notebook scales before splitting.** The notebook fits its scaler on all rows before the split, which can leak a little information. The training pipeline in `src/` fits on the training set only.
- **Windows file paths.** `prediction_pipeline.py` loads the model with Windows-style paths (`artifacts\model.pkl`), so the web app currently runs on Windows. Switching to `os.path.join` would fix it on macOS and Linux.
- **Sensitive inputs.** The model uses attributes such as ethnicity group and lunch type. This is a learning project, and its predictions should not be used to make decisions about real students.

## Ideas for next steps

- Cross-validation and a separate validation set for model selection.
- Pin dependency versions. The saved files were created with scikit-learn 1.4.0.
- Fix the file paths, add a Dockerfile and add tests for the prediction pipeline.
- Track experiments and add model monitoring.

## Author

Parmeshwar Vishwakarma
