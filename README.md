This project aims to build a machine learning solution capable of analyzing customer behavior, identifying distinct customer profiles through segmentation, and predicting the risk of subscription cancellation (churn).
The objective is to implement a comprehensive solution, ranging from data exploration to the deployment of a prediction application.

## MLflow tracking

Every experiment is logged to an MLflow SQLite store at the project root (`mlflow.db`), with artifacts and trained models stored under `mlruns/`.

- `notebooks/classification.ipynb` - one run per baseline model (hyperparameters, metrics, model).
- `notebooks/training_optimisation.ipynb` - one parent run per tuned model family, plus nested runs for the top GridSearchCV candidates; logs best params and test metrics.
- `notebooks/train_final_model.ipynb` - a single `final-model` run with test metrics, a confusion matrix artifact, and the final XGBoost model.

### Explore the runs

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

### Run order

eda -> preprocessing -> classification -> clustring -> clustring_vs_classifcation -> training_optimisation -> train_final_model

## Churn prediction web interface

A Streamlit app lets you enter a customer's data in a form and get back the predicted churn probability.

```bash
streamlit run app/app.py
```

Then open the URL shown in the terminal (default http://localhost:8501)

- Features are transformed with the exact preprocessing used to train the model. The fitted preprocessor is auto-built on first run and cached at `data/processed/models/preprocessor.pkl`.
- Result shows the churn probability (%) and a verdict (likely to stay / high churn risk).
