# Linear Regression Investigation: Chicago Taxi Fares

An educational machine learning investigation exploring how **learning rate, batch size, training duration, feature selection, and feature units** affect the training of a linear regression model.

This project uses the Chicago taxi fare dataset and instructional tools from Google's Machine Learning Course. The notebook records my exploration of model behavior and the questions that emerged while experimenting with training configurations.

> **Scope:** This is a learning investigation, not a validated or production-ready fare prediction system. All RMSE results reported below are **training metrics**; there is no held-out validation or test evaluation in the current notebook.

## Research questions

- How does the optimizer's learning rate affect convergence?
- How does changing batch size and the number of epochs affect training?
- Does adding trip duration to trip distance improve the observed training fit?
- What happens when trip duration is expressed in minutes rather than seconds?
- What additional testing would be necessary to assess generalization?

## Dataset

The notebook loads Google's Chicago taxi training dataset directly from:

`https://download.mlcc.google.com/mledu-datasets/chicago_taxi_train.csv`

- **Observations:** 31,694 taxi trips
- **Target:** `FARE`
- **Features investigated:** `TRIP_MILES`, `TRIP_SECONDS`, and derived `TRIP_MINUTES`
- **Other fields examined:** `COMPANY`, `PAYMENT_TYPE`, `TIP_RATE`

Initial exploration uses summary statistics, a correlation matrix, and a scatter-matrix visualization. In this dataset, `TRIP_MILES` has a strong positive correlation with `FARE` (approximately 0.975), as does `TRIP_SECONDS` (approximately 0.830). Correlation alone does not establish predictive performance on new data.

## Methods and tools

- **Python:** pandas, NumPy
- **Machine learning:** Keras, RMSprop optimizer, linear regression with a single Dense output
- **Visualization:** Plotly, Matplotlib, and `google-ml-edu` experiment utilities
- **Metric:** Root Mean Squared Error (RMSE), in fare units (USD)

The notebook defines model-building and training functions, varies training settings across six experiments, compares recorded metrics, and displays predictions for samples drawn from the training dataframe.

## Experiments and recorded results

| Experiment | Features | Learning rate | Epochs | Batch size | Final training RMSE |
|---|---|---:|---:|---:|---:|
| 1 — Baseline | Miles | 0.001 | 20 | 50 | $3.7459 |
| 2 — Higher learning rate | Miles | 1.0 | 20 | 50 | $6.8953 |
| 3 — Lower learning rate | Miles | 0.0001 | 20 | 50 | $21.9655 |
| 4 — Larger batch / longer training | Miles | 0.001 | 50 | 500 | $3.9483 |
| 5 — Add duration in seconds | Miles, seconds | 0.001 | 20 | 50 | $3.6101 |
| 6 — Express duration in minutes | Miles, minutes | 0.001 | 20 | 50 | $3.4780 |

These values come from the notebook's **saved runs**. Model initialization and training randomness mean that rerunning the notebook may produce different results. Experiment 4 changes both batch size and epochs, so its results cannot isolate the effect of either setting.

## Observations

1. **Learning rate:** A learning rate of 1.0 produced a higher final training RMSE than the baseline; a learning rate of 0.0001 did not reach the baseline's training fit within 20 epochs. This illustrates the importance of learning rate relative to the training budget.
2. **Feature selection:** Including trip duration in seconds alongside trip miles reduced recorded training RMSE from **$3.7459** to **$3.6101**.
3. **Feature units:** Converting trip duration from seconds to minutes reduced recorded training RMSE again, to **$3.4780**. This change does **not** add information to the dataset; it raises a question about the effects of feature magnitude on optimization.
4. **Interpretation:** Lower training RMSE is not proof of better performance on unseen observations. RMSE also differs from mean absolute error; it penalizes large prediction errors more strongly.

## Limitations and next steps

- Split the data into training, validation, and held-out test sets.
- Select hyperparameters using validation metrics rather than training metrics.
- Compare against a simple baseline and report test RMSE and MAE.
- Standardize numeric features and compare convergence under controlled conditions.
- Hold other settings constant when isolating a hyperparameter; use random seeds and repeated runs where appropriate.
- Inspect residuals and investigate cases with large fare prediction errors.

The current prediction demonstration samples rows from the **same dataframe used for training**, so it is not an independent validation study.

## Repository contents

- [`linear-regression_Chicago-taxi.ipynb`](linear-regression_Chicago-taxi.ipynb) — notebook containing the linear regression experiments **and earlier probability/sampling simulations**. The sampling simulations are a separate learning topic and are not part of the fare prediction experiments.

## How to view and run

1. Open [`linear-regression_Chicago-taxi.ipynb`](linear-regression_Chicago-taxi.ipynb) on GitHub to browse the notebook and saved outputs, or open it in Google Colab.
2. Run the cells in order in a Python environment with the required scientific Python and Keras packages. The notebook installs `google-ml-edu==0.1.3` in a code cell and fetches the taxi dataset from Google's URL, so internet access is required to rerun it.
3. Expect small differences in results due to random initialization and software environments.

## Attribution

Dataset and educational experiment utilities: Google Machine Learning Course. This repository documents a course-informed study and experimental observations; it should not be interpreted as claiming that all underlying instructional code or materials were independently created.
