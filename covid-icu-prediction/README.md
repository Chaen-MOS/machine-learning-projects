# COVID-19 ICU Prediction

This notebook compares Lasso Regression, Linear Regression, an Artificial Neural Network, and KNN Regression for next-day national ICU prediction.

The primary evaluation uses chronological train, validation, and test periods. Feature scaling is fitted on the training period only. A previous-day ICU persistence baseline is included.

Shuffled results are retained only as secondary sensitivity analysis and should not be presented as the main forecasting result.

The notebook downloads current public data from the Malaysia Ministry of Health and CITF GitHub repositories. Re-run the notebook to regenerate metrics; existing outputs are not treated as newly verified results.
