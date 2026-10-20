# Spam Classification

This project compares Logistic Regression, Decision Tree, kNN, and exploratory PCA analysis using the UCI Spambase dataset.

## Run

1. Open R or RStudio in this folder.
2. Install the required package once with install.packages("tree").
3. Run spambase_classification.R.

The script keeps the original R workflow and uses a stratified train/validation/test split. kNN k selection and model selection use the validation set. The test set is reserved for final reporting.
