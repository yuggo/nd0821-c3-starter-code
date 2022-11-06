# Model Card

## Model Details
This model has been created by Denis Vuckovac and it is based on Gradient Boosting from the sklearn library.

## Intended Use
The aim of this model is to predict if a person's annual income will exceed 50,000 USD or not based on available census data.

## Training Data
The data used in this project is the Census Income Data Set provided by UCI Machine Learning Repository.
The training subset comprises 80% of this file, with the remainder being the testing set. Cross-validation of the training data is used for parameter tuning.

## Evaluation Data

## Metrics
The final model has the following metrics
Precision: 0.8019578313253012
Recall: 0.6893203883495146
Fbeta: 0.7413853115210581

The model performance based on slices of education are the following.

education,precision,recall,fbeta
9th,1.000,0.333,0.500
Bachelors,0.788,0.821,0.804
Assoc-voc,0.794,0.667,0.725
HS-grad,0.859,0.385,0.532
Some-college,0.701,0.556,0.621
Doctorate,0.853,0.928,0.889
Prof-school,0.875,0.975,0.922
Masters,0.854,0.881,0.867
7th-8th,0.750,0.375,0.500
11th,1.000,0.600,0.750
1st-4th,1.000,0.000,0.000
Assoc-acdm,0.806,0.707,0.753
12th,1.000,0.222,0.364
10th,1.000,0.188,0.316
Preschool,1.000,1.000,1.000
5th-6th,1.000,0.200,0.333


## Ethical Considerations
Race and Sex play an important role in predictions and these factors might be solely confounding and not causing success as measured by income.

## Caveats and Recommendations
Given the models performance on 1st to 4th grade education subjects we might want to consider these outliers and evaluate them further.

The model has consistently higher precision than recal, consider this given your model requirements.

