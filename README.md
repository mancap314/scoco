This is the implementation of a method to blend together several predictions.

## Parameters
The main function is `scoco.blending`. It takes as arguments the `categories` to predict.
Those `categories` are binary categorical dependant variables to predict.
`train_file` is a file containing the ground truth, with at least one id column called per default
`id` (if this id column has another name, put this name for the parameter `id_col`) and the true
value of the categories for those items. `cv_folder` is the path to the folder containing
the cross-validation (out-of-folds) predictions computed on the `train` file.
`prediction_folder` is the folder containing the real prediction data. On both folders, the files
must be called the same. `method` is the method used to blend the prediction together

## Blending methods
Basically, this blending takes the file with the worst (out-of-fold) score, called *worst*,
and then compute the *badness* of each model according to *worst* according to the formula
_badness(model) := correlation(worst, model) * oof_score(model)_. It takes then the model
with the highest badness and mixes it with the worst model:
_new_model = (coeff_worst * model_worst + coeff_highest_badness * model_highest_badness) / (coeff_worst + coeff_highest_badness)_
And then *model_worst* and *model_highest_badness* are removed from the list of models, and this
procedure goes until there is just one model left. This is done for every category to predict.

The different values for the parameter `method` are:
* `simple`: *coeff_worst* and *coeff_highest_badness* are always equal to 1.
* `coeff` (default): *coeff_worst = oof_score(model_worst)* and *coeff_highest_badness = oof_score(model_highest_badness)*
* `poly`: *coeff_worst = oof_score(model_worst) ** `power`* and *coeff_highest_badness = oof_score(model_highest_badness) ** `power`*
* `exp`: *coeff_worst = exp(oof_score(model_worst))* and *coeff_highest_badness = exp(oof_score(model_highest_badness))*

NB: The argument `power` is used only for the method `poly`, otherwise ignored.

## example
*example.py* is a programm formatting the iris dataset, binarizing the target variable, dividing
the data in a train and test data, applying different classifiers to generate oof-prediction files
in the *iris_cv_folder* and prediction files in *iris_prediction_folder* (both folders are
created if not existent). And then it applies the *blending* method on it.
