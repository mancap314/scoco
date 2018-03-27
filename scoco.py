import pandas as pd
import os
from sklearn.metrics import roc_auc_score
from numpy import exp

def make_data(folder,categories, id_col='id'):
    '''
    concatenates horizontally the prediction files of a folder according to the id column according to the categories to predict
    :param folder: folder with the prediction files in csv format
    :param categories: categories to predict
    :param id_col: column with the row id, typically called 'id'
    :return: dataframe with id_col and one column for each category by each prediction file name '{category}_{file}'
    '''
    df_res = None
    for f in os.listdir(folder):
        if f.endswith('.csv') and not f.startswith('blending'): #go through all the prediction files
            df_ = pd.read_csv(os.path.join(folder, f))[[id_col] + categories]
            for category in categories:
                df_.rename(columns={category: '{}_{}'.format(category, f[:-4])}, inplace=True)
            if df_res is None:
                df_res = df_
            else:
                df_res = pd.merge(df_res, df_, on=id_col)
    return df_res

def get_assessment(train, data, categories):
    '''return dataframe with one row per method and one col per category to predict'''
    methods = [col.replace('{}_'.format(categories[0]), '') for col in data.columns.values if col.startswith(categories[0])]
    res = {'method': methods}
    for category in categories:
        data_category = data[[col for col in data.columns.values if col.startswith(category)]]
        scores_category = [roc_auc_score(train[category], data_category[col]) for col in data_category.columns.values]
        res[category] = scores_category
    return pd.DataFrame(res)



def blending(categories, train_file, cv_folder, prediction_folder, id_col='id', method='coeff', power=2):
    '''

    :param categories: categories to predict or dependant binary categorical variables
    :param train_file: path to the file containing labelled data used to train the models used for the predictions
    :param cv_folder: folder containing the cross-validation (oof, out-of-fold) predictions
    :param prediction_folder: folder containing the prediction on the test data
    :param id_col: name of the column containing the row id
    :param method: 'simple': merge predictions with the same weight ; 'coeff': weight proportional to cv-score,
    'poly': weight proportional to cv-score ** poser, 'exp': weight proportional to exp(cv-score)
    :param power: power used for method 'poly', ignored otherwise
    :return: persists the result on the test and on the cv data in the corresponding folders
    '''
    train = pd.read_csv(train_file)
    cv_predictions = make_data(cv_folder, categories)
    test_predictions = make_data(prediction_folder, categories)
    assessment = get_assessment(train, cv_predictions, categories)

    cv_blending = cv_predictions[[id_col]].copy() #will contain the result (blending) for the cv data
    pred_blending = test_predictions[[id_col]].copy() #will contain the result (blending) for the test data

    for category in categories:
        assessment_category = assessment[['method', category]]

        tr_pred = cv_predictions[[col for col in cv_predictions.columns.values if col.startswith(category)]]
        te_pred = test_predictions[[col for col in cv_predictions.columns.values if col.startswith(category)]]
        i = 0
        while assessment_category.shape[0] > 1:
            # take the worst model for this category in assessment
            worst_model = assessment_category.loc[assessment_category[category] == assessment_category[category].min(), 'method'].values[0]
            coeff_worst = assessment_category.loc[assessment_category['method'] == worst_model, category].values[0]
            assessment_category = assessment_category.loc[assessment_category['method'] != worst_model]
            # compute the correlation of this model with all the other models
            assessment_category['corr'] = [te_pred['{}_{}'.format(category, worst_model)].corr(te_pred['{}_{}'.format(category, model)]) for model in assessment_category['method'].tolist()]
            # compute the "badness" (correlation * 1/score) relative to the worst model for all the other models
            assessment_category['badness'] = assessment_category['corr'] * 1 / assessment_category[category]
            # select the model with the highest badness to mix with the worst model
            tomix = assessment_category.loc[assessment_category['badness'] == assessment_category['badness'].max(), 'method'].values[0]

            print('worst: {}, tomix: {}'.format(worst_model, tomix))

            # make a mix between worst_model and tomix

            coeff_tomix = assessment_category.loc[assessment_category['method'] == tomix, category].values[0]

            if method is 'simple':
                coeff_tomix, coeff_worst = 1, 1
            if method is 'poly':
                coeff_tomix = coeff_tomix ** power
                coeff_worst = coeff_worst ** power
            if method is 'exp':
                coeff_tomix = exp(coeff_tomix)
                coeff_worst = exp(coeff_worst)

            tr_pred['{}_mix_{}'.format(category, i)] = (coeff_worst * tr_pred['{}_{}'.format(category, worst_model)] +
                                                            coeff_tomix * tr_pred['{}_{}'.format(category, tomix)]) / (coeff_worst + coeff_tomix)
            te_pred['{}_mix_{}'.format(category, i)] = (coeff_worst * te_pred['{}_{}'.format(category, worst_model)] +
                                                            coeff_tomix * te_pred['{}_{}'.format(category, tomix)]) / (coeff_worst + coeff_tomix)
            # remove columns corresponding to worst_model and tomix
            tr_pred.drop(['{}_{}'.format(category, worst_model), '{}_{}'.format(category, tomix)], 1, inplace=True) #in train
            te_pred.drop(['{}_{}'.format(category, worst_model), '{}_{}'.format(category, tomix)], 1, inplace=True) #in test
            # remove the row corresponding to 'tomix' from assessment
            assessment_category = assessment_category.loc[assessment_category['method'] != tomix]
            assessment_category.reset_index(inplace=True)
            assessment_category.drop(['corr', 'badness', 'index'], 1, inplace=True)

            # append assessment corresponding to new mixed prediction
            to_append = pd.DataFrame({'method': ['mix_{}'.format(i)], category: [roc_auc_score(train[category], tr_pred['{}_mix_{}'.format(category, i)])]})
            assessment_category = assessment_category.append(to_append, ignore_index=True)
            i += 1

        # the last mix is named with suffix i-1, it's the blending prediction for the category
        pred_blending[category] = te_pred['{}_mix_{}'.format(category, i-1)]
        cv_blending[category] = tr_pred['{}_mix_{}'.format(category, i-1)]

    pred_blending.to_csv(os.path.join(prediction_folder, 'blending_{}_prediction.csv'.format(method)), index=False)
    cv_blending.to_csv(os.path.join(cv_folder, 'blending_{}_cv.csv'.format(method)), index=False)


