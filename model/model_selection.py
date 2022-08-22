from unittest import skipUnless
from sklearn import ensemble
from sklearn import gaussian_process
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn import neighbors
from sklearn import svm, tree, discriminant_analysis
from xgboost import XGBClassifier
from sklearn import model_selection
import pandas as pd
#Machine Learning Algorithm (MLA) Selection and Initialization
MLA = [
    #Ensemble Methods
    ensemble.AdaBoostClassifier(),
    ensemble.BaggingClassifier(),
    ensemble.ExtraTreesClassifier(),
    ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier(),

    #Gaussian Processes
    gaussian_process.GaussianProcessClassifier(),

    #GLM
    linear_model.LogisticRegressionCV(),
    linear_model.PassiveAggressiveClassifier(),
    linear_model.RidgeClassifierCV(),
    linear_model.SGDClassifier(),
    linear_model.Perceptron(),

    #Navies Bayes
    naive_bayes.BernoulliNB(),
    naive_bayes.GaussianNB(),

    #Nearest Neighbor
    neighbors.KNeighborsClassifier(),

    #SVM
    svm.SVC(probability=True),
    svm.NuSVC(probability=True),
    svm.LinearSVC(),
    #Trees
    tree.DecisionTreeClassifier(),
    tree.ExtraTreeClassifier(),

    #Discriminant Analysis
    discriminant_analysis.LinearDiscriminantAnalysis(),
    discriminant_analysis.QuadraticDiscriminantAnalysis(),

    #xgboost: http://xgboost.readthedocs.io/en/latest/model.html
    XGBClassifier()
]


def multi_model_val(data, feature_col, target):
    """使用多种数据集来训练与评估模型

    Parameters:
    -----------
    data : pd.DataFrame
        数据
    feature_col : list of str
        特征列
    target : str
        标签

    Returns
    -------
    MLA_compare : pd.DataFrame
    """
    cv_split = model_selection.ShuffleSplit(n_splits=10,
                                            test_size=.3,
                                            train_size=.6,
                                            random_state=0)

    #create table to compare MLA metrics
    MLA_columns = [
        'MLA Name', 'MLA Parameters', 'MLA Train Accuracy Mean',
        'MLA Test Accuracy Mean', 'MLA Test Accuracy 3*STD', 'MLA Time'
    ]
    MLA_compare = pd.DataFrame(columns=MLA_columns)

    #create table to compare MLA predictions
    MLA_predict = data[target]

    #index through MLA and save performance to table
    row_index = 0
    for alg in MLA:

        #set name and parameters
        MLA_name = alg.__class__.__name__
        MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
        MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())

        #score model with cross validation: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate
        cv_results = model_selection.cross_validate(alg,
                                                    data[feature_col],
                                                    data[target],
                                                    cv=cv_split)

        MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()
        MLA_compare.loc[
            row_index,
            'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()
        MLA_compare.loc[
            row_index,
            'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()
        #if this is a non-bias random sample, then +/-3 standard deviations (std) from the mean, should statistically capture 99.7% of the subsets
        MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD'] = cv_results[
            'test_score'].std() * 3  #let's know the worst that can happen!
        #save MLA predictions - see section 6 for usage
        alg.fit(data[feature_col], data[target])
        MLA_predict[MLA_name] = alg.predict(data[feature_col])

        row_index += 1
    MLA_compare.sort_values(by=['MLA Test Accuracy Mean'],
                            ascending=False,
                            inplace=True)
    # MLA_compare后续可使用barplot可视化
    return MLA_compare
