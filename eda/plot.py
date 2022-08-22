import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_unique(df,
                features,
                figsize=(10, 7),
                log=True,
                title="Number of unique values per feature TRAIN"):
    """绘制某个dataframe指定列的unique值的柱状图

    Parameters:
    -----------
    df : pd.DataFrame

    features : list of str
    """
    plt.figure(figsize=figsize)
    # d_features = list(train_transaction.columns[30:45])
    uniques = [len(df[col].unique()) for col in features]
    sns.set(font_scale=1.2)
    ax = sns.barplot(features, uniques, log=log)
    ax.set(xlabel='Feature', ylabel='log(unique count)', title=title)
    for p, uniq in zip(ax.patches, uniques):
        height = p.get_height()
        ax.text(p.get_x() + p.get_width() / 2., height + 10, uniq, ha="center")


def plot_categorical(train: pd.DataFrame,
                     test: pd.DataFrame,
                     feature: str,
                     target: str,
                     values: int = 5):
    """
    Plotting distribution for the selected amount of most frequent values between train and test
    along with distibution of target
    Args:
        train (pandas.DataFrame): training set
        test (pandas.DataFrame): testing set
        feature (str): name of the feature
        target (str): name of the target feature
        values (int): amount of most frequest values to look at
    """
    df_train = pd.DataFrame(data={feature: train[feature], 'isTest': 0})
    df_test = pd.DataFrame(data={feature: test[feature], 'isTest': 1})
    df = pd.concat([df_train, df_test], ignore_index=True)
    df = df[df[feature].isin(
        df[feature].value_counts(dropna=False).head(values).index)]
    train = train[train[feature].isin(
        train[feature].value_counts(dropna=False).head(values).index)]
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))
    sns.countplot(data=df.fillna('NaN'), x=feature, hue='isTest', ax=axes[0])
    sns.countplot(data=train[[feature, target]].fillna('NaN'),
                  x=feature,
                  hue=target,
                  ax=axes[1])
    axes[0].set_title(
        'Train / Test distibution of {} most frequent values'.format(values))
    axes[1].set_title(
        'Train distibution by {} of {} most frequent values'.format(
            target, values))
    axes[0].legend(['Train', 'Test'])
