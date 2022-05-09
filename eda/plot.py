import matplotlib.pyplot as plt
import seaborn as sns


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
