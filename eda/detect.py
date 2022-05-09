import pandas as pd
import numpy as np
import os
import seaborn as sns


def get_files(path):
    """列举某个目录下所有文件及其大小
    """
    print('# File sizes')
    for f in os.listdir(path):
        if 'zip' not in f:
            print(
                f.ljust(30) +
                str(round(os.path.getsize(os.path.join(path, f)) /
                          1000000, 2)) + 'MB')


def detect_null(df, limit=10):
    """缺失值情况分析
    """
    missing_values_count = df.isnull().sum()
    missing_values_rate = df.isnull().sum() / df.shape[0] * 100
    missing_value = pd.concat([missing_values_count, missing_values_rate],
                              axis=1)
    missing_value.columns = ["null_count", "null_rate"]
    print(missing_value[0:limit])
    total_cells = np.product(df.shape)
    total_missing = missing_values_count.sum()
    print("----------")
    print("total missing rate = ", (total_missing / total_cells) * 100)


def detect_imbalance(df, target):
    """检测数据集不平衡的情况
    """
    x = df[target].value_counts().index
    y = df[target].value_counts().values
    f = sns.barplot(x, y)
    f.set_title(f"Data imbalance - {target}")
    for row, col in zip(x, y):
        #在柱状图上绘制该类别的数量
        f.text(row, col, col, color="black", ha="center")
    print("negative samples/ postive samples : ", y[0] / y[1])
