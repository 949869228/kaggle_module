import pandas as pd
import numpy as np
import os
import seaborn as sns


def show_files(path):
    """列举某个目录下所有文件及其大小
    """
    print('File name'.ljust(30), ' File sizes')
    for f in os.listdir(path):
        if 'zip' not in f:
            print(
                f.ljust(30) +
                str(round(os.path.getsize(os.path.join(path, f)) /
                          1000000, 2)) + 'MB')


def detect_null(df, limit=None):
    """缺失值情况分析
    """
    missing_values_count = df.isnull().sum()
    missing_values_rate = df.isnull().sum() / df.shape[0] * 100
    missing_value = pd.concat([missing_values_count, missing_values_rate],
                              axis=1)
    missing_value.columns = ["null_count", "null_rate"]
    missing_value = missing_value.sort_values(by='null_rate')
    if limit is None:
        print(missing_value)
    else:
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


def get_dtypes(data, drop_col=[]):
    """Return the dtypes for each column of a pandas Dataframe

    Parameters
    ----------
    data : pandas Dataframe

    drop_col : columns to omit in a list

    Returns
    -------
    str_var_list, num_var_list, all_var_list
    
    """

    name_of_col = list(data.columns)
    num_var_list = []
    str_var_list = []
    all_var_list = []

    str_var_list = name_of_col.copy()
    for var in name_of_col:
        # check if column belongs to numeric type
        if (data[var].dtypes in (np.int, np.int64, np.uint, np.int32, np.float,
                                 np.float64, np.float32, np.double)):
            str_var_list.remove(var)
            num_var_list.append(var)
    # drop the omit column from list
    for var in drop_col:
        if var in str_var_list:
            str_var_list.remove(var)
        if var in num_var_list:
            num_var_list.remove(var)

    all_var_list.extend(str_var_list)
    all_var_list.extend(num_var_list)
    return str_var_list, num_var_list, all_var_list