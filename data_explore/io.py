"""
将数据转化为arraow格式, 以减少数据的io事件。
"""
import pandas as pd
import pyarrow as pa


def save_arrow(data, filepath, max_chunksize=None):
    """save dataframe to a arrow file

    Parameters:
    -----------
    data : pandas.DataFrame
        data to save
    filepath : str
    max_chunksize : int,option

    Examples:
    ---------
    >>> data = pd.DataFrame()
    >>> save_arrow(data, 'data.arrow')
    """
    table = pa.Table.from_pandas(data)
    with pa.OSFile(filepath, 'wb') as sink:
        with pa.RecordBatchFileWriter(sink, table.schema) as writer:
            writer.write_table(table, max_chunksize=max_chunksize)


def read_arrow(filepath, n_bacth=None):
    """read arrow file to dataframe

    Parameters:
    -----------
    filepath : str
        path of arrow file.
    n_batch : int
        if you just want to get nth bacth, use this parameter.

    Returns:
    --------
    data : pd.DataFrame
    """
    source = pa.memory_map(filepath, 'r')
    batch_group = pa.ipc.RecordBatchFileReader(source)
    if n_bacth is None:
        return batch_group.read_pandas()
    else:
        return batch_group.get_batch(n_bacth).to_pandas()