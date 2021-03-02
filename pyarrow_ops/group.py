import numpy as np
import pyarrow as pa
from pyarrow_ops.helpers import combine_column, columns_to_array, groupify_array

# Splitting tables by columns
def split_array(arr):
    arr = arr.dictionary_encode()
    ind, dic = arr.indices.to_numpy(zero_copy_only=False), arr.dictionary.to_numpy(zero_copy_only=False)

    if len(dic) < 1000:
        # This method is much faster for small amount of categories, but slower for large ones
        return {v: (ind == i).nonzero()[0] for i, v in enumerate(dic)}
    else:
        idxs = [[] for _ in dic]
        [idxs[v].append(i) for i, v in enumerate(ind)]
        return dict(zip(dic, idxs))

def split(table, columns, group=(), idx=None):
    # idx keeps track of the orginal table index, getting split recurrently
    if not isinstance(idx, np.ndarray):
        idx = np.arange(table.num_rows)
    val_idxs = split_array(combine_column(table, columns[0]))
    if columns[1:]:
        return [s for v, i in val_idxs.items() for s in split(table, columns[1:], group + (v,), idx[i])]
    else:
        return [(group + (v,), i) for v, i in val_idxs.items()]

# Grouping / groupby methods
agg_methods = {
    'sum': np.sum,
    'max': np.max,
    'min': np.min,
    'mean': np.mean,
    'median': np.median
}
def add_agg_method(self, name, method):
    def f(agg_columns=[]):
        methods = {col: method for col in (agg_columns if agg_columns else self.table.column_names) if col not in self.columns}
        return self.aggregate(methods=methods)
    setattr(self, name, f)

class Grouping():
    def __init__(self, table, columns):
        self.table = table
        self.columns = list(set(columns))

        # Initialize array + groupify
        self.arr = columns_to_array(table, columns)
        self.dic, self.counts, self.sort_idxs, self.bgn_idxs = groupify_array(self.arr)

        # Create index columns
        self.table_new = self.table.select(self.columns).take(self.sort_idxs[self.bgn_idxs])

        self.set_methods()

    def __iter__(self):
        for i in range(len(self.dic)):
            idxs = self.sort_idxs[self.bgn_idxs[i] : self.bgn_idxs[i] + self.counts[i]]
            yield {k:v[0] for k, v in self.table_new.take([i]).to_pydict().items()}, self.table.take(idxs)

    # Aggregation methods
    def set_methods(self):
        for k, m in agg_methods.items():
            add_agg_method(self, k, m)

    def aggregate(self, methods):
        data = {k: self.table.column(k).to_numpy() for k in methods.keys()}
        for col, f in methods.items():
            vf = np.vectorize(f, otypes=[object])
            agg_arr = vf(np.split(data[col][self.sort_idxs], self.bgn_idxs[1:]))
            self.table_new = self.table_new.append_column(col, pa.array(agg_arr))
        return self.table_new

    def agg(self, methods):
        methods = {col: agg_methods[m] for col, m in methods.items()}
        return self.aggregate(methods=methods)

def groupby(table, by):
    return Grouping(table, by)