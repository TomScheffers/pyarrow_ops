import numpy as np
import pyarrow as pa
from collections import defaultdict
import operator, itertools

def column(table, name):
    return table.column(name).combine_chunks()

# Array to idxs
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

def value_counter(arr):
    return {d['values']:d['counts'] for d in arr.value_counts().to_pylist()}

def split_array_v2(arr):
    dic = arr.dictionary_encode().dictionary.to_numpy(zero_copy_only=False)
    cnt = value_counter(arr)
    sort_idxs = pa.compute.sort_indices(arr).to_numpy()
    idxs, count = {}, 0
    for d in dic:
        idxs[d] = sort_idxs[count:count + cnt[d]].tolist()
        count += cnt[d]
    return idxs

# Joining functionality
def align_tables(t1, t2, l1, l2):
    # Align tables
    table = t1.take(l1)
    for c in t2.column_names:
        if c not in t1.column_names:
            table = table.append_column(c, t2.column(c).take(l2))
    return table

def single_key_hash_join(t1, t2, key):
    # Create idx_maps per distinct value
    #ht = defaultdict(list, split_array(column(t2, key)))
    ht = defaultdict(list)
    [ht[t].append(i) for i, t in enumerate(column(t2, key).to_numpy(zero_copy_only=False))]
    f = operator.itemgetter(*column(t1, key).to_numpy(zero_copy_only=False))
    idx_maps = f(ht)

    # Gather indices
    l1 = [i1 for i1, idx_map in enumerate(idx_maps) for i2 in idx_map]
    l2 = list(itertools.chain.from_iterable(idx_maps))
    return align_tables(t1, t2, l1, l2)

def multi_key_hash_join(t1, t2, on):
    # List of tuples of columns
    on1, on2 = [c.to_numpy() for c in t1.select(on).itercolumns()], [c.to_numpy() for c in t2.select(on).itercolumns()]

    # Zip idx / on values
    tup1 = map(hash, zip(*on1))
    tup2 = map(hash, zip(*on2))

    # Hash smaller table into dict {(on):[idx1, idx2, ...]}
    ht = defaultdict(list)
    [ht[t].append(i) for i, t in enumerate(tup2)]
    f = operator.itemgetter(*tup1)
    idx_maps = f(ht)

    # Gather indices
    l1 = [i1 for i1, idx_map in enumerate(idx_maps) for i2 in idx_map]
    l2 = list(itertools.chain.from_iterable(idx_maps))
    return align_tables(t1, t2, l1, l2)

def join(left, right, on):
    # We want the smallest table to be on the right
    if left.num_rows >= right.num_rows:
        t1, t2 = left, right
    else:
        t1, t2 = right, left

    # Choose join method
    if len(on) == 1:
        return single_key_hash_join(t1, t2, on[0])
    else:
        return multi_key_hash_join(t1, t2, on)

# Filter functionality
def arr_op_to_idxs(arr, op, value):
    if op in ['=', '==']:
        return np.where(arr == value)
    elif op == '!=':
        return np.where(arr != value)
    elif op == '<':
        return np.where(arr < value)
    elif op == '>':
        return np.where(arr > value)
    elif op == '<=':
        return np.where(arr <= value)
    elif op == '>=':
        return np.where(arr >= value)
    elif op == 'in':
        mask = np.isin(arr, value)
        return np.arange(len(arr))[mask]
    elif op == 'not in':
        mask = np.invert(np.isin(arr, value))
        return np.arange(len(arr))[mask]
    else:
        raise Exception("Operand {} is not implemented!".format(op))

def filters(table, filters):
    filters = ([filters] if isinstance(filters, tuple) else filters)
    # Filter is a list of (col, op, value) tuples
    idxs = np.arange(table.num_rows)
    for (col, op, value) in filters: #= or ==, !=, <, >, <=, >=, in and not in
        arr = table.column(col).to_numpy()
        f_idxs = arr_op_to_idxs(arr[idxs], op, value)
        idxs = idxs[f_idxs]
    return table.take(idxs)

# Splitting tables by columns
def split(table, columns, group=(), idx=None):
    # idx keeps track of the orginal table index, getting split recurrently
    if not isinstance(idx, np.ndarray):
        idx = np.arange(table.num_rows)
    val_idxs = split_array(table.column(columns[0]).combine_chunks())
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
        self.columns = columns
        self.groups = split(table, columns)
        self.group_keys, self.group_idxs = zip(*self.groups)
        self.set_methods()

    def __iter__(self):
        for g, idx in self.groups:
            yield g, self.table.take(idx)

    # Aggregation methods
    def set_methods(self):
        for k, m in agg_methods.items():
            add_agg_method(self, k, m)

    def aggregate(self, methods):
        agg = dict(zip(self.columns, [pa.array(l) for l in zip(*self.group_keys)]))
        # Gather columns to numpy
        data = {k: self.table.column(k).to_numpy() for k in methods.keys()}
        for col, f in methods.items():
            agg[col] = pa.array([f(data[col][i]) for i in self.group_idxs])
        return pa.Table.from_arrays(list(agg.values()), names=list(agg.keys()))

    def agg(self, methods):
        methods = {col: agg_methods[m] for col, m in methods.items()}
        return self.aggregate(methods=methods)

    # TODO: Add Windowed methods
    def window(self, methods):
        raise Exception("Window function is not implemented yet")

def groupby(table, by):
    return Grouping(table, by)

# Drop duplicates
def set_idx(d, key, value, keep):
    if keep == 'last':
        d[key] = value
    elif keep == 'first':
        if key not in d:
            d[key] = value
    elif keep == 'drop':
        d[key] = (None if key in d.keys() else value)
    else:
        raise Exception("Keep method {} not implemented!".format(keep))

def drop_duplicates(table, on=[], keep='last'):
    # List of tuples of columns
    idx = range(table.num_rows)
    val = [c.to_numpy() for c in (table.select(on) if on else table).itercolumns()]
    tup = map(hash, zip(*val))

    # Search distinct indices
    ht = defaultdict(list)
    [ht[t].append(i) for i, t in zip(idx, tup)]
    
    # Perform keep logic
    if keep == 'last':
        idxs = map(lambda x: x[-1], ht.values())
    elif keep == 'first':
        idxs = map(lambda x: x[0], ht.values())
    elif keep == 'drop':
        idxs = map(lambda x: x[0], filter(lambda x: len(x) == 1, ht.values()))
    return table.take(list(idxs))

# Show for easier printing
def head(table, n=5, max_width=100):
    # Extract head data
    t = table.slice(length=n)
    head = {k: list(map(str, v)) for k, v in t.to_pydict().items()}

    # Calculate width
    col_width = list(map(len, head.keys()))
    data_width = [max(map(len, h)) for h in head.values()]

    # Print data
    data = [list(head.keys())] + [[head[c][i] for c in head.keys()] for i in range(t.num_rows)]
    for i in range(len(data)):
        adjust = [w.ljust(max(cw, dw) + 2) for w, cw, dw in zip(data[i], col_width, data_width)]
        print(('Row  ' if i == 0 else str(i-1).ljust(5)) + "".join(adjust)[:max_width])
    print('\n')

# Table wrapper: does not work because pa.Table.from_pandas/from_arrays/from_pydict always returns pa.Table
class Table(pa.Table):
    def __init__(*args, **kwargs):
        super(Table, self).__init__(*args, **kwargs)
    
    def join(self, right, on):
        return join(self, right, on)
    
    def filters(self, filters):
        return filters(self, filters)
    
    def groupby(self, by):
        return Grouper(self, by)

    def drop_duplicates(self, on=[], keep='last'):
        return drop_duplicates(self, on, keep)

    def head(self, n=5):
        return head(self, n)

# Add methods to class pa.Table or instances of pa.Table: does not work because pyarrow.lib.Table is build in C
def add_table_methods(table):
    def join(self, right, on):
        return join(self, right, on)
    table.join = join
    
    def filters(self, filters):
        return filters(self, filters)
    table.filters = filters
    
    def groupby(self, by):
        return Grouper(self, by)
    table.groupby = groupby

    def drop_duplicates(self, on=[], keep='last'):
        return drop_duplicates(self, on, keep)
    table.drop_duplicates = drop_duplicates

    def head(self, n=5):
        return head(self, n)
    table.head = head
    

