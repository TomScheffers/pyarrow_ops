import numpy as np
import pyarrow as pa
from collections import defaultdict

# Joining functionality
def join(left, right, on):
    # We want the smallest table to be on the right
    if left.num_rows >= right.num_rows:
        t1, t2 = left, right
    else:
        t1, t2 = right, left

    # Gather on columns
    idx1, idx2 = list(range(t1.num_rows)), list(range(t1.num_rows))

    # List of tuples of columns
    on1, on2 = [c.to_numpy() for c in t1.select(on).itercolumns()], [c.to_numpy() for c in t2.select(on).itercolumns()]

    # Zip idx / on values
    tup1, tup2 = list(zip(*[idx1] + on1)), list(zip(*[idx2] + on2))

    # Hash smaller table into dict {(on):[idx1, idx2, ...]}
    ht = defaultdict(list)
    for r in tup2:
        ht[r[1:]].append(r[0]) # Save index in hash
    m = [(r[0], i2) for r in tup1 for i2 in ht[r[1:]]]

    # Gather indices
    m1, m2 = zip(*m)
    l1, l2 = list(m1), list(m2)

    # Align tables
    fin = t1.take(l1)
    for c in t2.column_names:
        if c not in t1.column_names:
            fin = fin.append_column(c, t2.column(c).take(l2))
    return fin

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
    idxs = np.array(range(table.num_rows))
    for (col, op, value) in filters: #= or ==, !=, <, >, <=, >=, in and not in
        arr = table.column(col).to_numpy()
        f_idxs = arr_op_to_idxs(arr[idxs], op, value)
        idxs = idxs[f_idxs]
    return table.take(idxs)

# Splitting tables by columns
def split(table, columns, group=()):
    arr = table.column(columns[0]).to_numpy()
    if np.unique(arr).size < 1000:
        # This method is much faster for small amount of categories, but slower for large ones
        val_idxs = {v: (arr == v).nonzero()[0] for v in np.unique(arr)}
    else:
        un, rev = np.unique(arr, return_inverse=True)
        idxs = [[] for _ in un]
        [idxs[rev[i]].append(i) for i in range(len(rev))]
        val_idxs = dict(zip(un, idxs))
    
    # TODO: Do not take from table, but pass indexes instead. Take in underlying functions

    # Gathering splits of the table, this is the main botteneck for large amounts of categories
    tables = {v: table.take(i) for v, i in val_idxs.items()}
    if columns[1:]:
        return [s for v, t in tables.items() for s in split(t, columns[1:], group + (v,))]
    else:
        return [(group + (v,), t) for v, t in tables.items()]

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
        self.group_keys, self.group_tables = zip(*self.groups)
        self.set_methods()

    def __iter__(self):
        return iter(self.groups)

    # Aggregation methods
    def set_methods(self):
        for k, m in agg_methods.items():
            add_agg_method(self, k, m)

    def aggregate(self, methods):
        agg = dict(zip(self.columns, [pa.array(l) for l in zip(*self.group_keys)]))
        for col, f in methods.items():
            agg[col] = pa.array([f(t.column(col).to_numpy()) for t in self.group_tables])
        return pa.Table.from_arrays(list(agg.values()), names=list(agg.keys()))

    def agg(self, methods):
        methods = {col: agg_methods[m] for col, m in methods.items()}
        return self.aggregate(methods=methods)

    # TODO: Add Windowed methods
    def window(self, methods):
        raise Exception("Window function is not implemented yet")

def groupby(table, columns):
    return Grouping(table, columns)

# Drop duplicates
def set_idx(d, key, value, keep):
    if keep == 'last':
        d[key] = value
    elif keep == 'first':
        if key not in d:
            d[key] = value
    elif keep == 'drop':
        if key in d:
            d[key] = None
    else:
        raise Exception("Keep method {} not implemented!".format(keep))

def drop_duplicates(table, on=[], keep='last'):
    # List of tuples of columns
    idx = list(range(table.num_rows))
    val = [c.to_numpy() for c in (table.select(on) if on else table).itercolumns()]
    tup = list(zip(*[idx] + val))

    # Search distinct indices
    d = {}
    [set_idx(d, r[1:], r[0], keep) for r in tup]
    return table.take(sorted([i for i in d.values() if i is not None]))

# Show for easier printing
def head(table, n=10, max_width=100):
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