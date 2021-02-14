# Pyarrow ops
Pyarrow ops is Python libary for data crunching operations directly on the pyarrow.Table class, using only numpy. For convenience, function naming and behavior tries to replicates that of the Pandas API. The performance is currently on par with pandas, however performance can be significantly improved by utilizing pyarrow.compute functions or improving algorithms in numpy.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install pyarrow_ops.

```bash
pip install pyarrow_ops
```

## Usage
See test_func.py for full runnable test example

```python
import pyarrow as pa 
from pyarrow_ops import join, filters, groupby, head, drop_duplicates

# Create pyarrow.Table
t = pa.Table.from_pydict({
    'Animal': ['Falcon', 'Falcon', 'Parrot', 'Parrot', 'Parrot'],
    'Max Speed': [380., 370., 24., 26., 24.]
})
head(t) # Use head to print, like df.head()

# Drop duplicates based on column values
d = drop_duplicates(t, on=['Animal'], keep='first')

# Groupby iterable
for key, value in groupby(t, ['Animal']):
    print(key)
    head(value)

# Group by aggregate functions
g = groupby(t, ['Animal']).median()
g = groupby(t, ['Animal']).sum()
g = groupby(t, ['Animal']).min()
g = groupby(t, ['Animal']).agg({'Max Speed': 'max'})

# Group by window functions

# Use filter predicates using list of tuples (column, operation, value)
f = filters(t, ('Animal', '=', 'Falcon'))
f = filters(t, [('Animal', 'not in', ['Falcon', 'Duck']), ('Max Speed', '<', 25)])

# Join operations (currently performs inner join)
t2 = pa.Table.from_pydict({
    'Animal': ['Falcon', 'Parrot'],
    'Age': [10, 20]
})
j = join(t, t2, on=['Animal'])
```

### To Do's
- [x] Improve groupby speed by not create copys of table
- [ ] Add window functions on Grouping class
- [ ] Improve speed of split function by avoiding for loops
- [ ] Allow for functions to be class methods of pa.Table (t.groupby(['Animal']))*
- [ ] Extend the pq.ParquetDataset class (leverage partitions for joins)
- [ ] Add more join options (left, right, outer, full, cross)

*One of the main difficulties is that the pyarrow classes are written in C and do not have a __dict__ method, this hinders inheritance, adding classmethods, etc.

## Relation to pyarrow
In the future many of these functions might be obsolete by enhancements in the pyarrow package, but for now it is a convenient alternative to switching back and forth between pyarrow and pandas.

## Contributing
Pull requests are very welcome, however I believe in 80% of the utility in 20% of the code. I personally get lost reading the tranches of the pandas source code. If you would like to seriously improve this work, please let me know!