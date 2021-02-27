import pyarrow as pa 
import pyarrow.parquet as pq
from pyarrow_ops import Table, join, filters, groupby, head, drop_duplicates

# Read data
t1 = pq.ParquetDataset('data/skus/file0.parquet').read()
t2 = pq.ParquetDataset('data/stock_current/file0.parquet').read()
j = join(t1, t2, on=['sku_key'])
g = groupby(j, by=['option_key']).agg({'economical': 'sum'})

head(g)