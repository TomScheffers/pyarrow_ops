import pyarrow as pa 
from pyarrow_ops.ml import clean_num, clean_cat, clean_hot

# Numericals
arr = pa.array(["1", "2", None, "4"])
print(clean_num(arr, impute=0, clip_min=2.))

# Categoricals
arr = pa.array(["A", "B", None, "C"])
print(clean_cat(arr, categories=["A", "C", "B"]))
print(clean_cat(arr))

# One hots
arr = pa.array(["A", "B", None, "C"])
print(clean_hot(arr, categories=["A", "C", "B"]))
print(clean_hot(arr))
