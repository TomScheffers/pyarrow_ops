import pyarrow as pa
import numpy as np
import pyarrow.compute as c

# Cleaning functions
def clean_num(arr, impute=0.0, clip_min=None, clip_max=None):
    return (pa.array(np.nan_to_num(arr.to_numpy(zero_copy_only=False).astype(np.float64), nan=impute).clip(clip_min, clip_max)), )

def clean_cat(arr, categories=[]):
    arr = arr.cast(pa.string()).dictionary_encode()
    dic = arr.dictionary.to_pylist()
    if categories:
        d = {i:(categories.index(v) + 1 if v in categories else 0) for i, v in enumerate(dic)}
        d[-1] = 0 # NULLs
        return (pa.array(np.vectorize(d.get)(arr.indices.fill_null(-1).to_numpy())), ['Unknown'] + categories)
    else:
        return (c.add(arr.indices, pa.array([1])[0]).fill_null(0), ['Unknown'] + dic)

def clean_hot(arr, categories=[], drop_first=False):
    arr = arr.cast(pa.string())
    if categories:
        return ([c.equal(arr, v).fill_null(False) for v in categories], categories)
    else:
        un = [u for u in arr.unique().to_pylist() if u]
        return ([c.equal(arr, v).fill_null(False) for v in un], un)

