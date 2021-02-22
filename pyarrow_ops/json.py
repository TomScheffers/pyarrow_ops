import pyarrow as pa

def arr_to_map(arr):
    udt = pa.map_(pa.string(), pa.string())
    return arr.cast(udt)