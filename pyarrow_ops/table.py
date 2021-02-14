import os
import pyarrow as pa
import pyarrow.parquet as pq

class ParquetUniqueDataset():
    def __init__(self, path, unique_cols=[]):
        self.path = path
        self.load()
        self.unique_cols = [u for u in unique_cols if u not in self.partition_cols]

        if not self.unique_cols:
            print("Warning: there are no unique columns specified, there is no added value in using this wrapper.")

    def load(self, verbose=False):
        # Loading data as ParquetDataset
        self.data = pq.ParquetDataset(self.path)
        self.pieces = self.data.pieces
        self.metas = [p.get_metadata() for p in self.pieces]
        self.columns = [c['path_in_schema'] for c in self.metas[0].row_group(0).to_dict()['columns']]

        # Partition information
        self.partition_cols = [c[0] for c in self.data.pieces[0].partition_keys]
        self.partitions = [p.partition_keys for p in self.pieces]
        self.partitions_val = [tuple(v[1] for v in p) for p in self.partitions]        

        if verbose:
            print("Loaded all the data:", [p.path for p in self.pieces])
            print("Column names:", self.columns)

    def partition_dict(self, partition_val):
        return dict(zip(self.partition_cols, list(partition_val)))

    def get_path(self, partition_val, name):
        partition = self.partition_dict(partition_val)
        return self.path + '/' + '/'.join(str(k) + '=' + str(v) for k, v in partition.items()) + '/' + name + '.parquet'

    def get_idxs(self, partition_val):
        return [idx for idx, p in enumerate(self.partitions_val) if p == partition_val]

    def save(self, table, partition_val):
        paths_old = [self.pieces[i].path for i in self.get_idxs(partition_val)]
        paths_new = [self.get_path(partition_val, 'file0')]

        # Delete old which are not written in new
        for path in [p for p in paths_old if p not in paths_new]:
            os.remove(path)

        # Write new table
        print("Writing new table to:", paths_new[0], table.num_rows)
        pq.write_table(table.select(self.columns), paths_new[0])

        # Reload the Dataset if files have changes
        if set(paths_old) != set(paths_new):
            self.load()

    def concat(self, tables):
        return pa.concat_tables([t.select(self.columns) for t in tables])

    def sanitize(self, table):
        return table.select(self.partition_cols + self.columns)

    def split(self, table):
        df_part = table.select(self.partition_cols).to_pandas()
        for key, df_group in df_part.groupby(self.partition_cols):
            yield (key if isinstance(key, tuple) else (key,)), table.take(df_group.index.values)

    def remove(self, table, values):
        return

    def deduplicate(self, table):
        uniq = table.select(self.unique_cols).to_pandas()
        idxs = uniq.drop_duplicates().index.values
        return table.take(idxs)

    def read_part(self, partition_val):
        idxs = self.get_idxs(partition_val)
        return self.concat([self.pieces[i].read(columns=self.columns, partitions=self.data.partitions) for i in idxs])

    def cleanup(self):
        for p in set(self.partitions_val):
            print("Cleaning up:", p)
            table = self.read_part(p)
            self.save(self.deduplicate(table), p)

    def read(self, columns=None, use_threads=True, use_pandas_metadata=False):
        return self.data.read(columns, use_threads, use_pandas_metadata)

    def upsert(self, table):
        table = self.sanitize(table)
        for partition_val, table_part in self.split(table):
            # If partition exists, gather original data, before deduplication
            if partition_val in self.partitions_val:
                print("Upserting new data for partition", partition_val)
                table_part = self.concat([table_part, self.read_part(partition_val)])
            self.save(self.deduplicate(table_part), partition_val)

    def delete(self, table):
        indices = table.select(self.partition_cols + self.unique_cols) 
        for partition_val, table_part in self.split(table):
            if partition_val in self.partitions_val:
                table = self.remove(self.read_part(partition_val), table_part)
                self.save(table_part, partition_val)
            else:
                print("Values already deleted for partition:", self.partition_dict(partition_val))

t = ParquetUniqueDataset('data/skus', unique_cols=['sku_key'])
#t.cleanup()
table = t.read()

import numpy as np
idxs = np.random.choice(table.num_rows, 100_000)
table_new = table.take(idxs)

t.upsert(table_new)




