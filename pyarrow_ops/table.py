import os, time, s3fs
import pyarrow as pa
import pyarrow.parquet as pq
from ops import drop_duplicates, head, split

class ParquetUniqueDataset(pq.ParquetDataset):
    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs
        # super().__init__(*args, **kwargs) 
        self.path = args[0]
        self.tables = {}
        self.load()

    def load(self, verbose=False):
        # Initiate the parent class
        super().__init__(*self.args, **self.kwargs) 

        self.meta = self.pieces[0].get_metadata()
        self.columns = [c['path_in_schema'] for c in self.meta.row_group(0).to_dict()['columns']]

        # Partition information
        self.partition_cols = [c[0] for c in self.pieces[0].partition_keys]
        self.partitions_ = [p.partition_keys for p in self.pieces]
        self.partitions_val = [tuple(v[1] for v in p) for p in self.partitions_]      

        if verbose:
            print("Loaded all the data:", [p.path for p in self.pieces])
            print("Column names:", self.columns)
    
    def set_unique(self, columns):
        self.unique_cols = [u for u in columns if u not in self.partition_cols]
        return self

    def partition_dict(self, partition_val):
        return dict(zip(self.partition_cols, list(partition_val)))

    def get_path(self, partition_val, name):
        partition = self.partition_dict(partition_val)
        return self.path + '/' + '/'.join(str(k) + '=' + str(v) for k, v in partition.items()) + '/' + name + '.parquet'

    def get_idxs(self, partition_val):
        return [idx for idx, p in enumerate(self.partitions_val) if p == partition_val]

    # Cleaning tables
    def concat(self, tables):
        return pa.concat_tables([t.select(self.columns) for t in tables])

    def sanitize(self, table):
        # TODO: Add casting to default schema of class
        return table.select(self.partition_cols + self.columns)

    def cleanup(self):
        for p in set(self.partitions_val):
            print("Cleaning up:", p)
            table = self.read_parts(p)
            table_dedup = drop_duplicates(table, on=self.unique_cols, keep='last')
            self.save(self.deduplicate(table), p)

    # Reading / writing tables
    def read_parts(self, partition_val=None):
        # See what pieces we need to load
        idxs = (self.get_idxs(partition_val) if partition_val else range(len(self.pieces)))
        for i in idxs:
            if self.pieces[i].path not in self.tables.keys():
                print("Reading {} as it is not in cache".format(self.pieces[i].path))
                self.tables[self.pieces[i].path] = self.pieces[i].read(columns=self.columns, partitions=self.partitions)
        return self.concat([self.tables[self.pieces[i].path] for i in idxs])

    def save(self, table, partition_val):
        paths_old = [self.pieces[i].path for i in self.get_idxs(partition_val)]
        paths_new = [self.get_path(partition_val, 'file0')]

        # Delete old which are not written in new
        for path in [p for p in paths_old if p not in paths_new]:
            os.remove(path)

        # Write new table
        pq.write_table(table.select(self.columns), paths_new[0])

        # Add table to caching
        self.tables[paths_new[0]] = table

        # Reload the Dataset if file names have changed
        if set(paths_old) != set(paths_new):
            return True
        else:
            return False

    # Upsertion
    def upsert_part(self, table, partition_val, partition_idxs, keep):
        # If partition exists, gather original data, before deduplication. Else use new table
        rows_b4 = None
        if partition_val in self.partitions_val:
            table_part = self.read_parts(partition_val)
            rows_b4 = table_part.num_rows
            table_new = self.concat([table_part, table.take(partition_idxs)])
        else:
            table_new = table.take(partition_idxs)
        table_dedup = drop_duplicates(table_new, on=self.unique_cols, keep=keep)
        print("Upserting data for partition {0}. Added {1} unique records".format(partition_val, table_dedup.num_rows - rows_b4))
        return self.save(table_dedup, partition_val)

    def upsert(self, table, keep='last'):
        table = self.sanitize(table)
        # bools = self.pool.map(lambda p: self.upsert_part(*p), [(table, val, idxs, keep) for val, idxs in split(table=table, columns=self.partition_cols)])
        bools = [self.upsert_part(table, val, idxs, keep) for val, idxs in split(table=table, columns=self.partition_cols)]
        if max(bools):
            print("Reloading dataset!")
            self.load()

    # Deletion by table
    def delete_part(self, table, partition_val, partition_idxs):
        if partition_val in self.partitions_val:
            table_part = self.read_parts(partition_val)
            table_new = self.concat([table_part, table.take(partition_idxs)])
            table_dedup = drop_duplicates(table_new, on=self.unique_cols, keep='drop')
            print("Removing data for partition {0}. Removed {1} unique records".format(partition_val, table_part.num_rows - table_dedup.num_rows))
            return self.save(table_dedup, partition_val)
        else:
            print("There does not data for partition:", self.partition_dict(partition_val))
            return False

    def delete(self, table):
        table = self.sanitize(table)
        # bools = self.pool.map(lambda p: self.delete_part(*p), [(table, val, idxs) for val, idxs in split(table=table, columns=self.partition_cols)])
        bools = [self.delete_part(table, val, idxs) for val, idxs in split(table=table, columns=self.partition_cols)]
        if max(bools):
            self.load()

    # Delete full partition
    def delete_predicate(self, partition_val, predicate):
        return


if __name__ == '__main__':
    t = ParquetUniqueDataset('data/skus').set_unique(columns=['sku_key'])
    table = t.read()
    head(table)

    # Upsert functionality
    import numpy as np
    idxs = np.random.choice(table.num_rows, 100_000)
    table_new = table.take(idxs)

    t1 = time.time()
    t.upsert(table_new)

    # Remove functionality
    t2 = time.time()
    idxs = np.random.choice(table.num_rows, 100)
    t.delete(table.take(idxs))

    print("Time upsert / delete", t2 - t1, time.time() - t2)




