import pyarrow as pa 
from pyarrow_ops import head, TableCleaner

# Training data
t1 = pa.Table.from_pydict({
    'Animal': ['Falcon', 'Falcon', 'Parrot', 'Parrot', 'Parrot'],
    'Max Speed': [380., 370., None, 26., 24.],
    'Value': [2000, 1500, 10, 30, 20],
})

# Create TableCleaner
cleaner = TableCleaner()
cleaner.register_numeric('Max Speed', impute='min', clip=True)
cleaner.register_label('Animal', categories=['Goose', 'Falcon']) # Categories is optional, unknown values get set to 0
cleaner.register_one_hot('Animal')

# Clean table and split into train/test
X, y = cleaner.clean_table(t1, label='Value')
head(X)
X_train, X_test, y_train, y_test = cleaner.split(X, y)


# Train a model + save cleaner dictionary for reuse (serialize to JSON or pickle)
cleaner_dict = cleaner.to_dict()
for c in cleaner_dict:
    print(c)

# Prediction data
t2 = pa.Table.from_pydict({
    'Animal': ['Falcon', 'Goose', 'Parrot', 'Parrot'],
    'Max Speed': [380., 10., None, 26.]
})
new_cleaner = TableCleaner().from_dict(cleaner_dict)
X_pred = new_cleaner.clean_table(t2)
head(X_pred)