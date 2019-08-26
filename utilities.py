import pandas
#Print an entire dataframe or a series
def full_print(obj):
    with pandas.option_context('display.max_rows', None, 'display.max_columns', None):
        print(obj)
