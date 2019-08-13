#Obtain a list of columns by type
def get_cols_by_dtype(df, dtype):
    s_dtypes = df.dtypes
    return s_dtypes[s_dtypes == dtype].index.tolist()
#Obtain a list of object columns
def get_object_cols(df):
    s_dtypes = df.dtypes
    return s_dtypes[s_dtypes == 'object'].index.tolist()
#Obtain a list of float columns
def get_float_cols(df):
    s_dtypes = df.dtypes
    return s_dtypes[s_dtypes == 'float'].index.tolist()
#Obtain a list of int columns
def get_int_cols(df):
    s_dtypes = df.dtypes
    return s_dtypes[s_dtypes == 'int'].index.tolist()
#Obtain a list of categorical columns
def get_cat_cols(df):
    s_dtypes = df.dtypes
    return s_dtypes[s_dtypes == 'category'].index.tolist()
#Obtain a list of numerical columns
def get_num_cols(df):
    s_dtypes = df.dtypes
    return s_dtypes[(s_dtypes == 'int') | (s_dtypes == 'float')].index.tolist()

#Tackle cases where we have to get rid of a dollar sign in the front
def remove_dollar(obj):
    if isinstance(obj, float):
        return obj
    if isinstance(obj, int):
        return float(obj)
    if not isinstance(obj, str):
        print(f'{obj} is weird!')
        return np.nan
    string = obj
    if string[0] == '$':
        try:
            num = float(string[1:])
            return num
        except ValueError as e:
            print(f'{string} is weird!')
            return np.nan
    else:
        try:
            num = float(string)
            return num
        except ValueError as e:
            print(f'{string} is weird!')
            return np.nan
#Tackle cases where we have to deal with a percentage sign in the end
def remove_percentage(obj):
    if isinstance(obj, float):
        return obj
    if isinstance(obj, int):
        return float(obj)
    if not isinstance(obj, str):
        print(f'{obj} is weird!')
        return np.nan
    string = obj
    if string[-1] == '%':
        try:
            num = float(string[:-1])
            return num/100
        except ValueError as e:
            print(f'{string} is weird!')
            return np.nan
    else:
        try:
            num = float(string)
            return num
        except ValueError as e:
            print(f'{string} is weird!')
            return np.nan
def _series_forcefillna(s, preserve_int = False):
    dtype = s.dtype
    if dtype == 'int':
        if preserve_int:
            s = s.fillna(int(s.mean()))
        else:
            s = s.fillna(s.mean())
    elif dtype == 'float':
        s = s.fillna(s.mean())
    else:#bool, category, object
        s = s.fillna(s.mode().iloc[0])
    return s
#There are three ways to use this function.
#1.Use fill_list to determine which cols to fill
#2.Use exempt_list to determine which cols not to fill
#3.Use fill_<type> to fill cols by type
def forcefillna(df, preserved_int = False, preserved_int_list = [], fill_list = None, exempt_list = None, fill_float = True, fill_int = True, fill_cat = True, fill_obj = True):
    if isinstance(df, pd.Series):
        return _series_forcefillna(df, preserved_int)
    elif isinstance(df, pd.DataFrame):
        col_list = df.columns.tolist()
        if fill_list and exempt_list:
            raise ValueError('fill_list and exempt_list can not both be used.')
        if fill_list:
            for col in fill_list:
                df[col] = _series_forcefillna(df[col], col in preserved_int_list)
        elif exempt_list:
            for col in col_list:
                if col not in exempt_list:
                    df[col] = _series_forcefillna(df[col], col in preserved_int_list)
        else:#Use the individual lists
            fill_list = []
            if fill_float:
                fill_list.extend(get_cols_by_dtype(df, float))
            if fill_int:
                fill_list.extend(get_cols_by_dtype(df, int))
            if fill_cat:
                fill_list.extend(get_cols_by_dtype(df, 'category'))
            if fill_obj:
                fill_list.extend(get_cols_by_dtype(df, 'object'))
            for col in fill_list:
                df[col] = _series_forcefillna(df[col], col in preserved_int_list)
        return df
    else:
        raise ValueError(f'Type {type(df)} is not supported.')
