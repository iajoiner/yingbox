import pandas
import copy
#Obtain a list of columns with at least one na
#treats_empty_inf_as_na is True if empty strings and numpy.inf are treated as na
def find_cols_with_na(df, treats_empty_inf_as_na = False):
    if treats_empty_inf_as_na:
        pandas.options.mode.use_inf_as_na = True
    return df.columns[df.isna().any()].tolist()
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
def _series_forcefillna_choice(s, col_name, preserve_int = False, add_col = False):
    new_s = _series_forcefillna(s, preserve_int)
    if not add_col:
        return new_s
    exists_s = s.notna().apply(int)
    return pandas.DataFrame({col_name: new_s, col_name + '_exists': exists_s})
#There are three ways to use this function.
#1.Use fill_list to determine which cols to fill
#2.Use exempt_list to determine which cols not to fill
#3.Use fill_<type> to fill cols by type
def forcefillna(obj, preserved_int = False, preserved_int_list = [], fill_list = None, exempt_list = None, fill_float = True, fill_int = True, fill_cat = True, fill_obj = True, add_col = False, mandatory_exist_add_list = None):
    if isinstance(obj, pandas.Series):
        s = copy.deepcopy(obj)
        s_name = s.name
        if not s_name:#What if a series doesn't have a name?
            s_name = '0'
        if not s.isna().any():#Nothing to fill
            return s
        if add_col:
            return _series_forcefillna_choice(s, s_name, preserved_int, True)
        else:
            return _series_forcefillna(s, preserved_int)
    elif isinstance(obj, pandas.DataFrame):
        df = copy.deepcopy(obj)
        col_list = df.columns.tolist()
        if fill_list and exempt_list:
            raise ValueError('fill_list and exempt_list can not both be used.')
        if fill_list:
            for col in fill_list and df[col].isna().any():
                if add_col:
                    df[[col, col + '_exists']] = _series_forcefillna_choice(df[col], col, col in preserved_int_list, True)
                else:
                    df[col] = _series_forcefillna(df[col], col in preserved_int_list)
        elif exempt_list:
            for col in col_list:
                if col not in exempt_list and df[col].isna().any():
                    if add_col:
                        df[[col, col + '_exists']] = _series_forcefillna_choice(df[col], col, col in preserved_int_list, True)
                    else:
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
                if df[col].isna().any():
                    if add_col:
                        df[[col, col + '_exists']] = _series_forcefillna_choice(df[col], col, col in preserved_int_list, True)
                    else:
                        df[col] = _series_forcefillna(df[col], col in preserved_int_list)
        if mandatory_exist_add_list:#Add more lists!
            #print('Meow')
            mandatory_exist_add_list = [col+'_exists' for col in mandatory_exist_add_list]
            added_list = [col for col in mandatory_exist_add_list if col not in df.columns]
            #print(added_list)
            for col in added_list:
                df[col] = 1
        return df
    else:
        raise ValueError(f'Type {type(obj)} is not supported.')
def forcefillna_pair(df_train, df_test, preserved_int = False, preserved_int_list = [], fill_list = None, exempt_list = [], fill_float = True, fill_int = True, fill_cat = True, fill_obj = True, add_col = False, y_col = 'y'):
    train_exempt_list = exempt_list
    if isinstance(exempt_list, list):
        train_exempt_list.append(y_col)#Add y_col to train_exempt_list
    if add_col:
        full_extra_cols_list = list(set(find_cols_with_na(df_train)).union(set(find_cols_with_na(df_test))))
        #print(full_extra_cols_list)
        df_train_filled = forcefillna(df_train, preserved_int, preserved_int_list, fill_list, train_exempt_list, fill_float, fill_int, fill_cat, fill_obj, add_col, full_extra_cols_list)
        df_test_filled = forcefillna(df_test, preserved_int, preserved_int_list, fill_list, exempt_list, fill_float, fill_int, fill_cat, fill_obj, add_col, full_extra_cols_list)
    else:
        df_train_filled = forcefillna(df_train, preserved_int, preserved_int_list, fill_list, train_exempt_list, fill_float, fill_int, fill_cat, fill_obj, add_col)
        df_test_filled = forcefillna(df_test, preserved_int, preserved_int_list, fill_list, exempt_list, fill_float, fill_int, fill_cat, fill_obj, add_col)
    return df_train_filled, df_test_filled
