from toolz import pipe as p


def series_filter(fn):
    return lambda arr: arr[arr.map(fn)]

def filter(fn):
    def ret(df):
        ixs = fn(df)
        return df.loc[ixs]
    
    return ret

def select(col):
    return lambda df: df[col]

def select_ixs(ixs):
    return lambda df: df.iloc[ixs]

def apply(fn):
    if type(fn) is str:
        return lambda df: getattr(df, fn)()
    return lambda df: fn(df)

def groupby(col):
    return lambda df: df.groupby(by = col)


def mutate(col, fn):
    def ret(df):
        df2 = df.copy()
        df2[col] = fn(df)
        return df2
    
    return ret


def summarize(named_aggs):
    return lambda _: _.agg(**named_aggs)