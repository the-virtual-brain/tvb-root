import pandas as pd


limited_columns = ["make","price","city mpg","highway mpg","horsepower",
                   "weight","riskiness","losses"]


def get_make_names(data):
    return sorted(list(pd.Series(data["make"]).unique()))


def get_make_labels(data):
    return [x.title() for x in get_make_names(data)]


def get_make_id_map(data):
    return {x:i for (i,x) in enumerate(get_make_names(data))}


def get_make_ids(data):
    return sorted(get_make_id_map(data).values())


def get_raw_data():
    data_file = "../data/autos-clean.csv"
    return pd.read_csv(data_file)


def get_make_data(make, pddata):
    return pddata[(pddata["make"] == make)]


def get_make_counts(pddata, lower_bound=0):
    counts = []
    filtered_makes = []
    for make in get_all_auto_makes():
        data = get_make_data(make, pddata)
        count = len(data.index)
        if count >= lower_bound:
            filtered_makes.append(make)
            counts.append(count)
    return (filtered_makes, list(zip(filtered_makes, counts)))


def get_limited_data(cols=None, lower_bound=None):
    if not cols:
        cols = limited_columns
    data = get_raw_data()[cols]
    if lower_bound:
        (makes, _) = get_make_counts(data, lower_bound)
        data = data[data["make"].isin(makes)]
    return data


def norm_column(col_name, pddata, inverted=False):
    pddata[col_name] -= pddata[col_name].min()
    pddata[col_name] /= pddata[col_name].max()
    if inverted:
        pddata[col_name] = 1 - pddata[col_name]


def norm_columns(col_names, pddata):
    for col in col_names:
        norm_column(col, pddata)


def invert_norm_columns(col_names, pddata):
    for col in col_names:
        norm_column(col, pddata, inverted=True)


def get_all_auto_makes():
     return pd.Series(get_raw_data()["make"]).unique()


def get_numeric_data(pddata):
    return pddata.replace({"make": get_make_id_map(pddata)})
