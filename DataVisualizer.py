import matplotlib.pyplot as plt
from DataHandler import load_data, HOUSING_PATH
from pandas.plotting import scatter_matrix

def scatter_plot_by_column(data, x_column, y_column, radius_column, column_name):
    data.plot(
        kind="scatter", 
        x=x_column, y=y_column, 
        s=data[radius_column]/100, 
        label=radius_column,
        c=column_name, 
        cmap=plt.get_cmap("jet"), colorbar=True, alpha=0.4
     )
    plt.show()

def show_correlation_with_column(data, column_name):
    corr_matrix = data.corr()

    print ('Correlation with {column_name}'.format(column_name = column_name))
    print (corr_matrix[column_name].sort_values(ascending=False))

if __name__ == '__main__':
    housing = load_data(HOUSING_PATH, 'housing.csv')
    print (housing.head())
    print (housing.describe())
    #scatter_plot_by_column(housing, 'longitude', 'latitude', 'population', 'median_house_value')
    # attributes = ["median_house_value", "median_income", "total_rooms",
    #               "housing_median_age"]
    # scatter_matrix(housing[attributes], figsize=(12, 8))
    # plt.show()

    #feature mapping - better correlation with the target value
    housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
    housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
    housing["population_per_household"] = housing["population"]/housing["households"]

    #see correlation
    show_correlation_with_column(housing, 'median_house_value')