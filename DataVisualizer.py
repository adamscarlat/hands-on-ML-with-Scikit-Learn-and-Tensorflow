import matplotlib.pyplot as plt
from DataHandler import load_data, HOUSING_PATH
from pandas.plotting import scatter_matrix
import os, time

def scatter_plot_by_column(data, x_column, y_column, radius_column, heat_column):
    '''
    scatter plot where each data point is located at (x,y) has a radius equal to radius_column
    and a heat map according to heat_column.
    '''
    data.plot(
        kind="scatter", 
        x=x_column, y=y_column, 
        s=data[radius_column]/100, 
        label=radius_column,
        c=heat_column, 
        cmap=plt.get_cmap("jet"), colorbar=True, alpha=0.4
     )
    save_fig('scatter_plot_by_column')

def scatter_matrix_for_attributes(data, attributes, figsize=(12, 8)):
    '''
    scatter matrix for given attributes in data. number of graphs 
    will be equal to attributs squared
    '''
    scatter_matrix(data[attributes], figsize=(12, 8))
    save_fig('scatter_matrix_for_attributes')

def show_correlation_with_column(data, column_name):
    '''
    print numerical pearson coefficient (linear correlation) between column_name
    to the rest of the columns in the data.
    '''
    corr_matrix = data.corr()

    print ('Correlation with {column_name}'.format(column_name = column_name))
    print (corr_matrix[column_name].sort_values(ascending=False))

def save_fig(fig_id, tight_layout=True):
    '''
    save figue to folder. file name will include current timestamp
    '''
    fig_id += '_' + str(time.time())
    path = os.path.join('.', "images", fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)

if __name__ == '__main__':
    housing = load_data(HOUSING_PATH, 'housing.csv')
    print (housing.head())
    print (housing.describe())
    
    scatter_plot_by_column(housing, 'longitude', 'latitude', 'population', 'median_house_value')
    
    attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
    scatter_matrix_for_attributes(housing, attributes)

    #example for feature mapping - better correlation with the target value
    housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
    housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
    housing["population_per_household"] = housing["population"]/housing["households"]

    #see correlation
    show_correlation_with_column(housing, 'median_house_value')