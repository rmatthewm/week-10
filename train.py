import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression 
from sklearn.tree import DecisionTreeRegressor

roast_list = []

def numerify_roast(roast):
    """ Return an integer for any given roast.

    Args:
        roast (str): a type of coffee roast

    Returns:
        int: a unique number corresponding to the given roast
    """
    global roast_list

    # We can store the roasts we have seen in a list and use the index 
    # as a unique identifier
    if not roast in roast_list:
        roast_list.append(roast)

    return roast_list.index(roast)
    

def main():
    """ Trains two scikit models, pickles them, and saves them."""

    # Load in the coffee dataset
    df_coffee = pd.read_csv('https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv')

    # Get the data to fit the model to
    X1 = df_coffee[['100g_USD']]
    y = df_coffee['rating']

    # Create the linear regression model
    lr_model = LinearRegression()

    # Fit the model to the data
    lr_model.fit(X1, y)

    # Save the linear regression model
    with open('model_1.pickle', 'wb') as file:
        pickle.dump(lr_model, file)


    # Create a numerical category for the roast so we can 
    # use it to train the model
    df_coffee['roast_cat'] = df_coffee['roast'].apply(numerify_roast)


    # Get the data to fit the decision tree model to
    X2 = df_coffee[['100g_USD', 'roast_cat']]

    # Create the model
    dtr_model = DecisionTreeRegressor()

    # Fit the model to the data
    dtr_model.fit(X2, y)

    # Save the decision tree model
    with open('model_2.pickle', 'wb') as file:
        pickle.dump(dtr_model, file)


if __name__ == '__main__':
    main()
