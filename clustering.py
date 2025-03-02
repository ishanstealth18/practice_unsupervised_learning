import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler



def import_file():
    input_source_file = pd.read_csv("Customer_Transactions_Behavior_Analysis.csv")
    return input_source_file


def remove_columns(input_file):
    # remove Customer_ID, Spending_Score,Last_Purchase_Amount, Days_Since_Last_Purchase, Customer_Satisfaction,
    # Referral_Score, Customer_Review
    input_file.drop(['Customer_ID', 'Spending_Score', 'Last_Purchase_Amount', 'Days_Since_Last_Purchase',
                     'Customer_Satisfaction', 'Referral_Score', 'Customer_Review', 'Total_Purchases',
                     'Subscription_Type'], axis=1, inplace=True)

    return input_file


def check_replace_null_values(input_file):
    # check null values
    print("Check null values :\n", input_file.isnull().sum())
    # remove records with null values in 'Annual_Income' and 'Subscription_Type' column
    input_file.dropna(axis=0, inplace=True)
    print("After null values removal: \n", input_file.isnull().sum())

    return input_file


def pca_analysis(input_file):
    pca_model = PCA()
    pca_model.fit()

    return None


def text_to_number(input_file):
    column_names = input_file.select_dtypes(include='object').columns
    print(column_names)
    # do one hot encoding
    for column in column_names:
        dummy_columns = pd.get_dummies(input_file[column], drop_first=True, prefix=column, dtype='int64')
        input_file = pd.concat([dummy_columns, input_file], axis=1)
        input_file.drop([column], axis=1, inplace=True)

    print(input_file.info())
    return input_file


def visualize_data(input_file):
    column_names = list(input_file.columns)
    # Creating subplots
    for c in range(len(column_names)):
        plt.subplot(4, 5, c+1)
        data = column_names[c]
        plt.boxplot(input_file[data])
        plt.title(column_names[c])
        plt.plot()
    plt.tight_layout()
    plt.show()



def check_initial_data_condition(input_file):
    # check basic info
    print(input_file.info())

    # remove unwanted columns
    remove_columns(input_file)

    # check and remove null values
    null_removed_input_file = check_replace_null_values(input_file)

    # convert text to numeric values
    all_numeric_input_file = text_to_number(null_removed_input_file)

    # visualize data to check outliers
    visualize_data(all_numeric_input_file)

    return all_numeric_input_file


def normalize_data(input_file):
    scaler = StandardScaler()
    input_file['Age'] = scaler.fit_transform(input_file['Age'])
    input_file['Annual_Income'] = scaler.fit_transform(input_file['Annual_Income'])

    print(input_file.head(10))
    return None


def cluster_main_func():
    input_file = import_file()

    # check and process initial data
    input_file_after_initial_check = check_initial_data_condition(input_file)

    # normalize data for columns 'Age' and 'Annual_Income'
    normalize_data(input_file_after_initial_check)

    # decide which features are important, do PCA analysis
    # pca_analysis(input_file)


cluster_main_func()