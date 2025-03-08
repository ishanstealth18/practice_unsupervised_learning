import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn


# function to import file
def import_file():
    input_source_file = pd.read_csv("Customer_Transactions_Behavior_Analysis.csv")
    return input_source_file


# remove unnecessary columns
def remove_columns(input_file):
    # remove Customer_ID, Spending_Score,Last_Purchase_Amount, Days_Since_Last_Purchase, Customer_Satisfaction,
    # Referral_Score, Customer_Review
    input_file.drop(['Customer_ID', 'Spending_Score', 'Last_Purchase_Amount', 'Days_Since_Last_Purchase',
                     'Customer_Satisfaction', 'Referral_Score', 'Customer_Review', 'Total_Purchases',
                     'Subscription_Type'], axis=1, inplace=True)

    return input_file


# function to check and replace null values
def check_replace_null_values(input_file):
    # check null values
    #print("Check null values :\n", input_file.isnull().sum())
    # remove records with null values in 'Annual_Income' and 'Subscription_Type' column
    input_file.dropna(axis=0, inplace=True)
    #print("After null values removal: \n", input_file.isnull().sum())

    return input_file


# function to convert text to number using dummies method
def text_to_number(input_file):
    column_names = input_file.select_dtypes(include='object').columns
    print(column_names)
    # do one hot encoding
    for column in column_names:
        dummy_columns = pd.get_dummies(input_file[column], drop_first=True, prefix=column, dtype='int64')
        input_file = pd.concat([dummy_columns, input_file], axis=1)
        input_file.drop([column], axis=1, inplace=True)

    #print(input_file.info())
    return input_file


# function to visualize data for outliers
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


# function to check initial data conditions
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


# function to normalize data
def normalize_data(feature_data):
    scaler = StandardScaler()
    scaler.fit(feature_data)
    feature_data = scaler.transform(feature_data)
    return feature_data


# function to do PCA (Principal Component Analysis)
def pca_analysis(input_file):
    # selected n_components = 2 to make visualization easy
    pca_model = PCA(n_components=2)
    pca_model.fit(input_file)
    transformed_val = pca_model.transform(input_file)
    print("variance ratio: ", pca_model.explained_variance_ratio_)

    # create a plot to check the variance ratio (optional)
    pca_features = range(pca_model.n_components_)
    plt.bar(pca_features, pca_model.explained_variance_ratio_)
    plt.xticks(pca_features)
    plt.xlabel('PCA features')
    plt.ylabel('Variance Ratio')
    plt.show()

    return transformed_val


# function to check k values for KMeans
def select_k_values(data):
    k_inertia_values = []
    # loop through a range o k values and calculate inertia
    for cluster in range(1, 20):
        model = KMeans(n_clusters=cluster)
        model.fit(data)
        k_inertia_values.append(model.inertia_)

    # plot the k values plot
    plt.plot(range(1, 20), k_inertia_values)
    plt.xlabel('K values')
    plt.ylabel('Inertia values')
    plt.show()


# function to apply K means
def apply_kmeans(samples):
    # check k values for different range
    select_k_values(samples)
    # from the k inertia graphs, k=5 is ideal value
    model = KMeans(n_clusters=5)
    model.fit(samples)
    pred_val = model.predict(samples)

    return pred_val


# main function
def cluster_main_func():
    input_file = import_file()

    # check and process initial data
    input_file_after_initial_check = check_initial_data_condition(input_file)

    # divide features and category
    x = input_file_after_initial_check.drop(['Fraudulent_Activity'], axis=1).values
    y = input_file_after_initial_check['Fraudulent_Activity'].values

    # normalize data for columns 'Age' and 'Annual_Income'
    normalized_input_file = normalize_data(x)

    # decide which features are important, do PCA analysis
    pca_transformed_val = pca_analysis(normalized_input_file)

    # apply k means
    clusters = apply_kmeans(pca_transformed_val)

    # do cross tabulation to check count of fraudulent entries for each clusters
    ct = pd.crosstab(clusters, y)
    print(ct)

    # create new df
    new_df = pd.DataFrame({'PC1': pca_transformed_val[:, 0], 'PC2': pca_transformed_val[:, 1], 'Clusters': clusters})
    print(new_df.shape)

    # plot PC1 vs PC2 with cluster values
    seaborn.scatterplot(x='PC1', y='PC2', data=new_df, hue='Clusters')
    plt.show()


cluster_main_func()