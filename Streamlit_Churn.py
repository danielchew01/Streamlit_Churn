#import required libraries
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu
from imblearn.over_sampling import SMOTE
from PIL import Image

import warnings
warnings.filterwarnings('ignore')

st.title('E-Commerce Churn Prediction by Supervised Learning Machine Algorithm')

description = pd.read_excel('E Commerce Dataset.xlsx', sheet_name='Data Dict', header=1, usecols=[1,2,3])
data = pd.read_excel('E Commerce Dataset.xlsx', sheet_name='E Comm')
description
data

with st.sidebar:
    selected = option_menu(
        menu_title="Process of Project",
        options=["Exploratory Data Analysis (EDA)", "Data Preprocessing", "Model Evaluation"],
    )

if selected == "Exploratory Data Analysis (EDA)":
    st.header('Exploratory Data Analysis (EDA)')
    st.write('As we can see in the aforementioned cells, the data includes information for 5630 customers with 20 features.')
    st.write('Another thing to keep in mind is that some of the columns lack values, which will be handled when developing the model later.')

    st.subheader('Distribution of the Customer Churn according to dataset')
    figcc = plt.figure(figsize = (12, 6))
    sns.countplot(x='Churn', palette='viridis', data=data)
    plt.title("Distribution of Churn")
    st.pyplot(figcc)
    st.write('According to the graph, we can see that majority of the customers will not churn with the number of approximately 4900.')

    st.subheader('Distribution of the Tenure of the customers on the platform')
    figtc = plt.figure(figsize = (20, 6))
    sns.countplot(x='Tenure', palette='viridis', data=data)
    plt.title("Distribution of Tenure of the Customers on the platform")
    st.pyplot(figtc)
    st.write('According to the graph, we can see that majority of the users are less job tenure period.')

    st.subheader('Distribution of the Preferred Login Device')
    figpl = plt.figure(figsize = (12, 6))
    sns.countplot(x='PreferredLoginDevice', palette='viridis', data=data)
    plt.title("Distribution of Preferred Login Device")
    st.pyplot(figpl)
    st.write('Based on the graph, we can see that the majority of the customer will use mobil phone to login and follow up by Computer and Phone.')

    st.subheader('Distribution of the City Tier of the customers on the platform')
    figct = plt.figure(figsize = (12, 6))
    sns.countplot(x='CityTier', palette='viridis', data=data)
    plt.title("Distribution of City Tier")
    st.pyplot(figct)
    st.write('According to the graph, we can pbserve that the majority of the customers live in the City Tier 1, follow up by City Tier 3 and 2.')

    st.subheader('Distribution of distance of Warehouse to customers home')
    figdw = plt.figure(figsize = (20, 6))
    sns.countplot(x='WarehouseToHome', palette='viridis', data=data)
    plt.title("Distribution of distance of Warehouse to customers home")
    st.pyplot(figdw)
    st.write('According to the graph above, we can see that majority of the customers are near the Warehouse.')

    st.subheader('Distribution of Customer Preferred Payment Mode')
    figpm = plt.figure(figsize = (12, 6))
    sns.countplot(x='PreferredPaymentMode', palette='viridis', data=data)
    plt.title("Distribution of Customer Preferred Payment Mode")
    st.pyplot(figpm)
    st.write('Based on the graph above, we can see that many of the customer more prefer using Debit Card to proceed the payment.')

    st.subheader('Distribution of Gender of Customer in the platform')
    figgc = plt.figure(figsize = (12, 6))
    sns.countplot(x='Gender', palette='viridis', data=data)
    plt.title("Distribution of Gender of Customer in the platform")
    st.pyplot(figgc)
    st.write('We can see that in the platform we will have more male customers than the female customers.')

    st.subheader('Distribution of Hours spent on the app by the customers')
    fighs = plt.figure(figsize = (12, 6))
    axx = sns.countplot(x='HourSpendOnApp', data=data)
    for a in axx.patches:
        axx.annotate(format((a.get_height()/5630)*100,'.2f'), (a.get_x() + a.get_width()/2., a.get_height()),\
                    ha='center',va='center',size=12,xytext=(0, 6),textcoords='offset points')
    plt.title("Distribution of hours spent on the app by the customers")
    st.pyplot(fighs)
    st.write('We can see that majority of the customers spent 3 hours on our app.')

    st.subheader('Distribution of Number of Device that registered for each customer')
    fignd = plt.figure(figsize = (12, 6))
    sns.countplot(x='NumberOfDeviceRegistered', palette='viridis', data=data)
    plt.title("Distribution of Number of Device that registered for each customer")
    st.pyplot(fignd)
    st.write('Based on the graph above, we can see that majority of the customers have 3 to 4 devices registered for using the app.')

    st.subheader('Distribution of Customer Preferred Order Category')
    figoc = plt.figure(figsize = (12, 6))
    sns.countplot(x='PreferedOrderCat', palette='viridis', data=data)
    plt.title("Distribution of Customer Preferred Order Category")
    st.pyplot(figoc)
    st.write('From the graph above, we can see that the majority of the customers preferred to the Laptop and Accessory category more than other category.')

    st.subheader('Distribution of Satisfaction Score from each customer')
    figss = plt.figure(figsize = (12, 6))
    sns.countplot(x='SatisfactionScore', palette='viridis', data=data)
    plt.title("Distribution of Satisfaction Score from each customer")
    st.pyplot(figss)
    st.write('From the graph above, we can see that majority of the customers are feel moderate for the performance of the app.')

    st.subheader('Distribution of Marital Status for each customer')
    figms = plt.figure(figsize = (12, 6))
    sns.countplot(x='MaritalStatus', palette='viridis', data=data)
    plt.title("Distribution of Marital Status for each customer")
    st.pyplot(figms)
    st.write('According to the graph, we can observe that majority of our customers are married, follow up by single and divorced.')

    st.subheader('Distribution of Number of address they have for each customer')
    figna = plt.figure(figsize = (12, 6))
    sns.countplot(x='NumberOfAddress', palette='viridis', data=data)
    plt.title("Distribution of Number of address they have for each customer")
    st.pyplot(figna)
    st.write('We can observe that majority of the customers will 2 to 3 address from the graph above.')

    st.subheader('Distribution of Complain from each customer')
    figccust = plt.figure(figsize = (12, 6))
    sns.countplot(x='Complain', palette='viridis', data=data)
    plt.title("Distribution of Complain from each customer")
    st.pyplot(figccust)
    st.write('From the graph above, we can see that majority of the customers are not complain to our app which they didnt have much bad experience using our app.')

    st.subheader('Distribution of Percentage increase in customer orders')
    figpi = plt.figure(figsize = (12, 6))
    sns.countplot(x='OrderAmountHikeFromlastYear', palette='viridis', data=data)
    plt.title("Distribution of Percentage increase in customer orders")
    st.pyplot(figpi)
    st.write('According to the graph above, we can see that majority of increase orders in approximately 14% in last year.')

    st.subheader('Distribution of Number of Coupun Used by the customers')
    fignc = plt.figure(figsize = (12, 6))
    sns.countplot(x='CouponUsed', palette='viridis', data=data)
    plt.title("Distribution of Number of Coupun Used by the customers")
    st.pyplot(fignc)
    st.write('From the graph above, we can see that majority of the customer are less using our coupon.')

    st.subheader('Distribution of Order Count of customers')
    figocc = plt.figure(figsize = (12, 6))
    sns.countplot(x='OrderCount', palette='viridis', data=data)
    plt.title("Distribution of Number of customer orders")
    st.pyplot(figocc)
    st.write('According to the graph above, we can see that the majority of number order count of customers is at 2 and follow up by 1.')

    st.subheader('Distribution of Recency of the customers')
    st.write('Recency means that the days since the last order of each customer.')
    figrc = plt.figure(figsize = (12, 6))
    sns.countplot(x='DaySinceLastOrder', palette='viridis', data=data)
    plt.title("Distribution of Recency of customer orders")
    st.pyplot(figrc)
    st.write('From the graph above, we can see that majority of the customers have order something recently.')

    st.subheader('Distribution of Amount returned for money spent by customers')
    figcb = plt.figure(figsize = (12, 10))
    sns.histplot(x='CashbackAmount_RM', palette='viridis', data=data)
    plt.title("Distribution of Cashback for customers")
    st.pyplot(figcb)
    st.write('From the graph above, we can see that majority of the customers have use the cashback function to get back approximately RM160.')

    st.subheader('Distribution Satisfaction score for churned and retained customers')
    figsscr = plt.figure(figsize = (12, 6))
    sns.countplot(x='SatisfactionScore', hue='Churn', palette='viridis', data=data)
    plt.title("Distribution of Satisfaction Score for Churned and Retained customers")
    st.pyplot(figsscr)
    st.write('As the graph show that the although the satisfaction score is moderate but the churn rate it have is the highest among all the satisfaction score.')

    st.subheader('Distribution of Gender for churned and retained customers')
    figgcr = plt.figure(figsize = (12, 6))
    sns.countplot(x='Gender', hue='Churn', palette='viridis', data=data)
    plt.title("Distribution of Gender for Churned and Retained customers")
    st.pyplot(figgcr)
    st.write('As the graph above, we can see that male customers will have higher precentage to churn than female customers.')

    st.subheader('Distribution of marital status for churned and retained customers')
    figmscr = plt.figure(figsize = (12, 6))
    sns.countplot(x='MaritalStatus', hue='Churn', palette='viridis', data=data)
    plt.title("Distribution of marital status for churned and retained customers")
    st.pyplot(figmscr)
    st.write('From the graph above, we can see that the customers that are single have more percentage to churn other than the other marital status.')

    st.subheader('Distribution of complain for churned and retained customers')
    figccr = plt.figure(figsize = (12, 6))
    sns.countplot(x='Complain', hue='Churn', palette='viridis', data=data)
    plt.title("Distribution of complain for churned and retained customers")
    st.pyplot(figccr)
    st.write('Based on the graph above, we can the customer that have complain for the experience will have higher chance to churn.')

    st.subheader('Relationship between the Tenure and Churn rate')
    figrtc = plt.figure(figsize = (12, 6))
    sns.scatterplot(x=data['Tenure'],y=data.groupby('Tenure').Churn.mean())
    plt.title("Relationship between Tenure and Churn rate")
    st.pyplot(figrtc)
    st.write('Based on the graph above, we can see that the relationship betweeen tenure for each customer and churn rate have no correlation relationship.')

    st.subheader('Relationship between the Order Count and Churn rate')
    figroc = plt.figure(figsize = (12, 6))
    sns.scatterplot(x=data['OrderCount'],y=data.groupby('OrderCount').Churn.mean())
    plt.title("Relationship between OrderCount and Churn rate")
    st.pyplot(figroc)
    st.write('Based on the graph above, we can see that the relationship betweeen order count for each customer and churn rate have no correlation relationship.')

    st.subheader('Relationship between the Coupon Used and Churn rate')
    figrcc = plt.figure(figsize = (12, 6))
    sns.scatterplot(x=data['CouponUsed'],y=data.groupby('CouponUsed').Churn.mean())
    plt.title("Relationship between CouponUsed and Churn rate")
    st.pyplot(figrcc)
    st.write('Based on the graph above, we can see that the relationship betweeen coupon used for each customer and churn rate have no correlation relationship.')

    st.subheader('Heatmaps for all the variable in the ECommerceDataset')
    fighav = plt.figure(figsize = (12, 6))
    sns.heatmap(data.drop('CustomerID',axis=1).corr(), annot=True)
    plt.title("Correlation Matrix for the Customer Dataset")
    st.pyplot(fighav)
    st.write('From the Heatmap above, we can see that just only have few variables have roughly strong and moderate stong relationship between each other.')
    st.write('For an example:')
    st.write('1. The CoupunUsed have roughly strong relationship with OrderCount with score of 0.75.')
    st.write('2. DaySinceLastOrder have moderate relationship with OrderCount with score of 0.5.')
    st.write('3. Tenure have moderate relationship with CashbackAmount with score of 0.48.')
    st.write('4. CouponUsed have rough moderate relationship with DaySinceLastOrder with score of 0.36.')
    st.write('5. OrderCount have rough moderate relationship with CashbackAmount(RM) with score 0.36.')
    st.write('6. DaySinceLastOrder have rough moderate relationship with CashbackAmount(RM) with score 0.35.')

if selected == "Data Preprocessing":
    st.header('Data Preprocessing')
    st.subheader('Handling the Missing Value')
    df = data.copy()
    df.drop(['CustomerID'],axis=1, inplace=True)
    st.write(df.isna().sum())
    st.dataframe(df)

    for i in df.columns:
        if data[i].isna().sum() > 0:
            st.write(i)
            st.write('- the total null values are ', df[i].isna().sum())
            st.write('- the datatype is', df[i].dtypes)

    st.write('In total there are 1856 missing value, meaning that each of the missing value is on a different row.')
    st.write('So if we drop all the rows with missing values we would be dropping 1856 rows that is 32.97 percent of the dataset and will cause a data loss.')
    st.write('And above are the variables that have missing values in it so we have replace numerical missing value with mean for each variables that show above.')

    st.subheader('Replace the missing value with mean')
    for i in df.columns:
        if df[i].isnull().sum() > 0:
            df[i].fillna(df[i].mean(),inplace=True)

    left_column, right_column = st.columns(2)
    with left_column:
        st.write(data.isna().sum()) 
        st.caption('Before Missing Value Treatment')
    with right_column:
        fill_df = df
        st.write(fill_df.isna().sum())
        st.caption('After Missing Value Treatment')

    st.subheader('Outliers Treatment')
    for_outlier_df = fill_df.drop(['Churn'],axis=1)
    churn_df = fill_df['Churn']

    dChurn_df = pd.DataFrame(churn_df)

    st.subheader('Boxplot of the Dataset for checking the outliers')
    boxplt = plt.figure(figsize=(60,10))
    sns.boxplot(data=for_outlier_df)
    plt.title('The boxplot to study outliers')
    plt.xlabel('Variables that predict the customer churn')
    plt.ylabel('Values')
    st.pyplot(boxplt)
    st.caption('Boxplot of Dataset before Outlier Treatment')
    st.write('We can see that ther are quite a lot outliers in almost all of the variables. Lets treat those outliers.')

    def remove_outlier(col):
        sorted(col)
        Q1,Q3=np.percentile(col,[25,75])
        IQR=Q3-Q1
        lr= Q1-(1.5 * IQR)
        ur= Q3+(1.5 * IQR)
        return lr, ur

    for column in for_outlier_df.columns:
        if for_outlier_df[column].dtype != 'object': 
            lr,ur=remove_outlier(for_outlier_df[column])
            for_outlier_df[column]=np.where(for_outlier_df[column]>ur,ur,for_outlier_df[column])
            for_outlier_df[column]=np.where(for_outlier_df[column]<lr,lr,for_outlier_df[column])

    ot_boxplt = plt.figure(figsize=(50,10))
    sns.boxplot(data=for_outlier_df)
    plt.title('The boxplot to study outliers')
    plt.xlabel('Variables that predict the customer churn')
    plt.ylabel('Values')
    st.pyplot(ot_boxplt)
    st.caption('Boxplot of Dataset after Outlier Treatment')

    st.subheader('Data Imbalance Treatment (SMOTE)')
    numerical_df = for_outlier_df.select_dtypes(exclude=['object'])
    object_df = for_outlier_df.select_dtypes(include=['object'])

    frames = [dChurn_df, numerical_df]
    cleaned_df = pd.concat(frames, axis = 1)
    
    ax = plt.figure(figsize=(10,6))
    sns.countplot(x='Churn', data=cleaned_df)
    st.pyplot(ax)
    st.write('From the graph above we can see that The data is skewed because there are more kept customers than churned ones the ratio of retained to churned customers is roughly 5 to 1 and since the churn rate is quite low which makes the data imbalanced.')
    st.write('So we need to do oversampling using SMOTE in order to make it balance.')
    
    X = cleaned_df.loc[:, cleaned_df.columns != 'Churn']
    y = cleaned_df.loc[:, cleaned_df.columns == 'Churn']

    os = SMOTE(random_state = 42, k_neighbors = 2)
    columns = X.columns

    os_data_X, os_data_y = os.fit_resample(X, y)

    os_data_X = pd.DataFrame(data = os_data_X, columns = columns )
    os_data_y = pd.DataFrame(data = os_data_y, columns = ['Churn'])
    st.write("Length of oversampled data is", len(os_data_X))

    ax = plt.figure(figsize = (10,6))
    plt.title("After Oversampling with SMOTE")
    sns.countplot(x = 'Churn', data = os_data_y)
    st.pyplot(ax)

    frames = [os_data_X, os_data_y]
    df_smoted = pd.concat(frames, axis = 1)

    frames = [df_smoted, object_df]
    df_smoted = pd.concat(frames, axis = 1)

    st.subheader('Feature Selection (Boruta)')
    st.subheader('Ranking of features in Top 10 and Bottom 10')

    image_fs_top10 = Image.open('feature_selection_Top10.PNG')
    image_fs_bottom10 = Image.open('feature_selection_Bottom10.PNG')

    left_column, right_column = st.columns(2)
    with left_column:
        st.image(image_fs_top10, caption='Ranking of features in Top 10') 
    with right_column:
        st.image(image_fs_bottom10, caption='Ranking of features in Bottom 10')

    st.subheader('Choose the optimal features')
    image_drop_features = Image.open('decision_drop_features.PNG')
    st.image(image_drop_features, caption='Ranking of Each Feartures and condition of keep or drop') 
    st.write('Above shown that, there are some features we can drop it like for an example: we can drop away the MaritalStatus_Divorced, PreferredPaymentMode_CC and OrderAmountHikeFromlastYear which have the rank 2, 3 and 4.')

    col_list = [col for col in df_smoted.columns.tolist() if 
            df_smoted[col].dtype.name == "object"]

    df_oh = df_smoted[col_list]
    df_smoted = df_smoted.drop(col_list,1)
    df_oh = pd.get_dummies(df_oh)

    final_df = pd.concat([df_smoted, df_oh], axis=1)

    X = final_df.drop("Churn", 1)

    selected_features_df = X.drop(columns=["MaritalStatus_Divorced", "PreferredPaymentMode_CC", "OrderAmountHikeFromlastYear"])
    
    frames = [selected_features_df, y]
    selectfeatures_Churn_df =  pd.concat(frames, axis = 1)

    st.write(selectfeatures_Churn_df)
    st.caption('Dataset after drop the "MaritalStatus_Divorced", "PreferredPaymentMode_CC", "OrderAmountHikeFromlastYear" features')


if selected == "Model Evaluation":
    st.header('Model Evaluation')
    st.subheader('Decision Tree Model')
    
    st.subheader('With features that selected using Boruta')
    st.write('Accuracy on training set: 0.855')
    st.write('Accuracy on test set: 0.848')
    st.write('According to the score of accuracy of both training set and test set, the difference score between them are not far so which mean the dataset is good and didnt need to do data imbalance treatment again')

    st.subheader('With all features')
    st.write('Accuracy on training set: 0.855')
    st.write('Accuracy on test set: 0.848')
    st.write('According to the score of accuracy of both training set and test set, the difference score between them are not far so which mean the dataset is good and didnt need to do data imbalance treatment again')

    st.subheader('Comparison of the result of Decision Tree with different features')
    image_dt_selectf = Image.open('DT_cm_fs.PNG')
    image_dt_allf = Image.open('DT_cm_allfs.PNG')

    left_column, right_column = st.columns(2)
    with left_column:
        st.image(image_dt_selectf, caption='Confusion Matrix for Decision Tree with selected features') 
    with right_column:
        st.image(image_dt_allf, caption='Confusion Matrix for Decision Tree with all features')

    st.write('According to the confusion matrix above we can observe that the impact for different features and using all features have no any differences. We can taking any of the decision tree with different features to do prediction.')

    st.subheader('Using selected Decision Tree model to do prediction')
    image_AUC_dt= Image.open('AUC_dt.PNG')
    image_PRC_dt = Image.open('PRC_dt.PNG')

    left_column, right_column = st.columns(2)
    with left_column:
        st.image(image_AUC_dt, caption='Receiver Operating Characteistic (ROC) Curve')
        st.write('AUC score: 0.92') 
    with right_column:
        st.image(image_PRC_dt, caption='Precision-Recall Curve')
        st.write('Precision-Recall AUC score: 0.93') 
    
    st.subheader('Naive Bayes Model')
    
    st.subheader('With features that selected using Boruta')
    st.write('Accuracy on training set: 0.899')
    st.write('Accuracy on test set: 0.900')
    st.write('According to the score of accuracy of both training set and test set, the difference score between them are not far so which mean the dataset is good and didnt need to do data imbalance treatment again')

    st.subheader('With all features')
    st.write('Accuracy on training set: 0.899')
    st.write('Accuracy on test set: 0.901')
    st.write('According to the score of accuracy of both training set and test set, the difference score between them are not far so which mean the dataset is good and didnt need to do data imbalance treatment again')

    st.subheader('Comparison of the result of Naive Bayes with different features')
    image_nb_selectf = Image.open('NB_cm_fs.PNG')
    image_nb_allf = Image.open('NB_cm_allfs.PNG')

    left_column, right_column = st.columns(2)
    with left_column:
        st.image(image_nb_selectf, caption='Confusion Matrix for Naive Bayes with selected features') 
    with right_column:
        st.image(image_nb_allf, caption='Confusion Matrix for Naive Bayes with all features')

    st.write('According to table above, we cant see clearly that which Naive Bayes that with what features have the more higher accuracy but in the comparison table want can clearly see that the Naive Bayes that with features selected by Boruta have slightly overall higher scores than the Naive Bayes that ahve all features which have 99.32% of precision, 80.17% of recall, 88.73% of F1 score and 90.01% of accuracy.') 
    st.write('So we will choose Naive Bayes that with features that selected by Boruta to do prediction.')
    
    st.subheader('Using selected Decision Tree model to do prediction')
    image_AUC_nb= Image.open('AUC_nb.PNG')
    image_PRC_nb = Image.open('PRC_nb.PNG')

    left_column, right_column = st.columns(2)
    with left_column:
        st.image(image_AUC_nb, caption='Receiver Operating Characteistic (ROC) Curve')
        st.write('AUC score: 0.93') 
    with right_column:
        st.image(image_PRC_nb, caption='Precision-Recall Curve')
        st.write('Precision-Recall AUC score: 0.95')

    st.subheader('Performance Comparison between Decision Tree and Naive Bayes')
    image_AUC_bm= Image.open('AUC_bm.PNG')
    image_PRC_bm = Image.open('PRC_bm.PNG')

    left_column, right_column = st.columns(2)
    with left_column:
        st.image(image_AUC_bm, caption='Receiver Operating Characteistic (ROC) Curve')
    with right_column:
        st.image(image_PRC_bm, caption='Precision-Recall Curve')

    AUC_PRC_bm = st.dataframe(data={"Model":  ["AUC", "PRC"],
                                "Decision Tree": ["0.921572", "0.925074"], 
                                "Naive Bayes": ["0.926074", "0.948051"]})

    st.write('According to the Comparison ROC Curve graph, the Naive Bayes will be better than Decision Tree at first as it is "hugging" the Decision Tree but after False Positive Rate at approximately 0.2 it will have a turning point which the Decision Tree will be the one who "hugging" the Naive Bayes which mean the Decision Tree after false positive rate approximately higher 0.2 will be more better than the Naive Bayesa and the duration afterward for "hugging" the Naive Bayes will be more longer which mean the Decision Tree is more suitable model for dataset according to the Comparison ROC Curve graph.')
    st.write('According to the Precision-Recall Curve graph, the Decision Tree also will be the more suitable model for this dataset due to the duration of the "hugging" process is longer than the Naive Bayes, although the turning point from Decision Tree to Naive Bayes is at approximately 0.85 score in recall. In other word, the more duration at precision state the model in, the more suitable the model is for the data.')

    st.header('Hyperparameter Tuning with Decision Tree')
    image_tuned_dt = Image.open('tuned_cm_dt.PNG')
    image_tuned_nb = Image.open('tuned_cm_nb.PNG')

    left_column, right_column = st.columns(2)
    with left_column:
        st.image(image_dt_selectf, caption='Confusion Matrix before Tuned') 
    with right_column:
        st.image(image_tuned_dt, caption='Confusion Matrix after Tuned')
    st.write('According to the graph and table above, we can see that the Tuned Decision Tree will have more accuracy than the Normal Decision Tree, so which mean that the hyperparameter tuning is important for the Decision Tree to get more higher accuracy.')

    left_column, right_column = st.columns(2)
    with left_column:
        st.image(image_nb_selectf, caption='Confusion Matrix before Tuned') 
    with right_column:
        st.image(image_tuned_nb, caption='Confusion Matrix after Tuned')
    st.write('According to the confusion above, we can see that the Normal Naive Bayes will have more accuracy than the Tuned Naive Bayes, so which mean that the hyperparameter tuning is not suitbale for the Naive Bayes model with this dataset.')

    st.subheader('Comparison of all the type of Decision Tree and Naive Bayes')

    Finalresult_df = st.dataframe( data={"Models":  ["Normal_Decision_Tree", "Tuned_Decision_Tree", "Normal_Naive_Bayes", "Tuned_Naive_Bayes"], 
                             "Precision":   ["0.807767", "0.953650", "0.993252", "0.542977"],
                             "Recall":      ["0.906318", "0.896514", "0.801743", "0.846405"],
                             "F1_Score":    ["0.854209", "0.924200", "0.887281", "0.661558"],
                             "Accuracy":    ["0.848372", "0.927923", "0.900160", "0.575547"]})
    
    image_allmodel_score = Image.open('allmodel_score.PNG')
    st.image(image_allmodel_score, caption='Comparison accuracy of all the models')

    st.write('According to the table and graph above, we can find out that the Decision Tree after the hyper-parameter tuning is the best model for this prediction with the dataset which have the 92% of accuracy, 93% of precision, 90% of recall and 91% of f1-score among all the models with different features.')
