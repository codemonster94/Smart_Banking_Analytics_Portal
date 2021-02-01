import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sqlalchemy import column

#=======================================================================================
class Fraud_preprocess:
    def initialize_columns(self, data):  # depend ......................
        data.columns = ['step', 'customer', 'age', 'gender', 'zipcodeOri', 'merchant',
                        'zipMerchant', 'category', 'amount']
        return data

    def drop_columns(self, data):
        data_reduced = data.drop(['zipcodeOri', 'zipMerchant'], axis=1)
        return data_reduced

    def obj_to_cat(self, data_reduced):
        col_categorical = data_reduced.select_dtypes(include=['object']).columns
        for col in col_categorical:
            data_reduced[col] = data_reduced[col].astype('category')
        # categorical values ==> numeric values
        data_reduced[col_categorical] = data_reduced[col_categorical].apply(lambda x: x.cat.codes)
        return data_reduced
# ======================================================================================================================
class LA_preprocess:
    def initialize_columns(self, data):  # depend ......................
        data.columns = ['ID', 'Age', 'Experience', 'Income', 'ZIP Code', 'Family', 'CCAvg',
                        'Education', 'Mortgage', 'Securities Account',
                           'CD Account', 'Online', 'CreditCard']
        return data

    def drop_columns(self, data):
        new_data = data.drop(["ID","ZIP Code"], axis=1)
        return new_data

    def encoder(self,data):
        le = LabelEncoder()
        cat_cols = ['Family', 'Education', 'Securities Account', 'CD Account', 'Online', 'CreditCard']
        data[cat_cols] = data[cat_cols].apply(le.fit_transform)
        return data
#=======================================================================================================================
class LR_preprocess:
    def initialize_columns(self, data):
        data.columns = ['RowID','Loan_Amount', 'Term', 'Interest_Rate', 'Employment_Years','Home_Ownership',
         'Annual_Income', 'Verification_Status','Loan_Purpose', 'State', 'Debt_to_Income', 'Delinquent_2yr',
         'Revolving_Cr_Util', 'Total_Accounts','Longest_Credit_Length']
        return data

    def drop_col(self,data):
        return data.drop(columns=['RowID'])

    def feature_engg(self,data):
        data['Term']= data['Term'].str.extract('(\d+)',expand=False)
        data['Term'] = pd.to_numeric(data['Term'])

        cols = ['Home_Ownership','Verification_Status','Loan_Purpose','State']
        data[cols]=data[cols].fillna(data.mode().iloc[0])

        le = LabelEncoder()
        data[cols]=data[cols].apply(le.fit_transform)
        return data

    def outlier_removal(self,data):
        def outlier_limits(col):
            Q3, Q1 = np.nanpercentile(col, [75,25])
            IQR= Q3-Q1
            UL= Q3+1.5*IQR
            LL= Q1-1.5*IQR
            return UL, LL

        for column in data.columns:
            if data[column].dtype != 'int64':
                UL, LL= outlier_limits(data[column])
                data[column]= np.where((data[column] > UL) | (data[column] < LL), np.nan, data[column])

        return data

    def imputer(self,data):
        df_mice = data.copy()

        mice= IterativeImputer(random_state=101)
        df_mice.iloc[:,:]= mice.fit_transform(df_mice)
        return df_mice

#=======================================================================================================================
class LE_preprocess:
    def initialize_columns(self, data):
        data.columns = ['Current Loan Amount', 'Credit Score', 'Annual Income',
        'Years in current job', 'Monthly Debt', 'Years of Credit History',
        'Months since last delinquent', 'Number of Open Accounts',
        'Number of Credit Problems', 'Current Credit Balance',
        'Maximum Open Credit', 'Term_Long Term']
        return data

#=======================================================================================================================
class MA_preprocess:
    def initialize_columns(self,data):
        data.columns = ["CONCAT","postcode","Qtr","unit"]
        return data

    def drop_columns(self,data):
        return data.drop(columns=['CONCAT'])


