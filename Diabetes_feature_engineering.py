################################################################################################
# Diabetes Feature Engineering Project
################################################################################################

# 1. Introduction:

# Project Scope: A medical supplier company needs to predict whether a person has diabetes using ML. However,
# before trying out the ML algorithms, we need to perform an EXPLORATORY DATA ANALYSIS and  FEATURE ENGINEERING.

# Story of The Data: The dataset is derived from  'USA national institute of diabetes and digestive and kidney diseases'
# dataset. The target variable is identified as 0 and 1 where 0 refers to negative diabetes test result and 1 refers to
# positive diabetes test result.

# 9 feature, 768 observations, 24 KB.

# 2. Exploratory Data Analysis:

# loading the dataset


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
!pip install missingno
import missingno as msno
from datetime import date

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

# loading the diabetes dataset.
data = pd.read_csv(r"C:\Users\nejat\PycharmProjects\pythonProject2\Datasets\diabetes.csv")
df = data.copy()

# Quick glance at a dataset
def check_df(data, x=5):
    print('################################# shape ##########################')
    print(data.shape )
    print('################################# type ##########################')
    print(data.dtypes)
    print('################################# head ##########################')
    print(data.head(x))
    print('################################# tail ##########################')
    print(data.tail(x))
    print('################################# null ##########################')
    print(data.isnull().sum().sort_values(ascending=False))
    print('################################# quantiles #####################')
    print(data.describe([0, 0.05, 0.5, 0.95, 0.99, 1]).T)
check_df(df)

# At the first sight, it seems that we have only integers and float values.
# NO NAN Values
# Pregnancy's max value falls far away from its mean or median values.
# Insulin's mean and median values are far away from each other.



# Numerical and Categorical Variable analysis:

def grab_col_names (dataframe, categorical = 10, cardinal =20):

    # categoricals
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtype == 'O']
    #  [col for col in df.columns if str(df[col].dtypes) in ["category","object","bool"]]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].dtype == 'O' and
                   dataframe[col].nunique() > cardinal]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].dtype != 'O' and
                   dataframe[col].nunique() < categorical]
    cat_cols += num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # numericals

    num_cols = [col for col in dataframe.columns if dataframe[col].dtype != 'O' and
                col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names (df, categorical = 10, cardinal =20)

# The only categorical variable is our target variable 'Pregnancies' and the target var.
# Rest of the variables are numerical.
# There is no categorical variable with high cardinality.

# Numerical Variable Analysis
def num_summary(dataframe, numeric_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numeric_col].describe(quantiles).T)

    if plot:
        dataframe[numeric_col].hist()
        plt.xlabel(numeric_col)
        plt.title(numeric_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(df, col, plot=True)

# Results
# In this dataset, majority has following characteritics;
# 75-150 glucose levels
# 60- 85 blood pressure
# 0 - 40 skin tickness.
# 0-100 insulin levels
# 20-45 BMI
# 20-35 Age
# 0.0 - 0.5 Diabetes pedigree function.

# Categorical Variable Analysis
def cat_summary( dataframe, col_name, plot = False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                       'Ratio':100 * dataframe[col_name].value_counts()/len(dataframe)}))
    print('#####################',col_name,'############################')

    if plot:
        sns.countplot(x = dataframe[col_name], data = dataframe)
        plt.show(block = True)

for col in cat_cols:
    cat_summary(df, col, plot = True)

# Results:
# In this dataset majority has following categorical characteristics:
# Number of pregnancies 0-5.
# Target variable / test result is negative(0).

########################## Target Summary #############################################
# Comparing Categorical Var. Values against the Target Var. Values
df.groupby('Pregnancies').agg({'Outcome': ['mean', 'sum']}).reset_index().\
    sort_values(by = ('Outcome','sum'),ascending = False)

# Comparing Numerical Var. Values against the Target Var. Values
def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: 'mean'}))

for col in num_cols:
    target_summary_with_num(df, 'Outcome', col)

# Until now, INSULIN levels and GLUCOSE levels has some
# distintive values. The mean difference for target variable in the other features
# are closer.

########################## Outlier Analysis #############################################

def outlier_thresholds(dataframe, col_name, q1=0.10, q3=0.90):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index


num_cols
# Out: ['Glucose',
#  'BloodPressure',
#  'SkinThickness',
#  'Insulin',
#  'BMI',
#  'DiabetesPedigreeFunction',
#  'Age']

grab_outliers(df, 'Pregnancies', True)
grab_outliers(df, 'Glucose', True)
grab_outliers(df, 'BloodPressure',True)
grab_outliers(df, 'Insulin',True)
grab_outliers(df, 'BMI',True)
grab_outliers(df, 'DiabetesPedigreeFunction',True)
grab_outliers(df, 'Age',True)

########################## NaN Value Analysis #############################################


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df)
# No missing values detected.

########################## Correlation Analysis #############################################

corr = df[num_cols].corr()

sns.set(rc={'figure.figsize': (12, 12)})
sns.heatmap(corr, cmap="RdBu")
plt.show()

# Insulin - Glucose relationship is above +0.30
# Insulin - Skin Thickness is around +0.43
# BMI - Skin Thickness is nearly +0.40
# Correlation between the other numeric features are
# even lower than these values.
# As a result, I did not find any high correlation (like above %70) variables
# in this dataset.

df.corr()

# In correlation matrix, I dont think I should compare categoric var. with numerics but,
# when I add pregnancy and the target val to our comparison
# I notice the relationship between age and pregnancy (0.54)
# Also Glucose level and the Outcome result. (nearly 0.5)



################################################################################################
# 2. Feature Engineering
################################################################################################

# Handling NaN.

df.isnull().sum()
# As I stated in section 1.0, this dataset does not contain any NaN value. However,
# datasource tells us that some features like Glucose, Insulin cannot have 0 value.
# Therefore, we might need to count them as NAN. Let's check descripteves of the variables again.

df.describe().T

# Glucose
# BloodPressure
# SkinThickness
# Insulin
# BMI :Based on these figures a mean BMI of 12 as the lower limit for human survival emerges - a value first proposed by James et al (1988)
# Thus, for these features, we will count 0 as NaN values. Why not outlier??
# BMI lower than 12 will also be NaN
# 90/60mmHg or less is considered the lowest blood pressure before death.

df[df['BloodPressure']<60]['BloodPressure'].count() #121
df[df['BloodPressure']<60].index
df[df['BMI']<12]['BMI'].count() #11
df[df['BMI']<12]['BMI'].index
df[df['Insulin'] ==0]['Insulin'].count() #374
df[df['Insulin'] ==0]['Insulin'].index
df[df['SkinThickness'] == 0]['SkinThickness'].count() #227
df[df['SkinThickness'] == 0]['SkinThickness'].index
df[(df['BloodPressure']<60) & (df['Insulin'] ==0) & (df['SkinThickness'] == 0)].count() #47

# Lets replace 0's with NaN values first.
df['BloodPressure'].replace(0, np.nan, inplace = True)
df['BMI'].replace(0, np.nan, inplace = True)
df['Insulin'].replace(0, np.nan, inplace = True)
df['SkinThickness'].replace(0, np.nan, inplace = True)
df['Glucose'].replace(0, np.nan, inplace = True)
df.isnull().sum()

#a= df.copy()
#a[a['BloodPressure']<60].count()
#a['BloodPressure']=a.apply(lambda x: x['BloodPressure'] if (x['BloodPressure']>=60) else np.nan,axis=1)
#a.isnull().sum()
#df["BloodPressure"] = np.where(df.BloodPressure < 60, nan, df["Glucose"])

missing_values_table(df)
# Almost 50% of the Insulin feature is missing.
# Almost 30% of  the SkinThickness feature is missing.


# The relationship between the missings and the target var.
na_cols = [col for col in df.columns if df[col].isnull().sum() != 0]
def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()

    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")


missing_vs_target(df, "Outcome", na_cols)
# BloodPressure : if NaN -> Target_mean = 0.45. if value -> Target is lower(0.34). 35 NaN.
# SkinThickness : if NaN -> Target_mean = 0.38. if value -> Target is lower(0.33). 227 NaN.
# Insulin       : if NaN -> Target_mean = 0.36. if Value-> Target is lower(0.33). 374 NaN.
# BMI           : if NaN -> Target_mean = 0.18. if value -> 0.35. 11 NaN.

############################# Filling Missing Values ####################################

#1. SkinThickness:

df.groupby('Outcome').agg({'SkinThickness': 'median'})
# OUT: 0:27 1:32
# I decided to fill skinThickness NaN values with their median value.
# 27 for a healthy person and 32 for a diabetic person
df['SkinThickness'].fillna(df.groupby('Outcome')['SkinThickness'].transform('median'), inplace = True)
#df.loc[(df['SkinThickness'].isnull() & df['Outcome'] == 0), 'SkinThickness'] = 27
#df.loc[(df['SkinThickness'].isnull() & df['Outcome'] == 1), 'SkinThickness'] = 27
df.isnull().sum()

#2. Insulin:

df.groupby('Outcome').agg({'Insulin': 'median'})
# OUT: 0:102.5 1:169.5
# Fill with the median value.
df['Insulin'].fillna(df.groupby('Outcome')['Insulin'].transform('median'),inplace = True)
df.isnull().sum()

#3. BloodPressure:

df['BloodPressure'].fillna(df.groupby('Outcome')['BloodPressure'].transform('median'),inplace = True)
df.isnull().sum()

#4. BloodPressure:

df['BMI'].fillna(df.groupby('Outcome')['BMI'].transform('median'),inplace = True)
df.isnull().sum()


#4. Glucose:

df['Glucose'].fillna(df.groupby('Outcome')['Glucose'].transform('median'),inplace = True)
df.isnull().sum()

# Approach 2: Finding out the bloodpressure value from insulin and glucose???
# df[df['BloodPressure'].isnull()].head(45)
# It seems that when the bloodpressure is NaN, the insulin is also NaN.
# To see the correlations between missing values;
#msno.bar(df)
#plt.show()

############################# Generating New Features ####################################

df['GLUCOSE_CAT_NEW'] = pd.cut(x = df['Glucose'], bins = [-1,80,140,160,200],
                               labels = ['Hypoglecimia','Normal', 'Impaired_Glucose', 'Diabetic_Glucose']) # 2 hour

df['AGE_CATEGORIES_NEW'] = pd.cut( x = df['Age'], bins = [18,44,64,100],
                               labels = ['Adults', 'Matures' , 'Boomers'])


df['DIASTOLIC_BLOOD_PRESSURE_NEW'] = pd.cut(x = df['BloodPressure'], bins = [0,80,89,120,300],
                               labels = ['Normal', 'Norm_Check_Sylostic' , 'Hypertension', 'Hypertension_Crisis'])

df['INSULIN_NEW'] = pd.cut(x= df['Insulin'], bins = [0, 120, 1000], labels = ['Normal','Abnormal'])

df['BMI_CAT_NEW'] = pd.cut(x = df['BMI'], bins = [0,18,25,29,68], labels = ['Underweight', 'Normal', 'Overweight', 'Obesity'])

df['Pregnancies'].describe()

df.loc[(df['Pregnancies'] == 0), 'PREGNANT_CAT_NEW']  = 'NO_TIME'
df.loc[(df['Pregnancies'] == 1), 'PREGNANT_CAT_NEW']  = 'ONE_TIME'
df.loc[(df['Pregnancies'] > 1), 'PREGNANT_CAT_NEW']   = 'MANY_TIME'

df.head()
df.isnull().sum()

############################# Handling Outliers ####################################

cat_cols, num_cols, cat_but_car = grab_col_names (df, categorical = 10, cardinal =20)

num_cols

grab_outliers(df, 'Glucose', True) #
grab_outliers(df, 'BloodPressure',True) #
grab_outliers(df, 'Insulin',True).shape #18
grab_outliers(df, 'BMI',True).shape #0
grab_outliers(df, 'DiabetesPedigreeFunction',True).shape #4
grab_outliers(df, 'Age',True).shape #0

x= 18/df.shape[0]

df.shape
replace_with_thresholds(df, 'Insulin')
replace_with_thresholds(df, 'DiabetesPedigreeFunction')
df.shape


############################# Encoding ####################################

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

df.head()
#GLUCOSE_CAT_NEW : Label encoder due to ordinality.
# BMI_CAT_NEW: Label encoder
label_encoder(df, 'GLUCOSE_CAT_NEW')
label_encoder(df, 'BMI_CAT_NEW')
label_encoder(df, 'INSULIN_NEW')
df.head()

# AGE_CATEGORIES: One-hot encoding
# PREGNANT_CAT_NEW: One-hot encoding
# INSULIN_NEW: One-hot
# DIASTOLIC_BLOOD_PRESSURE: One-hot

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

df = one_hot_encoder(df, ohe_cols)

df.head()

############################# Scaling ####################################

ss = StandardScaler()
df[num_cols] = ss.fit_transform(df[num_cols])
df.head()

num_cols

############################# Modelling ####################################

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)
#Accuracy Score: 0.883

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(rf_model, X_train)












