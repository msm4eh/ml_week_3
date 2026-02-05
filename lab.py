# %%
# Imports - Libraries needed for data manipulation and ML preprocessing
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For data visualization
# Make sure to install sklearn in your terminal first!
# Use: pip install scikit-learn
from sklearn.model_selection import train_test_split  # For splitting data
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # For scaling
from io import StringIO  # For reading string data as file
import requests  # For HTTP requests to download data

#%% 
# Question for College Data Set:
# How well can we predict if a university's 4-year graduation rate is above or below the median?

# Independent Business Metric: Graduation Rate

# %%
# Step Two: Data Preperation
# %%
# load college data set
college_url = (
    "https://raw.githubusercontent.com/UVADS/DS-3021/"
    "main/data/cc_institution_details.csv"
)

college = pd.read_csv(college_url)
college.info()

# %%
# Create target variable
college["grad_ontime_above_median"] = (
    college.grad_100_percentile > 50
).astype(int)

college.grad_ontime_above_median.value_counts()

# %%
# fill missing numerical values with median
num_cols = college.select_dtypes("number").columns
college[num_cols] = college[num_cols].fillna(college[num_cols].median())

# Fill missing categorical values with "Unknown"
cat_cols = college.select_dtypes("object").columns
college[cat_cols] = college[cat_cols].fillna("Unknown")

# %%
# prevalance 
prevalence = college.grad_ontime_above_median.mean()
print(f"Baseline (Prevalence): {prevalence:.2%}")

#%%

# Drop unnecessary columns
drop = [
    "index", "unitid", "chronname", "city", "site",
    "hbcu", "flagship", "nicknames", "similar", "state",
    "counted_pct", "long_x", "lat_y", "vsa_year",
    "vsa_grad_after4_first", "vsa_grad_elsewhere_after4_first",
    "vsa_enroll_after4_first", "vsa_enroll_elsewhere_after4_first",
    "vsa_grad_after6_first", "vsa_grad_elsewhere_after6_first",
    "vsa_enroll_after6_first", "vsa_enroll_elsewhere_after6_first",
    "vsa_grad_after4_transfer", "vsa_grad_elsewhere_after4_transfer",
    "vsa_enroll_after4_transfer", "vsa_enroll_elsewhere_after4_transfer",
    "vsa_grad_after6_transfer", "vsa_grad_elsewhere_after6_transfer",
    "vsa_enroll_after6_transfer", "vsa_enroll_elsewhere_after6_transfer"
]

college_clean = college.drop(columns=[c for c in drop if c in college.columns])

# %%
# Convert categorical variables to 'category' dtype

obj_cols = college_clean.select_dtypes(include="object").columns 
college_clean[obj_cols] = college_clean[obj_cols].astype("category")

# %% 
top_carnegie = college_clean["carnegie_ct"].value_counts().nlargest(5).index

college_clean["carnegie_ct"] = college_clean["carnegie_ct"].apply(
    lambda x: x if x in top_carnegie else "Other"
).astype("category")

college_clean["carnegie_ct"].value_counts()
# %%
# Scale numeric features using Min-Max Scaling
numeric_cols = list(college_clean.select_dtypes("number"))
numeric_cols.remove("grad_ontime_above_median")

college_clean[numeric_cols] = MinMaxScaler().fit_transform(
    college_clean[numeric_cols]
)

# %%
# One-Hot Encode categorical variables
category_list = list(college_clean.select_dtypes("category"))

college_encoded = pd.get_dummies(
    college_clean,
    columns=category_list
)

college_encoded.info()
# %%
# Split data into training and testing sets with stratification
train, test = train_test_split(
    college_encoded,
    train_size=0.7,
    stratify=college_encoded.grad_ontime_above_median,
    random_state=42
)
tune, test = train_test_split(
        test,
        train_size=0.5,
        stratify=test.placed,
        random_state=42
    )
# %%
# Step Three: What do your instincts tell you about the data. Can it address your problem, what areas/items are you worried about? 

# My instincts tell me that this can predict graduate rate in 4-years somewhat well given we have a wide range of categorical and numerical data and lots of data to train the model. However, there are missing values that needed to be filled and some values that needed to be collapsed, so the data is not as specific anymore. There may also be important factors that affect graduation rate that were not included in this data set. 

# %% 
# Step Four: creating the functions

def load_college_data(path):
    return pd.read_csv(path)

def fill_missing_values(df):
    num_cols = df.select_dtypes("number").columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    df[cat_cols] = df[cat_cols].fillna("Unknown")

    return df


def create_target(df):
    df["grad_ontime_above_median"] = (
        df["grad_100_percentile"] > 50
    ).astype(int)

    print("Prevalence:")
    print(df["grad_ontime_above_median"].value_counts(normalize=True))
    return df

def drop_unneeded_columns(df):
    drop = [
        "index", "unitid", "chronname", "city", "site",
        "hbcu", "flagship", "nicknames", "similar", "state",
        "counted_pct", "long_x", "lat_y", "vsa_year",
        "vsa_grad_after4_first", "vsa_grad_elsewhere_after4_first",
        "vsa_enroll_after4_first", "vsa_enroll_elsewhere_after4_first",
        "vsa_grad_after6_first", "vsa_grad_elsewhere_after6_first",
        "vsa_enroll_after6_first", "vsa_enroll_elsewhere_after6_first",
        "vsa_grad_after4_transfer", "vsa_grad_elsewhere_after4_transfer",
        "vsa_enroll_after4_transfer", "vsa_enroll_elsewhere_after4_transfer",
        "vsa_grad_after6_transfer", "vsa_grad_elsewhere_after6_transfer",
        "vsa_enroll_after6_transfer", "vsa_enroll_elsewhere_after6_transfer"
    ]

    return df.drop(columns=[c for c in drop if c in df.columns])

def collapse_factor_levels(df):
    top_carnegie = df["carnegie_ct"].value_counts().nlargest(5).index

    df["carnegie_ct"] = df["carnegie_ct"].apply(
        lambda x: x if x in top_carnegie else "Other"
    ).astype("category")

    return df


def scale_numeric(df):
    num_cols = list(df.select_dtypes("number"))
    num_cols.remove("grad_ontime_above_median")

    df[num_cols] = MinMaxScaler().fit_transform(df[num_cols])
    return df

def encode_categoricals(df):
    cat_cols = list(df.select_dtypes("category"))
    return pd.get_dummies(df, columns=cat_cols)

def split_data(df):
    train, test = train_test_split(
        df,
        train_size=0.7,
        stratify=df.grad_ontime_above_median,
        random_state=42
    )

    tune, test = train_test_split(
        test,
        train_size=0.5,
        stratify=test.grad_ontime_above_median,
        random_state=42
    )

    return train, tune, test


#%% 
# Question for Job Data Set:
# How well can we predict if a student was placed or not based on factors such as secondary and higher secondary school percentage, specialization, and work experience, and others?
# Independent Business Metric: Job Placement

# %% Step Two: Data Preperation
job_url = (
    "https://raw.githubusercontent.com/DG1606/CMS-R-2020/"
    "master/Placement_Data_Full_Class.csv"
)

job = pd.read_csv(job_url)
job.info()

#%%
# %%
# Create Target Variable
job["placed"] = (job["status"] == "Placed").astype(int)

job.placed.value_counts()
# %%
prevalence_job = job.placed.mean()
print(f"Baseline (Prevalence): {prevalence_job:.2%}")

# %%
# Fill numeric columns with median
num_cols_job = job.select_dtypes("number").columns
job[num_cols_job] = job[num_cols_job].fillna(job[num_cols_job].median())

# Fill categorical columns with "Unknown"
cat_cols_job = job.select_dtypes("object").columns
job[cat_cols_job] = job[cat_cols_job].fillna("Unknown")
# %%
# Drop unnecessary and empty columns
drop_job = [
    "sl_no",     
    "salary",    
    "status"    
]

job_clean = job.drop(columns=[c for c in drop_job if c in job.columns])
job_clean.info()
# %%
# convert objects to categories
obj_cols = job_clean.select_dtypes("object").columns
job_clean[obj_cols] = job_clean[obj_cols].astype("category")

#%%
# Collapse specializations
job_clean["specialisation"] = job_clean["specialisation"].apply(
    lambda x: x if x in ["Mkt&HR", "Mkt&Fin"] else "Other"
).astype("category")

# %%
# normalize with min max
numeric_cols = list(job_clean.select_dtypes("number"))
numeric_cols.remove("placed")

job_clean[numeric_cols] = MinMaxScaler().fit_transform(
    job_clean[numeric_cols]
)

# %%
# one hot encoding for categorical data
category_list_job = list(job_clean.select_dtypes("category"))

job_encoded = pd.get_dummies(
    job_clean,
    columns=category_list_job
)

job_encoded.info()

# %%
train_job, test_job = train_test_split(
    job_encoded,
    train_size=0.7,
    stratify=job_encoded.placed,
    random_state=42
)

tune_job, test_job = train_test_split(
    test,
    train_size=0.5,
    stratify=test.placed,
    random_state=42
)

train_job.shape, tune_job.shape, test.shape_job

#%%
# Part Three: What do your instincts tell you about the data. Can it address your problem, what areas/items are you worried about? 

# My instincts tell me that the reason salary is empty so often is because non-placed students don't have a salary. However, I wonder if filling missing values with zero and keeping salary would have made the model more accurate. While this data set is easily interpretable, I am worried that it may be too small.

#%% Part Four: Creating functions

def load_job_data(path):
    return pd.read_csv(path)

def fill_missing_values(df):
    num_cols_job = df.select_dtypes("number").columns
    df[num_cols_job] = df[num_cols_job].fillna(df[num_cols_job].median())

    cat_cols_job = df.select_dtypes(include=["object", "category"]).columns
    df[cat_cols_job] = df[cat_cols_job].fillna("Unknown")

    return df

def create_target(df):
    df["placed"] = (df["status"] == "Placed").astype(int)

    print("Prevalence:")
    print(df["placed"].value_counts(normalize=True))

    return df

def drop_unneeded_columns(df):
    drop_job = ["sl_no", "salary", "status"]
    return df.drop(columns=[c for c in drop_job if c in df.columns])

def collapse_factor_levels(df):
    df["specialisation"] = df["specialisation"].apply(
        lambda x: x if x in ["Mkt&HR", "Mkt&Fin"] else "Other"
    ).astype("category")

    return df

def scale_numeric(df):
    num_cols_job = list(df.select_dtypes("number"))
    num_cols_job.remove("placed")

    df[num_cols_job] = MinMaxScaler().fit_transform(df[num_cols_job])
    return df

def encode_categoricals(df):
    cat_cols_job = list(df.select_dtypes("category"))
    return pd.get_dummies(df, columns=cat_cols_job)

def split_data(df):
    train_job, test_job = train_test_split(
        df,
        train_size=0.7,
        stratify=df.placed,
        random_state=42
    )

    tune_job, test_job = train_test_split(
        test_job,
        train_size=0.5,
        stratify=test.placed,
        random_state=42
    )

    return train_job, tune_job, test_job