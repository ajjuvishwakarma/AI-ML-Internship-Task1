 # Titanic Dataset

 
import pandas as pd 
df = pd.read_csv("Titanic-Dataset.csv.csv")
print("Data Loaded Successfully")
print(df.head())
print("\nShape:", df.shape)
print("\nMissing Values:")
print(df.isnull().sum())


# Handle Missing Values
df["Age"]= df['Age'].fillna(df['Age'].mean())
df["Embarked"]= df['Embarked'].fillna(df['Embarked'].mode()[0])


# Remove missing values of Cabin column


df.drop('Cabin',axis=1, inplace=True)
print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# Categorical Data Encoding 
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["Sex"] = le.fit_transform(df["Sex"])
df["Embarked"]= le.fit_transform(df["Embarked"])

print(df.head())  

# Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
df[["Age", "Fare"]] = scaler.fit_transform(df[["Age", "Fare"]])
print(df[["Age","Fare"]].head())

# Create Boxplot 


import seaborn as sns 
import matplotlib.pyplot as plt 
sns.boxplot(x=df["Fare"])
plt.title("Fare Outliers")
plt.show() 

# Remove Outliers
Q1 = df["Fare"].quantitle(0.25)
Q3 = df["Fare"].quantitle(0.75)
IQR = Q3-Q1
lower = Q1 - 1.5*IQR
upper = Q3+1.5*IQR
df = df[(df["Fare"] >= lower) & (df["Fare"] <= upper)]
print("New Shape:", df.shape)

df.to_csv("cleand_titanic.csv", index=false)
print("Cleaned Dataset Saved Successfully")