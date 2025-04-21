import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

# Load Titanic dataset
df = sns.load_dataset('titanic')

# Basic info
print("Basic Info:")
print(df.info())
print("\nDescribe:")
print(df.describe())

# Handling null values
df = df.drop(columns=['deck'])  # Too many NaNs
df['age'] = df['age'].fillna(df['age'].median())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])

# Create a PDF report
pdf = PdfPages("Titanic_EDA_Report.pdf")

# 1. Survival count by gender
plt.figure(figsize=(6, 4))
sns.countplot(x='sex', hue='survived', data=df)
plt.title("Survival Count by Gender")
pdf.savefig()
plt.show()

# 2. Age distribution
plt.figure(figsize=(6, 4))
sns.histplot(df['age'], bins=30, kde=True)
plt.title("Age Distribution")
pdf.savefig()
plt.show()

# 3. Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
pdf.savefig()
plt.show()

# 4. Fare outlier detection
plt.figure(figsize=(6, 4))
sns.boxplot(x=df['fare'])
plt.title("Fare Outliers")
pdf.savefig()
plt.show()

# 5. Class distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='pclass', hue='survived', data=df)
plt.title("Class vs Survival")
pdf.savefig()
plt.show()

# 6. Embarked vs survival
plt.figure(figsize=(6, 4))
sns.countplot(x='embarked', hue='survived', data=df)
plt.title("Embarked vs Survival")
pdf.savefig()
plt.show()

# Finish PDF
pdf.close()
print("\n✅ PDF Report Saved as 'Titanic_EDA_Report.pdf'")
