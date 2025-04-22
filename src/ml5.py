import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


df = sns.load_dataset('titanic')


print("Basic Info:")
print(df.info())
print("\nDescribe:")
print(df.describe())


df = df.drop(columns=['deck'])  
df['age'] = df['age'].fillna(df['age'].median())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])

# Create a PDF report
pdf = PdfPages("Titanic_EDA_Report.pdf")

plt.figure(figsize=(6, 4))
sns.countplot(x='sex', hue='survived', data=df)
plt.title("Survival Count by Gender")
pdf.savefig()
plt.show()

plt.figure(figsize=(6, 4))
sns.histplot(df['age'], bins=30, kde=True)
plt.title("Age Distribution")
pdf.savefig()
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
pdf.savefig()
plt.show()

plt.figure(figsize=(6, 4))
sns.boxplot(x=df['fare'])
plt.title("Fare Outliers")
pdf.savefig()
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(x='pclass', hue='survived', data=df)
plt.title("Class vs Survival")
pdf.savefig()
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(x='embarked', hue='survived', data=df)
plt.title("Embarked vs Survival")
pdf.savefig()
plt.show()

pdf.close()
print("\nPDF Report Saved as 'Titanic_EDA_Report.pdf'")
