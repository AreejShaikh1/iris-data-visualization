#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 📌 Iris Dataset Exploration & Visualization

# Step 1: Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Step 2: Load the Iris dataset
def load_iris_data():
    """
    Load the Iris dataset from seaborn's built-in datasets.
    """
    iris = sns.load_dataset('iris')  # You can replace with pd.read_csv("iris.csv") if needed
    return iris

# Step 3: Inspect the dataset
def inspect_data(df):
    """
    Display basic information about the dataset.
    """
    print("🔹 Shape of dataset:", df.shape)
    print("\n🔹 Column names:", df.columns.tolist())
    print("\n🔹 First five rows:")
    print(df.head())
    print("\n🔹 Dataset info:")
    print(df.info())
    print("\n🔹 Summary statistics:")
    print(df.describe())

# Step 4: Plot pairwise relationships
def plot_pairwise_relationships(df):
    """
    Create a scatter matrix to show relationships between features.
    """
    sns.pairplot(df, hue='species', corner=True)
    plt.suptitle("🔗 Pairwise Feature Relationships", y=1.02)
    plt.tight_layout()
    plt.show()

# Step 5: Plot histograms
def plot_histograms(df):
    """
    Plot histograms to show value distributions for each numeric feature.
    """
    df.hist(figsize=(10, 6), edgecolor='black', bins=15)
    plt.suptitle("📊 Feature Value Distributions", fontsize=14)
    plt.tight_layout()
    plt.show()

# Step 6: Plot boxplots to detect outliers
def plot_boxplots(df):
    """
    Plot box plots to detect outliers in each numeric feature.
    """
    numeric_cols = df.select_dtypes(include='number').columns
    plt.figure(figsize=(10, 6))
    for i, col in enumerate(numeric_cols):
        plt.subplot(2, 2, i + 1)
        sns.boxplot(data=df, x=col)
        plt.title(f'📦 Box Plot: {col}')
        plt.tight_layout()
    plt.suptitle("📦 Outlier Detection with Box Plots", y=1.02)
    plt.tight_layout()
    plt.show()

# 🔄 Run all steps
iris_df = load_iris_data()
inspect_data(iris_df)
plot_pairwise_relationships(iris_df)
plot_histograms(iris_df)
plot_boxplots(iris_df)


# In[ ]:




