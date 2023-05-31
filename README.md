# Machine-Learning-Project-Sample-Super-Store-Analysis

![1624976](https://github.com/valsanathira12/Machine-Learning-Project-Sample-Super-Store-Analysis/assets/102414557/dd7403e4-c9e9-4927-8ce7-28a7e1ff23f7)

### INTRODUCTION:
In today's highly competitive retail landscape, businesses strive to gain a competitive edge by leveraging data-driven insights to optimize their operations and enhance customer experiences. One such business, a fictional superstore, aims to harness the power of machine learning and data analysis techniques to unlock valuable patterns and trends within their sales and customer data. This Kaggle Machine Learning project delves into the vast dataset provided by the superstore, with the objective of extracting meaningful insights and building predictive models to drive decision-making and operational efficiency.By conducting a comprehensive analysis on this rich dataset, we aim to uncover hidden patterns, understand customer behavior, identify influential factors affecting sales performance, and develop accurate predictive models. These insights will enable the superstore to make better decisions, optimize its inventory management, enhance customer targeting, and ultimately maximize its sales revenue and profitability.

### PROBLEM STATEMENT

The sample Dataset includes data for the Sales of multiple products sold by a company along with subsequent information related to geography, Product categories, and subcategories, sales, and profits, segmentation among the consumers, etc. Here we are trying to perform extensive data analysis to deliver insights on how the company can increase its profits while minimizing the losses and also understand which products, regions, categories and customer segments they should target or avoid. Our task is to analyse the sales data and identify weak areas and opportunities for Super Store to boost business growth. Analyzing this data to extract crucial information and to become an important part of the decision-making process. Our task is to analyse the sales data and identify weak areas and opportunities for Super Store to boost business growth.
1.	Which state yields the highest profit and highest sales?
2.	Which Sub Category yields the highest profit and highest sales?
3.	Name of the customers providing highest sales and profit?
4.	Most preferred Shipping Mode, and its relationship with sales and profit?
5.	Identifying the segment which contributes to sales and yields higher profit?
6.	Which region generates the most sales?
7.	Identify the city contributing the to sales and profit?
8.	Which category of products generates the highest revenue and profit?
9.	What are the top-selling products in the superstore?
10.	Most commonly given discount %?
11.	Most commonly given discount %?
12.	What is the profit trend over time (monthly, yearly)?
13.	Does delivery time affect sales and profit?

### TOOLS AND TECHNOLOGIES USED TO IMPLEMENT THE PROPOSED SOLUTION

Python
Pandas
Numpy
Mathplotlib
Seaborn
Sklearn
Juypter Notebook

### DATA FEATURES AND PREDICTION:
•	Row ID - Unique ID for each row.
•	Order ID - Unique Order ID for each Customer.
•	Order Date - Order Date of the product.
•	Ship Date - Shipping Date of the Product.
•	Ship Mode - Shipping Mode specified by the Customer.
•	Customer ID - Unique ID to identify each Customer.
•	Customer Name - Name of the Customer.
•	Segment - The segment where the Customer belongs.
•	Country - Country of residence of the Customer.
•	City - City of residence of of the Customer.
•	State - State of residence of the Customer.
•	Postal Code - Postal Code of every Customer.
•	Region - Region where the Customer belong.
•	Product ID - Unique ID of the Product.
•	Category - Category of the product ordered.
•	Sub-Category - Sub-Category of the product ordered.
•	Product Name - Name of the Product
•	Sales - Sales of the Product
•	Quantity - Quantity of the Product.
•	Discount - Discount provided.
•	Profit - Profit/Loss incurred.
Note:-
As we can see there are 3 types of datatypes.object: data is in categorical format like Order ID,Product Name, State, etc.
•	int64: data is in numerical format like Postal code, Quantity, row ID.
•	float64: data is in decimal format like Sales, Discount, Profit.

### IMPLEMENTATION STEPS:
As we already discussed in the methodology section about some of the implementation details.The language used in this project is Python Programming.We are running python code in anaconda navigators, Jupyter notebook. Jupyter Notebook is much faster than other available IDE platforms for implementing ML algorithms.The main advantage of Juypter notebook is that while writing code, its really helpful for Data visualization and plotting some visualization graphs like histogram , heatmap for correlation matrics etc.
The implementation steps of a machine learning model generally involve the following:
Data Preparation: Collect and preprocess the data by cleaning, transforming, and formatting it to ensure it is suitable for analysis. This includes handling missing values, outliers, and categorical variables, as well as splitting the data into training and testing sets.
Model Selection: Choose the appropriate machine learning algorithm based on the problem type (classification, regression, clustering, etc.) and the characteristics of the data. Consider factors such as interpretability, scalability, and performance.
Model Training: Train the selected model using the training dataset. This involves feeding the algorithm with the input features and their corresponding target variables to learn the underlying patterns and relationships. Adjust hyperparameters to optimize model performance if necessary.
Model Evaluation: Assess the performance of the trained model using evaluation metrics such as accuracy, precision, recall, F1-score, or mean squared error, depending on the problem type. Evaluate the model's generalization ability by testing it on the unseen testing dataset.
Model Optimization: Fine-tune the model to improve its performance. This can involve adjusting hyperparameters, feature selection, or applying regularization techniques to prevent overfitting. Cross-validation can be used to estimate the model's performance on unseen data.
Model Deployment: Once the model has been trained and optimized, it can be deployed for real-world use. This may involve integrating the model into a larger software system or creating an API for others to access and utilize the model's predictions.

### LIBRARIES USED:
•	import pandas as pd
•	import numpy as np
•	import matplotlib.pyplot as plt
•	import seaborn as sns
•	import warnings
•	warnings.filterwarnings("ignore")
•	from sklearn.preprocessing import LabelEncoder
•	from sklearn.model_selection import train_test_split
•	from sklearn.tree import DecisionTreeClassifier
•	from sklearn.ensemble import RandomForestClassifier
•	from sklearn.naive_bayes import GaussianNB
•	from sklearn.neighbors import KNeighborsClassifier
•	from sklearn.metrics import accuracy_score
•	from sklearn.metrics import confusion_matrix
•	from sklearn.metrics import plot_confusion_matrix
•	from sklearn.metrics import classification_report

### ANALYSIS OF THE RESULT:
1.	We can infer from the analysis that the state having the highest sales and profit is  California, New York and then followed by Washington.
2.	Phones and chairs are the highest selling product. The products having the highest profits are Copies,Phones and Accessories. Whereas Supplies, Bookcases and Tables are being sold at a loss. 
3.	Mr.Sean Miller provides the highest sales. Ms.Tamara Chand has yield the highest profit.
4.	The most preferred and profitable mode of shipping is Standard Class followed by Second Class and then First Class and Same Day.
5.	The segment which contributes to highest sales and profit is Consumer, Corporate and then Home Office.
6.	Most of the sales happen in West side of country followed by East, Central and South.
7.	New York city is leading the list of cities leading sales and profit.
8.	Among the customers segment Office Supplies is the most preferred domain .Whereas technology provides the highest sales and profit.
9.	The most profitable product is Canon imageCLASS 2200 Advanced Copier. the most commonly preferred products are Staple envelop, easy-staple paper and Staples.
10.	Mostly discount is provided between 0% to 20%.
11.	Sales and Profit has been growing year by year. Highest sales happen in the months 11,12 and 9. We can concluse that most profitable sales happen in the month of December
12.	Delivery time of 4 days have the highest profit.

### OBSERVATION:
•	Decision Tree model having the highest accuracy score of 95.165%.
•	Random Forest model having the second highest accuracy of 95.065%.
•	GaussianNB Model having the third highest accuracy of 94.16%.
•	KNN Model having the lowest accuracy score of 81.36%.

### CONCLUSION:
To summarize, this Kaggle Machine Learning project embarks on a comprehensive analysis of the superstore's sales and customer data. Through the application of data analysis techniques, statistical modeling, and machine learning algorithms, we aim to unlock valuable insights that will help the superstore optimize their operations, improve customer experiences, and achieve a competitive advantage in the dynamic retail landscape. Let's dive into the data and uncover the secrets that lie within.
Our Decision Tree algorithm yields the highest accuracy 95.165%. Random Forest algorithm also has the accuracy score of 95.06%. Both these models top the list in accuracy score. Any accuracy score above 70% is considered good, but be careful because if your accuracy is extremely high, it may be too good to be true(an example of Overfitting).Thus,80% is the ideal accuracy!









