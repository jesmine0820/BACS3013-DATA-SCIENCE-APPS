import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Data Loading
df = pd.read_csv('online_Retail.csv', encoding='latin1')

# Calculate Total Sales
df['TotalSales'] = df['Quantity'] * df['UnitPrice']

# Adjust for InvoiceDate
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%d/%m/%Y %H:%M')
df['Year'] = df['InvoiceDate'].dt.year
df['Month'] = df['InvoiceDate'].dt.month
df['Day'] = df['InvoiceDate'].dt.day
df['DayOfWeek'] = df['InvoiceDate'].dt.strftime('%A')
df['YearMonth'] = df['InvoiceDate'].dt.strftime('%Y-%m')

# Data Cleaning
df.drop_duplicates(inplace=True)
df.dropna(subset=['CustomerID'], inplace=True)
df = df[df['StockCode'] != "M"]

df = df.dropna()
df = df[df['Quantity'] > 0]
df.reset_index(drop=True, inplace=True)

df = df[df['UnitPrice'] > 0]
df.reset_index(drop=True, inplace=True)

# Remove Outliers by using Z-Score
# Calculate the mean and standard deviation for each column
mean = df[['Quantity', 'UnitPrice', 'TotalSales']].mean()
std = df[['Quantity', 'UnitPrice', 'TotalSales']].std()

# Calculate the Z-scores manually
z_score = (df[['Quantity', 'UnitPrice', 'TotalSales']] - mean) / std

# Set the threshold
threshold = 3

# Remove outliers based on the Z-score threshold
df = df[(z_score.abs() < threshold).all(axis=1)]

# Normalize continuous data
numerical_Columns = ['Quantity', 'UnitPrice', 'Year']
x_normalized = (df[numerical_Columns] - df[numerical_Columns].min()) / (df[numerical_Columns].max() - df[numerical_Columns].min())
df['Quantity'] = x_normalized['Quantity']
df['UnitPrice'] = x_normalized['UnitPrice']

# Data Selection
y = df.TotalSales.values
x_Data = df.drop(['InvoiceNo', 'InvoiceDate', 'Day', 'DayOfWeek', 'YearMonth', 'TotalSales'], axis=1)

# Find the last purchase date
last_date = df['InvoiceDate'].max()

# Calculate the recency, frequency and monetary values (RFM)
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (last_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalSales': 'sum'
}).reset_index()

rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

# Save the column names
column_names = x_Data.columns

# Group the stockCode and description
stockcode_description = df.groupby('StockCode')['Description'].unique().to_dict()

# Save the country list
countries = df['Country'].unique()

# Encoding the categorical data
le_stockcode = LabelEncoder()
le_description = LabelEncoder()
le_customerid = LabelEncoder()
le_country = LabelEncoder()

x_Data['StockCode'] = le_stockcode.fit_transform(x_Data['StockCode'])
x_Data['Description'] = le_description.fit_transform(x_Data['Description'])
x_Data['CustomerID'] = le_customerid.fit_transform(x_Data['CustomerID'])
x_Data['Country'] = le_country.fit_transform(x_Data['Country'])

# Train Test Split
x_train, x_test, y_train, y_test = train_test_split(x_Data, y, test_size=0.2, random_state=0)

# Data Modeling using Random Forest Algorithm
parameter_grid = {
    'n_estimators': [300],
    'max_depth': [20],
    'min_samples_split': [2],
    'max_features': [None]
}

# Train the model
model = GridSearchCV(RandomForestRegressor(random_state=0), parameter_grid, cv=5, n_jobs=-1)
model.fit(x_train, y_train)

# Select the features for customer segmentation clustering
X = rfm[['Recency', 'Frequency', 'Monetary']]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)
kmeans.fit(X_scaled)

# Add unseen labels dynamically
def add_unseen_label(encoder, value):
    if value not in encoder.classes_:
        new_classes = np.append(encoder.classes_, value)
        encoder.classes_ = new_classes
    return encoder.transform([value])[0]

# Prediction function for total sales
def prediction(model, x):
    x = x.reindex(columns=column_names, fill_value=0)  
    return model.predict(x)

# Customer segmentation prediction function
def predict_segmentation(rfm_values, scaler, model):
    rfm_scaled = scaler.transform([rfm_values])
    return model.predict(rfm_scaled)

# Main Streamlit app
def main():
    # Create tabs
    tab1, tab2 = st.tabs(["Prediction for Total Sales", "Customer Segmentation Analysis"])

    # Predict Total Sales tab
    with tab1:
        st.header("Prediction for Total Sales")

        # Input field for StockCode
        stockcode_input = st.text_input("Enter the Stock Code:")

        # Automatically retrieve the corresponding description
        description_input = ""
        if stockcode_input:
            if stockcode_input in stockcode_description:
                # Fetch the first description associated with the StockCode
                description_input = stockcode_description[stockcode_input][0]
                st.write(f"Product Name: {description_input}")
            else:
                st.warning("StockCode not found in the database.")
        
        # Input fields for UnitPrice, Quantity, CustomerID, and Year, Month
        unitprice_input = st.text_input("Enter the Unit Price (Pounds):")
        quantity_input = st.text_input("Enter the Quantity Sold:")
        customerid_input = st.text_input("Enter the Customer ID:")

        # Country dropdown for selection
        country_input = st.selectbox("Select the Country:", countries)

        year_input = st.text_input("Enter the Year:")
        month_input = st.text_input("Enter the Month (1-12):")

        # Perform validation and prediction
        if st.button("Predict Total Sales"):
            try:
                # Convert inputs
                unit_price = float(unitprice_input)
                quantity = int(quantity_input)
                year = int(year_input)
                month = int(month_input)

                # Handle unseen labels for StockCode, Description, CustomerID, and Country
                stockcode_encoded = add_unseen_label(le_stockcode, stockcode_input)
                description_encoded = add_unseen_label(le_description, description_input)
                customerid_encoded = add_unseen_label(le_customerid, customerid_input)
                country_encoded = add_unseen_label(le_country, country_input)

                # Create input DataFrame
                X = pd.DataFrame([[stockcode_encoded, description_encoded, unit_price, quantity, customerid_encoded, country_encoded, year, month]],
                                columns=['StockCode', 'Description', 'UnitPrice', 'Quantity', 'CustomerID', 'Country', 'Year', 'Month'])
                
                # Predict total sales
                result = prediction(model, X)
                st.success(f"The Predicted Total Sales is: {result[0]}")

            except ValueError as e:
                st.error(f"Error: {e}")

    # Customer Segmentation tab
    with tab2:
        st.header("Customer Segmentation")

        recency_input = st.number_input("Enter Recency:", min_value=0)
        frequency_input = st.number_input("Enter Frequency:", min_value=0)
        monetary_input = st.number_input("Enter Monetary Value:", min_value=0)

        if st.button("Predict Customer Segment"):
            rfm_values = np.array([recency_input, frequency_input, monetary_input])
            segment = predict_segmentation(rfm_values, scaler, kmeans)
            st.write(f"Customer belongs to segment: {segment[0]}")

            # Plot Customer Segmentation
            st.subheader("Customer Segmentation Visualization")

            # Generate data for visualization
            rfm = pd.DataFrame({
                'Recency': [recency_input],
                'Frequency': [frequency_input],
                'Monetary': [monetary_input],
                'Segment': [segment[0]]
            })

            # Reload data again to gain a more accurate data
            X = pd.read_csv('onlineRetail.csv')  
            X['TotalSales'] = X['Quantity'] * X['UnitPrice']
            X['InvoiceDate'] = pd.to_datetime(X['InvoiceDate'], format='%d/%m/%Y %H:%M')
            last_date = pd.to_datetime(X['InvoiceDate']).max()
            rfm_existing = X.groupby('CustomerID').agg({
                'InvoiceDate': lambda x: (last_date - x.max()).days,
                'InvoiceNo': 'nunique',
                'TotalSales': 'sum'
            }).reset_index()
            rfm_existing.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
            rfm_existing['Segment'] = kmeans.predict(scaler.transform(rfm_existing[['Recency', 'Frequency', 'Monetary']])) 

            # Plotting cluster
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=rfm_existing, x='Recency', y='Monetary', hue='Segment', palette='viridis', alpha=0.7)
            plt.scatter(rfm['Recency'], rfm['Monetary'], color='red', s=100, label='New Customer')
            plt.title('Customer Segmentation')
            plt.xlabel('Recency')
            plt.ylabel('Monetary Value')
            plt.legend()
            st.pyplot(plt)

if __name__ == '__main__':
    main()
