import re
import holidays
import numpy as np
import pandas as pd
from datetime import datetime


def preprocess_books_df(books_df):
    """
    Preprocesses a DataFrame containing book data by cleaning and transforming specific columns.

    - Cleans the 'price' column by removing non-numeric characters and converting it to a float.
    - Cleans the 'pages' column by removing non-numeric characters and converting it to an integer.
    - Converts the 'publishedDate' column to a string containing only the year.
    - Ensures the 'authors' column is a list of strings; converts improperly formatted entries.
    - Ensures the 'categories' column is a list of strings; converts improperly formatted entries.
    - Handles missing values:
        - Replaces missing string values with 'Unknown'.
        - Replaces missing 'price' values with 0.
    - Adds a new column 'expensive', marking books as expensive if their price is above the median.
    - Renames all columns by prefixing them with 'book_'.

    :param books_df: pd.DataFrame - The input DataFrame containing book data.
    :return: pd.DataFrame - The cleaned and transformed DataFrame with modified column names.
    """
        
    def clean_price(price):
        if isinstance(price, str):
            match = re.match(r'([0-9,.]+)', price)
            if match:
                return float(match.group(1).replace(',', '')) 
        return price
    
    def clean_pages(pages):
        if isinstance(pages, str):
            pages = re.sub(r'[^0-9]', '', pages)
        try:
            return int(pages)
        except ValueError:
            return np.nan 

    books_df['price'] = books_df['price'].apply(clean_price)
    books_df['price'] = pd.to_numeric(books_df['price'], errors='coerce')  

    books_df['authors'] = books_df['authors'].apply(lambda x: eval(x) if isinstance(x, str) else x)
    books_df['authors'] = books_df['authors'].apply(lambda x: x if isinstance(x, list) else [])

    books_df['categories'] = books_df['categories'].apply(lambda x: eval(x) if isinstance(x, str) else [])
    books_df['categories'] = books_df['categories'].apply(lambda x: x[0] if x else "unknown")

    books_df['publishedDate'] = books_df['publishedDate'].apply(lambda x: str(x)[:4] if pd.notnull(x) else np.nan)
    books_df["age"] = datetime.now().year - books_df["publishedDate"].astype(float)  
    books_df["age"] = books_df["age"].replace(np.nan, 0)

    books_df['publisher'] = books_df['publisher'].fillna('unknown')
    books_df['pages'] = books_df['pages'].apply(clean_pages)

    books_df['price'] = books_df['price'].fillna(0)
    median_price = books_df['price'].median()
    books_df['expensive'] = books_df['price'].apply(lambda x: x > median_price)

    books_df.columns = 'book_' + books_df.columns
    return books_df



def preprocess_customers_df(customers_df):
    """
    *note: There are people born in the future, for the initial cleaning I will leave them intentionaly

    Preprocesses a DataFrame containing customer data by cleaning and transforming specific columns.

    This function performs the following preprocessing steps:
    - Standardizes text columns ('education', 'occupation', 'gender') by stripping extra spaces and converting to lowercase.
    - Standardizes address-related columns ('street_address', 'state', 'city', 'name') by stripping extra spaces and converting to title case.
    - Cleans the 'zipcode' column by removing non-numeric characters and converting it to an integer.
    - Extracts the birth year from 'birth_date' and calculates the customer's age.
    - Groups customers into predefined age categories: '0-17', '18-35', '36-65', '65+'.
    - Leaves future birth dates unmodified in the initial cleaning stage.
    - Renames all columns by prefixing them with 'customer_'.

    :param customers_df: pd.DataFrame-The input DataFrame containing customer data.
    :return: pd.DataFrame-The cleaned and transformed DataFrame with modified column names.
    """
    def clean_zipcode(zipcode):
        if isinstance(zipcode, str):
            zipcode = re.sub(r'[^0-9]', '', zipcode)
        try:
            return int(zipcode)
        except ValueError:
            return np.nan 
    
    customers_df['education'] = customers_df['education'].apply(lambda x: ' '.join(x.split()).lower() if isinstance(x, str) else "unknown")
    customers_df['occupation'] = customers_df['occupation'].apply(lambda x: ' '.join(x.split()).lower() if isinstance(x, str) else "unknown")
    customers_df['street_address'] = customers_df['street_address'].apply(lambda x: ' '.join(x.split()).title()if isinstance(x, str) else "unknown")
    customers_df['state'] = customers_df['state'].apply(lambda x: ' '.join(x.split()).title()if isinstance(x, str) else "unknown")
    customers_df['city'] = customers_df['city'].apply(lambda x: ' '.join(x.split()).title()if isinstance(x, str) else "unknown")
    customers_df['name'] = customers_df['name'].apply(lambda x: ' '.join(x.split()).title() if isinstance(x, str) else "unknown")
    customers_df['gender'] = customers_df['gender'].apply(lambda x: ' '.join(x.split()).lower() if isinstance(x, str) else "unknown")
    customers_df['zipcode'] = customers_df['zipcode'].apply(clean_zipcode)
    customers_df['birth_date'] = customers_df['birth_date'].apply(lambda x: int(str(x)[:4]) if pd.notnull(x) else np.nan)
    today = pd.to_datetime('today')
    customers_df['age']  = customers_df['birth_date'].apply(lambda x: int(today.year) - x)
    bins = [0, 17, 35, 65, float('inf')]
    labels = ['0-17', '18-35', '36-65', '65+']
    customers_df['age_group'] = pd.cut(customers_df['age'], bins=bins, labels=labels, right=True)

    customers_df.columns = 'customer_' + customers_df.columns
    return customers_df


def preprocess_libraries_df(libraries_df):
    """
    Preprocesses a DataFrame containing library data by cleaning and transforming specific columns.

    - Standardize text columns:
        - 'city', 'street_address', and 'name' are converted to title case.
        - 'region' is converted to uppercase.
    - Clean the 'postal_code' column by removing non-numeric characters and converting it to an integer.
    - Rename all columns by prefixing them with 'library_'.

    :param libraries_df: pd.DataFrame-The input DataFrame containing library data.
    :return pd.DataFrame-The cleaned and transformed DataFrame with modified column names.
    """

    def clean_postal_code(postal_code):
        if isinstance(postal_code, str):
            postal_code = int(re.sub(r'[^0-9]', '', postal_code))
        try:
            return int(postal_code)
        except ValueError:
            return np.nan 
    
    libraries_df['city'] = libraries_df['city'].apply(lambda x: ' '.join(x.split()).title()if isinstance(x, str) else "unknown")
    libraries_df['street_address'] = libraries_df['street_address'].apply(lambda x: ' '.join(x.split()).title()if isinstance(x, str) else "unknown")
    libraries_df['name'] = libraries_df['name'].apply(lambda x: ' '.join(x.split()).title()if isinstance(x, str) else "unknown")
    libraries_df['region'] = libraries_df['region'].apply(lambda x: ' '.join(x.split()).upper()if isinstance(x, str) else "unknown")
    libraries_df['postal_code'] = libraries_df['postal_code'].apply(clean_postal_code)
    libraries_df.columns = 'library_' + libraries_df.columns
    return libraries_df


def preprocess_checkouts_df(checkouts_df):
    """
    Preprocesses a DataFrame containing book checkout records by cleaning and transforming date-related columns.
    - Converts 'date_checkout' and 'date_returned' columns to datetime format.
    - Strips whitespace from date strings before conversion.
    - Computes the number of days a book was borrowed.
    - Creates a 'late_return' column, marking records where the book was returned after 28 days (1 for late, 0 otherwise).
    - Removes records where 'date_returned' is earlier than 'date_checkout'.
    - Adds is_holiday_checkout and is_holiday_for_return
    :param checkouts_df: pd.DataFrame-The input DataFrame containing checkout records.
    :return pd.DataFrame-the cleaned and transformed DataFrame with additional columns for analysis.
    """
    checkouts_df['date_checkout'] = pd.to_datetime(checkouts_df['date_checkout'].str.strip(), format='mixed', errors='coerce')
    checkouts_df['date_returned'] = pd.to_datetime(checkouts_df['date_returned'].str.strip(), format='mixed', errors='coerce')
    checkouts_df['days_borrowed'] = (checkouts_df['date_returned'] - checkouts_df['date_checkout']).dt.days
    checkouts_df['late_return'] = (checkouts_df['days_borrowed'] > 28).astype(int)
    checkouts_df = checkouts_df[checkouts_df['days_borrowed'] >= 0]

    checkouts_df["checkout_month"] = checkouts_df["date_checkout"].dt.month
    checkouts_df["checkout_dayofweek"] = checkouts_df["date_checkout"].dt.dayofweek

    checkouts_df["return_due_date"] = checkouts_df['date_checkout'].apply(lambda x: x + pd.Timedelta(days=28))
    checkouts_df["return_due_date_dayofweek"] = checkouts_df["return_due_date"].dt.day_name()


    # Define holidays for a US
    us_holidays = holidays.US(years=checkouts_df['date_checkout'].dt.year.unique())

    # Check if each date is a holiday
    checkouts_df['is_holiday_checkout'] = checkouts_df['date_checkout'].apply(lambda x: x in us_holidays)
    checkouts_df['is_holiday_for_return'] = checkouts_df['date_checkout'].apply(lambda x: (x + pd.Timedelta(days=28)) in us_holidays)

    checkouts_df["date_checkout"] = pd.to_datetime(checkouts_df["date_checkout"], errors="coerce")
    checkouts_df = checkouts_df.dropna(subset=["date_checkout"]) 

    return checkouts_df


def merge_data(checkouts_df, customers_df, books_df, libraries_df):
    """
    - Merge `checkouts_df` with `customers_df` on 'patron_id' and 'customer_id'.
    - Merge the resulting DataFrame with `books_df` on 'id' and 'book_id'.
    - Merge the final DataFrame with `libraries_df` on 'library_id'.

    :param checkouts_df: pd.DataFrame-The DataFrame containing book checkout records.
    :param customers_df: pd.DataFrame-The DataFrame containing customer information.
    :param books_df: pd.DataFrame-The DataFrame containing book details.
    :param libraries_df: pd.DataFrame-The DataFrame containing library information.
    :return pd.DataFrame-A merged DataFrame containing checkout, customer, book, and library details.
    """
    data = checkouts_df.merge(customers_df, left_on='patron_id', right_on='customer_id')
    data = data.merge(books_df, left_on='id', right_on='book_id')
    data = data.merge(libraries_df, left_on='library_id', right_on='library_id')

    data = data[data.customer_age > 0]
    data = data[data.customer_age < 115]
    data = data[data.days_borrowed < 20000]

    data["same_city"] = data['customer_city'] == data['library_city']

    return data
