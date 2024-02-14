#------------------------LIBRAIRIES----------------------------
import streamlit as st
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame
from pyspark.sql.window import Window
from pyspark.sql.types import ByteType, ShortType, IntegerType, LongType, FloatType, DoubleType
from pyspark.sql.functions import col, unix_timestamp, lag, row_number, weekofyear, month, year, avg, min as spark_min, max as spark_max, dayofmonth, expr
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import functions as F
import datetime
from datetime import datetime, timedelta
import pandas as pd 
import plotly.graph_objects as go


#------------------------FUNCTIONS----------------------------

def create_spark_session():
    """Create a SparkSession"""
    spark = SparkSession.builder \
        .appName("Stocks Analysis") \
        .getOrCreate()
    return spark

def read_file_with_spark(spark, file_path, file_type):
    """
    Read file using Spark and return a DataFrame
    
    Arguments:
    - spark: SparkSession object
    - file_path: path to the file
    - file_type: type of file (csv or xlsx)
    """
    if file_type == 'csv':
        df = spark.read.csv(file_path, header=True, inferSchema=True)
    elif file_type == 'xlsx':
        df = spark.read.format("com.crealytics.spark.excel") \
                      .option("header", "true") \
                      .option("inferSchema", "true") \
                      .load(file_path)
    else:
        raise ValueError("Unsupported file type. Only CSV and XLSX files are supported.")
    return df

def explore_dataframe(df: DataFrame):
    """
    Explore the DataFrame by displaying column names and types, as well as numeric column statistics if available.

    Arguments:
    - df: PySpark DataFrame to explore.
    """
    show_column_type = st.checkbox('Click to see the column names and types.')
    if show_column_type:
        st.write("Column names and types:")
        for field in df.schema.fields:
            st.write(f"Column: {field.name}, Type: {field.dataType}")
    st.divider()
    numeric_cols = [field.name for field in df.schema.fields if isinstance(field.dataType, (ByteType, ShortType, IntegerType, LongType, FloatType, DoubleType))]
    if numeric_cols:
        st.write("Numeric column statistics:")
        st.table(df.select(numeric_cols).describe().toPandas())
    else:
        st.write("No numeric column in the dataframe.")

def show_first_last_rows(df: DataFrame):
    """
    Selects the first 40 rows and the last 40 rows of each company_name group from the DataFrame.

    Arguments:
    - df: PySpark DataFrame to select rows from.
    """
    window_spec = Window.partitionBy("company_name").orderBy("Date")
    df_with_row_number = df.withColumn("row_number", F.row_number().over(window_spec))
    first_40_rows = df_with_row_number.filter(F.col("row_number") <= 40).select(df.columns)
    total_rows_per_company = df_with_row_number.groupBy("company_name").agg(F.max("row_number").alias("total_rows"))
    last_40_rows = df_with_row_number.join(total_rows_per_company, "company_name") \
                                       .filter(F.col("row_number") > (F.col("total_rows") - 40)) \
                                       .select(df.columns)
    result_df = first_40_rows.unionByName(last_40_rows)
    return result_df

def get_num_observations(df: DataFrame):
    """
    Return the number of observations in a dataframe.
    
    Args:
    - df: DataFrame PySpark
    """
    return df.count()

def deduce_period(date1, date2):
    """
    Deduces the period between two dates and returns a string representation.

    Arguments:
    - date1: The start date.
    - date2: The end date.

    Returns:
    - A string representing the period between the two dates.
    """
    diff = (date2 - date1).days

    if diff == 1:
        return "1 day"
    elif diff == 7:
        return "1 week"
    elif diff == 30:
        return "1 month"
    elif diff == 365:
        return "1 year"
    else:
        weeks = diff // 7
        days = diff % 7
        if weeks > 0 and days > 0:
            return f"{weeks} weeks and {days} days"
        elif weeks > 0:
            return f"{weeks} weeks"
        else:
            return f"{days} days"

def explore_company(df: DataFrame):
    """
    Displays statistics of numeric columns for each company in the DataFrame.

    Arguments:
    - df: PySpark DataFrame containing data for multiple companies.
    """
    st.subheader('Discover the statistics of each company:')
    discover = st.button('Click here!')
    if discover:
        numeric_cols = [field.name for field in df.schema.fields if isinstance(field.dataType, (ByteType, ShortType, IntegerType, LongType, FloatType, DoubleType))]
        if numeric_cols:
            for company_row in df.select("company_name").distinct().collect():
                company_name = company_row['company_name']
                st.write(f"Numeric column statistics for company: {company_name}")
                company_df = df.filter(col("company_name") == company_name)
                stats_df = company_df.select(numeric_cols).describe().toPandas()
                st.table(stats_df)
        else:
            st.write("No numeric column in the dataframe.")

def count_missing_values(df: DataFrame):
    """
    Calculate the number of missing values for each column in the dataframe.

    Args:
    - df: DataFrame PySpark
    """
    missing_values = {}
    for column in df.columns:
        missing_count = df.where(col(column).isNull() | (col(column) == "")).count()
        missing_values[column] = missing_count
    return missing_values

def calculate_and_visualize_correlation(df: DataFrame):
    """
    Calculate and visualize the correlation between values of numeric columns.

    Args:
    - df: DataFrame PySpark
    """
    numeric_cols = [col_name for col_name, data_type in df.dtypes if data_type in ('int', 'double', 'float')]
    correlation_matrix = df.select([col(c).alias(c) for c in numeric_cols]).toPandas().corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title('Correlation Matrix')
    st.pyplot(fig)

def calculate_averages(df, period):
    """
    Calculates the average open and close prices based on the specified period.

    Arguments:
    - df: PySpark DataFrame containing stock data.
    - period: A string indicating the period for which averages are calculated ("Year", "Month", or "Week").

    Returns:
    - A DataFrame with calculated average open and close prices based on the specified period.
    """
    if period == "Year":
        df = df.groupBy(F.year("Date").alias("Year"), "company_name").agg(F.avg("Open").alias("Avg_Open"), F.avg("Close").alias("Avg_Close"))
    elif period == "Month":
        df = df.groupBy(F.year("Date").alias("Year"), F.month("Date").alias("Month"), "company_name").agg(F.avg("Open").alias("Avg_Open"), F.avg("Close").alias("Avg_Close")).orderBy("Month")
    elif period == "Week":
        df = df.groupBy(F.year("Date").alias("Year"), F.weekofyear("Date").alias("Week"), "company_name").agg(F.avg("Open").alias("Avg_Open"), F.avg("Close").alias("Avg_Close")).orderBy("Week")
    return df

def analyze_stock_price_changes(df, company_name, start_date, end_date):
    """
    Analyzes stock price changes for a specific company within a specified date range.

    Arguments:
    - df: PySpark DataFrame containing stock data.
    - company_name: Name of the company to analyze.
    - start_date: Start date of the analysis period.
    - end_date: End date of the analysis period.

    Returns:
    - DataFrame containing stock price changes for the specified company and date range.
    """
    df_filtered = df.filter((F.col("company_name") == company_name) & (F.col("Date").between(start_date, end_date)))
    
    df_changes = df_filtered.withColumn("Price_Diff_Day", F.col("Close") - F.lag("Close", 1).over(Window.orderBy("Date")))
    df_changes = df_changes.withColumn("Price_Diff_Month", F.col("Close") - F.lag("Close", 30).over(Window.orderBy("Date")))
    
    return df_changes

def calculate_daily_returns(df):
    """
    Calculates the daily returns based on the closing and opening prices.

    Arguments:
    - df: PySpark DataFrame containing stock data.

    Returns:
    - DataFrame with a new column "Daily_Return" representing the daily returns.
    """
    df = df.withColumn("Daily_Return", ((F.col("Close") - F.col("Open")) / F.col("Open")) * 100)
    return df

def stocks_highest_daily_return(df):
    """
    Find the stocks with the highest daily return.

    Parameters:
    - df: Spark DataFrame containing stock data

    Returns:
    - stocks_with_highest_return: DataFrame containing stocks with the highest daily return
    """
    df = calculate_daily_returns(df)
    max_daily_return = df.agg(F.max("Daily_Return")).collect()[0][0]
    stocks_with_highest_return = df.filter(df["Daily_Return"] == max_daily_return)

    return stocks_with_highest_return

def calculate_average_daily_return(df, selected_companies, selected_year, period):
    """
    Calculates the average daily return for selected companies within a specified year and period.

    Arguments:
    - df: PySpark DataFrame containing stock data.
    - selected_companies: List of company names to include in the calculation.
    - selected_year: The selected year for the calculation.
    - period: A string indicating the period for the average daily return calculation ("Year", "Month", or "Week").

    Returns:
    - DataFrame with the average daily returns for the selected companies, year, and period.
    """
    df = calculate_daily_returns(df)
    df = df.filter(df["company_name"].isin(selected_companies))
    df = df.filter(F.year("Date") == selected_year)
    if period == "Year":
        avg_return_period = df.groupBy(F.year("Date").alias("Year"), "company_name").agg(F.avg("Daily_Return").alias("Avg_Return_Year"))
    elif period == "Month":
        avg_return_period = df.groupBy(F.year("Date").alias("Year"), F.month("Date").alias("Month"), "company_name").agg(F.avg("Daily_Return").alias("Avg_Return_Month")).orderBy("Year", "Month")
    elif period == "Week":
        avg_return_period = df.groupBy(F.year("Date").alias("Year"), F.weekofyear("Date").alias("Week"), "company_name").agg(F.avg("Daily_Return").alias("Avg_Return_Week")).orderBy("Year", "Week")
    else:
        raise ValueError("Invalid period. Please select 'Year', 'Month', or 'Week'.")

    return avg_return_period

def calculate_moving_average(df, company_name, price_type, start_date, num_periods):
    """
    Calculates the moving average for a specified company, price type, start date, and number of periods.

    Arguments:
    - df: PySpark DataFrame containing stock data.
    - company_name: The name of the company for which the moving average is calculated.
    - price_type: The type of price to consider ("Open" or "Close").
    - start_date: The start date from which the moving average calculation begins.
    - num_periods: The number of periods to consider for the moving average.

    Returns:
    - The calculated moving average.
    """
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d")

    day_before_start_date = start_date - timedelta(days=1)
    df_filtered = df.filter(F.col("company_name") == company_name)
    df_filtered = df_filtered.filter(F.col("Date") <= day_before_start_date)
    df_filtered = df_filtered.orderBy(F.col("Date").desc())
    
    windowSpec = Window.orderBy(F.col("Date").desc())
    df_filtered = df_filtered.withColumn("rank", F.dense_rank().over(windowSpec))

    df_filtered = df_filtered.filter(F.col("rank") <= num_periods)
    #st.write(df_filtered)
    if price_type == "Open":
        price_column = F.col("Open")
    elif price_type == "Close":
        price_column = F.col("Close")
    else:
        raise ValueError("Invalid price type. Please select 'Open' or 'Close'.")
    
    moving_average = df_filtered.agg(F.avg(price_column).alias("moving_average")).collect()[0]["moving_average"]
    
    return moving_average

def calculate_correlation(stock1_values, stock2_values):
    """
    Calculate the correlation coefficient between two sets of stock values using PySpark.

    Arguments:
    - stock1_values: PySpark DataFrame containing the values of the first stock.
    - stock2_values: PySpark DataFrame containing the values of the second stock.

    Returns:
    - The correlation coefficient between the two sets of stock values.
    """
    joined_df = stock1_values.join(stock2_values, on='Date', how='inner')
    correlation_coefficient = joined_df.select(F.corr('Value', 'Value')).collect()[0][0]

    return correlation_coefficient

def calculate_return_rate(df, start_date, selected_period):
    """
    Calculates the return rate for all companies based on the selected period.

    Arguments:
    - df: PySpark DataFrame containing stock data.
    - start_date: The selected start date for the calculation.
    - selected_period: A string indicating the period for the return rate calculation ("Month", or "Week").

    Returns:
    - DataFrame with the calculated return rate for all companies based on the selected period.
    """
    if selected_period == 'Year':
        previous_year = start_date.year - 1
        filtered_df = df.filter((F.year(df['Date']) == previous_year) & (df['Date'] < start_date))
        grouped_df = filtered_df.groupBy('company_name') \
                                .agg(((F.last('Close') - F.first('Open')) / F.first('Open')).alias('Return_Rate'))
        return grouped_df


    elif selected_period == 'Month':
        previous_month = start_date.replace(day=1) - timedelta(days=1)
        filtered_df = df.filter((F.year(df['Date']) == previous_month.year) & (F.month(df['Date']) == previous_month.month) & (df['Date'] < start_date))
        grouped_df = filtered_df.groupBy('company_name') \
                                 .agg(((F.last('Close') - F.first('Open')) / F.first('Open')).alias('Return_Rate'))
        return grouped_df

    elif selected_period == 'Week':
        previous_week = start_date - timedelta(weeks=1)
        filtered_df = df.filter((F.year(df['Date']) == previous_week.year) & (F.weekofyear(df['Date']) == previous_week.isocalendar()[1]) & (df['Date'] < start_date))
        grouped_df = filtered_df.groupBy('company_name') \
                                 .agg(((F.last('Close') - F.first('Open')) / F.first('Open')).alias('Return_Rate'))
        return grouped_df

    else:
        raise ValueError("Invalid selected period")

def calculer_rsi(dataframe, company_name, selected_year, window_size=14):
    """
    Calculate the RSI (Relative Strength Index) for a given company and selected year.

    Arguments:
    - dataframe: PySpark DataFrame containing stock data.
    - company_name: The name of the company for which the RSI is calculated.
    - selected_year: The selected year for the calculation.
    - window_size: The window size for RSI calculation (default: 14).

    Returns:
    - DataFrame containing the calculated RSI for the specified company.
    """
    filtered_df = dataframe.filter((dataframe['company_name'] == company_name) & (F.year(dataframe['Date']) == selected_year))
    delta_close = filtered_df['Close'] - F.lag(filtered_df['Close'], 1).over(Window.partitionBy('company_name').orderBy('Date'))

    gains = F.when(delta_close > 0, delta_close).otherwise(0)
    pertes = F.when(delta_close < 0, -delta_close).otherwise(0)
    sum_gains = F.sum(gains).over(Window.partitionBy('company_name').orderBy('Date').rowsBetween(-window_size, Window.currentRow))
    sum_pertes = F.sum(pertes).over(Window.partitionBy('company_name').orderBy('Date').rowsBetween(-window_size, Window.currentRow))

    avg_gains = sum_gains / window_size
    avg_pertes = sum_pertes / window_size

    rs = avg_gains / avg_pertes
    rsi = 100 - (100 / (1 + rs))

    dataframe_with_rsi = filtered_df.withColumn('RSI', rsi)

    return dataframe_with_rsi

def calculate_bollinger_bands(df, company_name, price_type, start_date, num_periods):
    """
    Calculate Bollinger Bands for a specified company, price type, start date, and number of periods.

    Arguments:
    - df: PySpark DataFrame containing stock data.
    - company_name: The name of the company for which Bollinger Bands are calculated.
    - price_type: The type of price to consider ("Open" or "Close").
    - start_date: The start date for the calculation.
    - num_periods: The number of periods to consider for the calculation.

    Returns:
    - DataFrame containing Bollinger Bands for the specified company and parameters.
    """
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d")

    day_before_start_date = start_date - timedelta(days=1)

    df_filtered = df.filter(F.col("company_name") == company_name)
    df_filtered = df_filtered.filter(F.col("Date") <= day_before_start_date)
    
    windowSpec = Window.partitionBy("company_name").orderBy(F.col("Date").desc())
    df_filtered = df_filtered.withColumn("rank", F.dense_rank().over(windowSpec))
    df_filtered = df_filtered.filter(F.col("rank") <= num_periods)

    if price_type == "Open":
        price_column = F.col("Open")
    elif price_type == "Close":
        price_column = F.col("Close")
    else:
        raise ValueError("Invalid price type. Please select 'Open' or 'Close'.")

    df_filtered = df_filtered.withColumn("moving_average", F.avg(price_column).over(windowSpec))
    df_filtered = df_filtered.withColumn("std_dev", F.stddev(F.col(price_type)).over(windowSpec))
    df_filtered = df_filtered.withColumn("upper_band", F.col("moving_average") + 2 * F.col("std_dev"))
    df_filtered = df_filtered.withColumn("lower_band", F.col("moving_average") - 2 * F.col("std_dev"))

    return df_filtered.filter(F.col("Date") <= start_date).orderBy(F.col("Date").desc()).limit(num_periods + 1)

def calculate_fibonacci_levels(df, company_name):
    """
    Calculate Fibonacci levels for a specific company based on its high and low prices.

    Arguments:
    - df: PySpark DataFrame containing stock data.
    - company_name: The name of the company for which Fibonacci levels are calculated.

    Returns:
    - Tuple containing the DataFrame filtered by the specified company and a dictionary
      of Fibonacci levels.
    """
    df_company = df.filter(F.col("company_name") == company_name)

    highest_price = df_company.agg(F.max("High")).collect()[0][0]
    lowest_price = df_company.agg(F.min("Low")).collect()[0][0]

    levels = {
        "23.6%": lowest_price + (highest_price - lowest_price) * 0.236,
        "38.2%": lowest_price + (highest_price - lowest_price) * 0.382,
        "50.0%": lowest_price + (highest_price - lowest_price) * 0.5,
        "61.8%": lowest_price + (highest_price - lowest_price) * 0.618,
        "100.0%": highest_price
    }

    return df_company, levels

def calculate_volume_ratio(df, company_name, target_date, start_date, end_date):
    """
    Calculate the volume ratio for a specific company between a target date and a specified period.

    Arguments:
    - df: PySpark DataFrame containing stock data.
    - company_name: The name of the company for which the volume ratio is calculated.
    - target_date: The target date for volume measurement.
    - start_date: The start date of the period for average volume calculation.
    - end_date: The end date of the period for average volume calculation.

    Returns:
    - The volume ratio for the specified company and dates.
    """
    target_date = target_date.strftime('%Y-%m-%d')
    start_date = start_date.strftime('%Y-%m-%d')
    end_date = end_date.strftime('%Y-%m-%d')

    df_company = df.filter(col("company_name") == company_name)

    target_volume_data = df_company.filter(col("Date") == target_date).select("Volume").collect()
    if target_volume_data:
        target_volume = target_volume_data[0][0]
    else:
        st.error(f"No volume data available for company {company_name} on target date {target_date}")
        return None

    average_volume_data = df_company.filter((col("Date") >= start_date) & (col("Date") <= end_date)) \
        .agg(avg(col("Volume"))).collect()
    if average_volume_data:
        average_volume = average_volume_data[0][0]
    else:
        st.error(f"No average volume data available for company {company_name} between {start_date} and {end_date}")
        return None

    volume_ratio = target_volume / average_volume

    return volume_ratio

def calculate_support_resistance(df):
    """
    Calculate support and resistance levels based on historical stock data.

    Arguments:
    - df: PySpark DataFrame containing stock data.

    Returns:
    - Tuple of lists containing support levels and resistance levels, each with Date and corresponding Low/High values.
    """
    windowSpec = Window.orderBy('Date')
    df = df.withColumn('prev_low', lag(df['Low']).over(windowSpec))
    df = df.withColumn('prev_high', lag(df['High']).over(windowSpec))

    df = df.withColumn('support_filter', expr("Low < prev_low AND Low < lag(prev_low, 1) OVER (ORDER BY Date) AND Low < lag(Low, 2) OVER (ORDER BY Date)"))
    df = df.withColumn('resistance_filter', expr("High > prev_high AND High > lag(prev_high, 1) OVER (ORDER BY Date) AND High > lag(High, 2) OVER (ORDER BY Date)"))

    support_levels = df.filter(df['support_filter']).select('Date', 'Low').collect()
    resistance_levels = df.filter(df['resistance_filter']).select('Date', 'High').collect()

    return support_levels, resistance_levels

def calculate_money_flow_indicator(df):
    """
    Calculate the Money Flow Index (MFI) indicator based on typical price and volume.

    Arguments:
    - df: PySpark DataFrame containing stock data.

    Returns:
    - DataFrame with the Money Flow Index (MFI) indicator added as a new column.
    """
    typical_price = (F.col('High') + F.col('Low') + F.col('Close')) / 3
    money_flow = typical_price * F.col('Volume')

    positive_money_flow = F.when(typical_price > F.lag(typical_price).over(Window.partitionBy().orderBy('Date')), money_flow).otherwise(0)
    negative_money_flow = F.when(typical_price < F.lag(typical_price).over(Window.partitionBy().orderBy('Date')), money_flow).otherwise(0)

    positive_money_flow_sum = F.sum(positive_money_flow).over(Window.orderBy('Date'))
    negative_money_flow_sum = F.sum(negative_money_flow).over(Window.orderBy('Date'))
    money_flow_ratio = positive_money_flow_sum / negative_money_flow_sum

    money_flow_index = 100 - (100 / (1 + money_flow_ratio))

    return df.withColumn('Money_Flow_Index', F.when(F.isnan(money_flow_index), 50).otherwise(money_flow_index))

#------------------------------PAGES---------------------------------

def explore_page(df):
    st.header("Let's discover the data!")
    
    # Explore the available columns, their statistics and types
    explore_dataframe(df)
    st.write('In this table, we have the statistics for all the columns of the file "STOCKS.csv". This file contains data about 7 companies : AMAZON, FACEBOOK, TESLA, GOOGLE, MICROSOFT, ZOOM and APPLE.')

    st.divider()
    
    # Show the first and last 40 rows of each stock price
    st.subheader("First and Last 40 rows of each stock price:")
    result_df = show_first_last_rows(df)
    result_pandas_df = result_df.toPandas()
    st.write(result_pandas_df)
    st.write('For each company in the dataset, we have collected the first and last 40 rows. As we have 7 companies, we obtain a table with 560 rows.')

    st.divider()

    # Get the number of observations
    st.subheader('Check the number of observations in the dataframe:')
    num_observations = get_num_observations(df)
    st.write(f"Total number of observations: {num_observations}")

    st.divider()

    # Deduce programmatically what is the period you have between the data points : for example, if you have data point with the following date [01/01, 02/01, .....], you shoud have a function that will analyse the difference between the dates automatically and deduce it is a day period
    st.subheader('Timestamp between two dates')
    st.write('It can be useful to deduce the timestamp between two dates to know which period we are working on : one week, month or year.')
    start_date = st.date_input("Select start date")
    end_date = st.date_input("Select end date")
    if start_date and end_date:
        period = deduce_period(start_date, end_date)
        st.write(f"The period between {start_date} and {end_date} is: {period}")
    else:
        st.write("Please select two valid dates.")

    st.divider()

    # Descriptive statistics for each dataframe and each column (min, max, standard deviation)
    st.write('Same as before with the statistics of the dataset but this time we want to see the statistics for each company.')
    explore_company(df)

    st.divider()

    # Number of missing values for each dataframe and column
    st.subheader('How many missing values in the dataset?')
    missing = st.button('Missing values')
    if missing :
        missing_values_dict = count_missing_values(df)
        st.write("Missing values for each column:")
        for column, count in missing_values_dict.items():
            st.write(f"{column}: {count}")
        st.write('There is no missing value in the dataset.')

    st.divider()

    # Correlation between values
    st.subheader("Is there any correlation between the data?")
    calculate_and_visualize_correlation(df)
    st.write('We can see that the variables are totally correlated except the value of the volume. It means that each value depends on each other, except the value of the volume.')

def avg_prices_page(df):
    st.header("Average of the opening and closing prices")
    # What is the average of the opening and closing prices for each stock price and for different time periods (week, month, year)
    st.subheader("Stock Price Trends:")
    st.markdown("*Explanation:*")
    st.markdown("*Analyzing the average of opening and closing prices of stocks provides valuable insights into market trends, volatility, and sentiment. Traders use these averages to develop trading strategies, identify support and resistance levels, and assess relative performance against benchmarks. The relationship between opening and closing prices helps investors gauge market sentiment and make informed decisions about buying or selling stocks. Overall, monitoring these averages is essential for understanding market dynamics and making effective investment choices.*")
    distinct_companies = sorted([row.company_name for row in df.select("company_name").distinct().collect()])
    st.warning('If you select ZOOM, data will only be available for years 2019 (April) and 2020. You will not have data if you select anterior years.')
    selected_companies = st.multiselect("Select one or more companies", distinct_companies)
    selected_price_type = st.radio("Select price type", ("Open", "Close"))
    df = df.withColumn("Date", F.to_date("Date", "yyyy-MM-dd"))
    distinct_years = sorted([row.Year for row in df.select(F.year("Date").alias("Year")).distinct().collect()])
    selected_year = st.selectbox("Select year", distinct_years)
    period = st.selectbox("Select period", ["Year", "Month", "Week"])
    df_filtered = df.filter((F.year("Date") == selected_year) & (col("company_name").isin(selected_companies)))
    df_averages = calculate_averages(df_filtered, period)
    if period != "Year":
        #st.write(df_averages.toPandas())
        show_chart = st.checkbox("Show Line Chart")
        if show_chart:
            st.write('On this graph you can visualize trends for one or more companies depending on your choice before. This line represents the trend of the type price you have choose.')
            st.write('If you select APPLE or ZOOM, you will observe that the prices are not really high compared to the prices of the other companies.')
            df_pandas = df_filtered.toPandas()
            chart_data = {}
            for company in selected_companies:
                df_company = df_pandas[df_pandas['company_name'] == company]
                chart_data[f'{company} {selected_price_type}'] = df_company.set_index('Date')[selected_price_type]
            chart_df = pd.DataFrame(chart_data)
            st.line_chart(chart_df)
    else:
        st.write(df_averages.toPandas())
        st.write('When you select the period "Year", there is an average of all the values of the year and that is why you obtain a single value by company.')

def stock_prices_change_page(df):
    st.header('Stock prices changes')
    st.markdown('*Explanation:*')
    st.markdown('*Analyzing stock price changes is crucial for investors to evaluate past performance, identify market trends, manage risk, and make informed investment decisions. By monitoring fluctuations in stock prices, investors can assess the direction of the market and adjust their investment strategies accordingly. Understanding price changes also plays a key role in technical analysis, where patterns and indicators derived from price movements help forecast future market trends and guide trading decisions.*')
    # How do the stock prices change day to day and month to month (may be you can create new columns to save those calculations)
    distinct_companies = sorted([row.company_name for row in df.select("company_name").distinct().collect()])
    selected_company = st.selectbox("Select company", distinct_companies)

    available_dates_df = df.filter(col("company_name") == selected_company).select("Date").distinct().collect()
    available_dates = [str(row.Date) for row in available_dates_df]
    available_dates_dt = [datetime.strptime(date, "%Y-%m-%d").date() for date in available_dates]

    min_date = min(available_dates_dt) if available_dates_dt else datetime.date.today()
    max_date = max(available_dates_dt) if available_dates_dt else datetime.date.today()

    start_date_default = min_date
    end_date_default = max_date

    start_date = st.date_input("Select start date", start_date_default, min_value=min_date, max_value=max_date, key="start")
    end_date = st.date_input("Select end date", end_date_default, min_value=min_date, max_value=max_date, key="end")

    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")

    df_changes = analyze_stock_price_changes(df, selected_company, start_date_str, end_date_str)

    start_close = df_changes.filter(col("Date") == start_date_str).select("Close").collect()
    end_close = df_changes.filter(col("Date") == end_date_str).select("Close").collect()

    if start_close and end_close:
        start_close_value = start_close[0][0]
        end_close_value = end_close[0][0]
        absolute_change = end_close_value - start_close_value
        percentage_change = (absolute_change / start_close_value) * 100
        st.write("Stock Price Changes Analysis:")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Close (Start Date)", f"{start_close_value:.2f} USD")
            st.metric("Close (End Date)", f"{end_close_value:.2f} USD")
        with col2:
            st.metric(label="Close Value Change", value=f"{absolute_change:.2f} USD", delta=f"{percentage_change:.2f}%")
    else:
        st.write("No data available for the selected company and date range. Choose different dates")

def daily_return_page(df):
    st.header('Daily return')
    st.subheader('Calculate Daily Return')
    st.markdown('*Explanation:*')
    st.markdown("*Calculating daily returns is essential in finance to assess investment performance, manage risk, and make informed decisions. Daily returns quantify the change in an asset's value from one trading day to the next, aiding in the evaluation of volatility and potential gains or losses. Investors and analysts rely on daily return data to monitor portfolio performance, compare against benchmarks, and develop effective trading strategies tailored to short-term market movements.*")
    distinct_companies = sorted([row.company_name for row in df.select("company_name").distinct().collect()])
    selected_companies = st.multiselect("Select company(s)", distinct_companies)
    distinct_years = sorted([row.year for row in df.select(F.year("Date").alias("year")).distinct().collect()])
    selected_years = st.multiselect("Select year(s)", distinct_years)
    df_filtered = df.filter(df["company_name"].isin(selected_companies) & F.year("Date").isin(selected_years))
    df_selected_years = calculate_daily_returns(df_filtered)
    df_chart = df_selected_years.select("Date", "Daily_Return", "company_name", F.year("Date").alias("year")).toPandas()
    df_chart['Daily_Return'] = pd.to_numeric(df_chart['Daily_Return'], errors='coerce')

    if st.button("See the graph"):
        plt.figure(figsize=(12, 8))
        colors = sns.color_palette("husl", len(selected_companies) * len(selected_years))
        for idx, company in enumerate(selected_companies):
            for year in selected_years:
                df_company_year = df_chart[(df_chart['company_name'] == company) & (df_chart['year'] == year)]
                plt.plot(df_company_year['Date'], df_company_year['Daily_Return'], label=f"{company} - {year}", color=colors.pop(0))

        plt.title('Daily Returns for Selected Companies and Years')
        plt.xlabel('Date')
        plt.ylabel('Daily Return (%)')
        plt.xticks(rotation=45)
        plt.legend(loc='upper left')
        plt.grid(True)
        st.pyplot(plt)

    st.divider()

    # What are the stocks with the highest daily return
    st.subheader('Highest Daily Return')
    st.markdown('*Explanation:*')
    st.markdown("*Calculating the highest daily return allows investors to pinpoint moments when asset prices experienced the most significant fluctuations within a single trading day. This metric is critical for assessing market volatility, identifying short-term trading opportunities, and managing investment risk. By understanding extreme price movements, investors can adjust their strategies and make informed decisions in a dynamic financial environment.*")
    st.write('Which company has the highest daily return ?')
    distinct_years = sorted([row.year for row in df.select(F.year("Date").alias("year")).distinct().collect()])
    selected_year = st.selectbox("Select a year", distinct_years)
    df_filtered = df.filter(F.year("Date").isin(selected_year))
    highest_return_stocks = stocks_highest_daily_return(df_filtered)
    st.table(highest_return_stocks)
    st.write('We can see that for each year, TELSA is the company with the highest daily return.')

    st.divider()

    # Calculate the average daily return for different periods (week, month, and year)
    st.subheader('Average Daily Return')
    st.markdown('*Explanation:*')
    st.markdown('*Calculating the daily average return of companies provides insights into their daily performance and volatility, aiding investors in assessing risk and comparing performance against industry benchmarks. It serves as a crucial metric for portfolio management, enabling investors to optimize asset allocation and balance risk within their portfolios. Additionally, the daily average return helps inform investment decisions and forecast future performance based on historical trends.*')
    distinct_companies = sorted([row.company_name for row in df.select("company_name").distinct().collect()])
    selected_companies = st.multiselect("Select companies", distinct_companies)
    st.warning('If you select ZOOM, data will only be available for years 2019 (April) and 2020. You will not have data if you select anterior years.')
    distinct_years = sorted([row.Year for row in df.select(F.year("Date").alias("Year")).distinct().collect()])
    selected_year = st.selectbox("Select year", distinct_years)

    period = st.selectbox("Select period", ["Year", "Month", "Week"])

    df_averages = calculate_average_daily_return(df, selected_companies, selected_year, period)

    if period == "Year":
        st.table(df_averages.toPandas())
        st.write('When you select the period "Year", there is an average of all the values of the year and that is why you obtain a single value by company.')
    else:
        show_chart = st.checkbox("Show Line Chart")
        if show_chart:
            st.write('On this graph you can visualize the average daily return for one or more companies depending on your choice before.')
            df_pandas = df_averages.toPandas()
            chart_data = {}
            for company in selected_companies:
                df_company = df_pandas[df_pandas['company_name'] == company]
                chart_data[f'{company} Avg_Return'] = df_company.set_index(period.capitalize())[f'Avg_Return_{period.capitalize()}']
            chart_df = pd.DataFrame(chart_data)
            st.line_chart(chart_df)

def moving_average_page(df):
    # Code a function that take as input a dataframe, a column name, the number of points to consider for the moving average (5 in the example) and add a new column to the dataframe with the values of calculated moving average
    st.header("Lagging indicator : Moving average")
    st.markdown('*Explanation:*')
    st.markdown('*The moving average is a technical analysis tool used to smooth out price fluctuations and identify trends in financial markets. It calculates the average price of an asset over a specified period to filter out short-term noise and highlight longer-term trends. Investors use moving averages to spot potential buy or sell signals, gauge trend strength, and set entry and exit points in trading strategies.*')

    distinct_companies = sorted([row.company_name for row in df.select("company_name").distinct().collect()])
    company_name = st.selectbox("Select company", distinct_companies)
    price_type = st.radio("Select price type", ["Open", "Close"])
    num_periods = st.number_input("Enter number of periods", min_value=2, step=1)

    available_dates = df.filter(df['company_name'] == company_name).select('Date').distinct().collect()
    available_dates = [row['Date'] for row in available_dates]

    st.warning('By default, you have the first date present in the dataset, but you have to choose an other date to calculate the moving average depending on the period chosen.')
    default_start_date = min(available_dates)
    start_date = st.date_input("Select a start date", min_value=min(available_dates), max_value=max(available_dates), value=default_start_date)

    moving_average = calculate_moving_average(df, company_name, price_type, start_date, num_periods)
    st.write(f"Moving average for {price_type} price of {company_name} starting from {start_date}: {moving_average}")

def return_rate_page(df):
    st.header('Return rate')
    st.markdown('*Explanation:*')
    st.markdown("*The return rate measures the percentage change in the value of an investment over a specific period, typically expressed on an annualized basis. It serves as a key metric for evaluating investment performance and comparing returns across different assets or portfolios. Investors use the return rate to assess the profitability of their investments and make informed decisions about allocation and strategy adjustments.*")

    st.warning('By default, the date is the first of the dataset, but as you need to select a period to calculate the return rate, choose another date.')
    available_dates = df.select('Date').distinct().collect()
    available_dates = [row['Date'] for row in available_dates]
    default_start_date = min(available_dates)  
    start_date = st.date_input("Select a start date", min_value=min(available_dates), max_value=max(available_dates), value=default_start_date)

    selected_period = st.selectbox("Select a period", ["Year", "Month", "Week"])

    if st.button("See the results"):
        st.write('Negative values indicate a decrease in value or a negative return, while positive values represent an increase in value or a positive return.')
        return_rate_df = calculate_return_rate(df, start_date, selected_period)
        pandas_df = return_rate_df.toPandas()

        max_return_rate = pandas_df['Return_Rate'].max()
        def style_return(value):
            if value == max_return_rate:
                return 'background-color: #60E66E'
            elif value < 0:
                return 'background-color: #F4C1BC'
            elif value > 0:
                return 'background-color: #C1F4BC'
            else:
                return ''
        st.dataframe(pandas_df.style.applymap(style_return, subset=['Return_Rate']))

def insights_page(df):
    st.header('More Insights about data...')
    st.subheader('Select the company you want to study in this page')
    st.warning('If you select ZOOM, data will only be available for years 2019 (April) and 2020. You will not have data if you select anterior years.')
    selected_company = st.selectbox("Choose a company", sorted([row.company_name for row in df.select("company_name").distinct().collect()]))

    st.divider()

    # RSI INDICATOR
    st.subheader("Leading indicator : RSI")
    st.markdown('*Explanation:*')
    st.markdown('*The Relative Strength Index (RSI) is a momentum oscillator that measures the speed and change of price movements. It provides insights into whether a stock or other asset is overbought or oversold. RSI values range from 0 to 100, with readings above 70 typically indicating overbought conditions, while readings below 30 suggest oversold conditions. Traders and analysts often use RSI to identify potential trend reversals and to confirm the strength of a prevailing trend.*')
    available_years = sorted([int(row.year) for row in df.filter(df['company_name'] == selected_company)
                              .select(F.year("Date").alias("year")).distinct().collect()])
    selected_year = st.slider("Select a year", min_value=min(available_years), max_value=max(available_years))
    if st.button("Calculate RSI"):
        st.write('Green : RSI considered as oversold (<30)')
        st.write('Red : RSI considered as overbought (>70)')
        st.write("In the financial context, such as that of technical indicators like the Relative Strength Index (RSI), None values can occur when there is insufficient historical data to perform a valid calculation. For instance, if an indicator requires the closing prices of the preceding 14 days to be computed, None values may appear for the initial 14 days of your dataset because there isn't enough previous data available. Similarly, None values may also result from missing or corrupted data within your original dataset.")
        rsi_data = calculer_rsi(df, selected_company, selected_year)
        pandas_df = rsi_data.toPandas()
        def style_rsi(value):
            if value > 70:
                return 'background-color: #ff6666'
            elif value < 30:
                return 'background-color: #66ff66'
            else:
                return ''
        st.dataframe(pandas_df.style.applymap(style_rsi, subset=['RSI']))

    st.divider()

    # BOLLINGER BANDS INDICATOR
    st.subheader("Lagging indicator : Bollinger Bands")
    st.markdown('*Explanation:*')
    st.markdown('*Bollinger Bands are a technical analysis tool that consists of a moving average line accompanied by two price channels plotted above and below it. The width of the bands expands and contracts based on the volatility of the market. Traders often use Bollinger Bands to identify potential overbought or oversold conditions and to gauge the likelihood of price reversals.*')
    price_type = st.radio("Select the price type", ["Open", "Close"])
    st.write('Bollinger Bands are interesting to analyze over fairly long periods. Traders typically use periods of 50, 100, or 200 days.')
    num_periods = st.number_input("Enter the number of periods", min_value=3, step=1)
    st.warning('By default, the date is the first of the dataset, but as you need to select a period to calculate the return rate, choose another date.')
    available_dates = df.filter(df['company_name'] == selected_company).select('Date').distinct().collect()
    available_dates = [row['Date'] for row in available_dates]
    default_start_date = min(available_dates) 
    start_date = st.date_input("Select a start date", min_value=min(available_dates), max_value=max(available_dates), value=default_start_date)
    df_bollinger = calculate_bollinger_bands(df, selected_company, price_type, start_date, num_periods)
    #st.write(df_bollinger)
    df_bollinger_pd = df_bollinger[['Date', 'upper_band', 'lower_band', 'moving_average']].toPandas()
    st.line_chart(df_bollinger_pd.set_index('Date'))

    st.divider()

    #FIBONACCI INDICATOR
    st.subheader('Fibonacci Levels')
    st.markdown('*Explanation:*')
    st.markdown('*The Fibonacci retracement levels are a technical analysis tool based on the Fibonacci sequence. These levels indicate potential support and resistance areas in a price trend. Traders use Fibonacci retracements to identify possible reversal points or areas of price continuation within financial markets.*')
    df = df.withColumn("Date", F.to_date(df["Date"], "yyyy-MM-dd"))
    distinct_years = sorted([row.Year for row in df.select(F.year("Date").alias("Year")).distinct().collect()])
    selected_year = st.selectbox("Select year", distinct_years)
    df_filtered = df.filter((F.col('company_name') == selected_company) & (F.year(F.col('Date')) == selected_year))

    df_company, levels = calculate_fibonacci_levels(df_filtered, selected_company)
    df_company_pd = df_company.toPandas()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_company_pd['Date'], y=df_company_pd['Close'], mode='lines', name='Close price'))
    for level, price in levels.items():
        fig.add_hline(y=price, line_dash="dot", annotation_text=level, annotation_position="bottom right")
    fig.update_layout(title=f'Fibonacci levels for {selected_company} in {selected_year}',
                    xaxis_title='Date',
                    yaxis_title='Price',
                    xaxis_rangeslider_visible=False)
    st.plotly_chart(fig)

    st.divider()

    #VOLUME RATIO INDICATOR
    st.subheader("Volume Ratio")
    st.markdown('*Explanation:*')
    st.markdown('*The volume ratio indicator is a technical analysis tool used to assess trading activity relative to historical volume levels. It compares the current trading volume to a reference volume, often the average volume over a specific period. It used to identify potential shifts in market sentiment or to confirm the strength of price movements.*')
    available_dates = df.filter(df['company_name'] == selected_company).select('Date').distinct().collect()
    available_dates = [row['Date'] for row in available_dates]
    default_date = min(available_dates)
    target_date = st.date_input("Select a target date:", min_value=min(available_dates), max_value=max(available_dates), value=default_date)
    start_date = st.date_input("Select a start date for the period:", min_value=min(available_dates), max_value=max(available_dates), value=default_date)
    end_date = st.date_input("Select a end date for the period:", min_value=min(available_dates), max_value=max(available_dates), value=default_date)
    if end_date <= start_date:
        st.error("End date must be after the start date. Please select a valid end date.")
    if st.button("Calculate volume ratio"):
        volume_ratio = calculate_volume_ratio(df, selected_company, target_date, start_date, end_date)
        st.write(f"The volume ratio for the company {selected_company} at date {target_date} is: {volume_ratio}")

    st.divider()

    # SUPPORT RESISTANCE INDICATOR
    st.subheader('Support and resistance levels')
    st.markdown('*Explanation:*')
    st.markdown('*The support and resistance levels are key concepts in technical analysis used to identify price levels where a financial asset is likely to encounter buying (support) or selling (resistance) pressure. Support represents a price level where buying interest is expected to outweigh selling pressure, preventing the price from falling further. Resistance, on the other hand, represents a price level where selling interest is expected to outweigh buying pressure, preventing the price from rising further.*')
    unique_years = sorted([row.Year for row in df.select(F.year("Date").alias("Year")).distinct().collect()])
    select_year = st.selectbox("Select a year", unique_years)

    df_filtered = df.filter((F.col('company_name').isin(selected_company)) & (F.year(F.col('Date')) == select_year))
    support_levels, resistance_levels = calculate_support_resistance(df_filtered)
    support_df = pd.DataFrame(support_levels, columns=['Date', 'Low'])
    resistance_df = pd.DataFrame(resistance_levels, columns=['Date', 'High'])

    col1, col2 = st.columns(2)
    with col1:
        st.line_chart(support_df.set_index('Date'))
    with col2:
        st.line_chart(resistance_df.set_index('Date'))

    st.divider()

    # MONEY FLOW INDICATOR
    st.subheader('Money Flow Indicator')
    st.markdown('*Explanation:*')
    st.markdown('*The Money Flow Indicator (MFI) is a technical analysis tool that measures the strength of money flowing in and out of a security over a specified period, typically 14 days. It combines price and volume data to assess buying and selling pressure. High MFI values suggest strong buying pressure, while low values indicate selling pressure. Traders use the MFI to confirm trends, identify potential reversals, and assess the overall market sentiment.*')
    unique_years = sorted([row.Year for row in df.select(F.year("Date").alias("Year")).distinct().collect()])
    select_year = st.selectbox("Select a year to study", unique_years)
    filtered_df = df.filter((F.year("Date") == select_year) & (col("company_name").isin(selected_company)))
    df_with_mfi = calculate_money_flow_indicator(filtered_df)
    pandas_df = df_with_mfi.toPandas()
    st.write('A value above 80 is considered overbought, indicating there may be excessive selling pressure, while a value below 20 is considered oversold, which may suggest a buying opportunity.')
    st.line_chart(pandas_df.set_index('Date')['Money_Flow_Index'])
