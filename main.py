import streamlit as st
import os
from utils import *

def main():
    st.set_page_config(page_title="Stocks analysis", page_icon=":chart_with_upwards_trend:")
    selected_file = st.sidebar.file_uploader("Choose the file called STOCKS.csv", type=['csv', 'xlsx'])

    page = st.sidebar.selectbox("Navigation", ["Exploration","Average of the opening and closing prices","Stock prices changes", "Daily return","Moving Average","Return Rate","More Insights"])
    spark = create_spark_session()

    if selected_file is not None:
        with open(os.path.join("temp", selected_file.name), "wb") as f:
            f.write(selected_file.getbuffer())
        
        file_type = selected_file.name.split('.')[-1]

        df = read_file_with_spark(spark, os.path.join("temp", selected_file.name), file_type)

        if page == "Exploration":
            explore_page(df)
        elif page == "Average of the opening and closing prices":
            avg_prices_page(df)
        elif page == "Stock prices changes":
            stock_prices_change_page(df)
        elif page == "Daily return":
            daily_return_page(df)
        elif page == "Moving Average":
            moving_average_page(df)
        elif page == "Return Rate":
            return_rate_page(df)
        elif page == "More Insights":
            insights_page(df)

        os.remove(os.path.join("temp", selected_file.name))

if not os.path.exists("temp"):
    os.makedirs("temp")

if __name__ == "__main__":
    main()
