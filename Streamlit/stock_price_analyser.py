
import pandas as pd
import streamlit as st
import datetime 

import yfinance as yf  

st.write("""

        # STOCK PRICE ANALYSER
        ## Shown are the stock price 

""")  # HEADER 1 h1

## AAPL stock symbol of Apply company

ticker_symbol = st.text_input(
        "Enter Stock Symbol",
        "AAPL",
        key = "placeholder"
 )

col1,col2 = st.columns(2)

with col1:
    Start_date = st.date_input("Input Start Date: ", datetime.date(2015,1,1))  # start date
with col2:
    End_date = st.date_input("Input End Date: ", datetime.date(2023,1,1))  # End date

ticker_data = yf.Ticker(ticker_symbol)

ticker_df = ticker_data.history(
                                period="1d", 
                                interval="1d",
                                start= Start_date,
                                end= End_date,
                                )

st.write(f"""
    ### {ticker_symbol} stock price info
""")

st.write(ticker_df)

col1, col2 = st.columns(2)


## showcasing the line chart
with col1:
    st.write(""" ### Daily closing Prices""")
    st.line_chart(ticker_df.Close)
with col2:
    st.write(""" ### Daily Volume""")
    st.line_chart(ticker_df.Volume)

    