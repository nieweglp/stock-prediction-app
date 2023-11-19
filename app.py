import streamlit as st
import pandas as pd
import yfinance
from datetime import date
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objs as go


def load_data(stock, start, end):
    data = yfinance.Ticker(stock)
    data = data.history(start=start, end=end, interval="1d")
    data.reset_index(inplace=True)
    data["Date"] = data["Date"].dt.tz_localize(None)
    return data


def main():
    st.header("Stock prediction app")

    start = st.date_input("Start date", value=date(2023, 1, 1))
    end = st.date_input("End date")
    days_to_predict = st.slider("Days to predict", 0, 365, 60)

    df_stocks = pd.read_csv("resources/nasdaq_tickers.csv")
    stocks_dict = pd.Series(df_stocks.name.values, index=df_stocks.symbol).to_dict()

    selected_stock = st.selectbox(
        "Select stock", stocks_dict.keys(), format_func=lambda x: stocks_dict[x]
    )

    data = load_data(selected_stock, start, end)
    st.subheader("Stock data")
    st.write(data.tail())

    # Plot selected stock price
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=data["Date"], y=data["Close"]))
    fig1.layout.update(title_text="Time series data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig1)

    # Candlestick chart
    fig2 = go.Figure(
        data=[
            go.Candlestick(
                x=data["Date"],
                open=data["Open"],
                high=data["High"],
                low=data["Low"],
                close=data["Close"],
            )
        ]
    )
    fig1.layout.update(title_text="Candlestick chart")
    st.plotly_chart(fig2)

    # Predict future stock price
    df_train = data[["Date", "Close"]]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    model = Prophet()
    model.fit(df_train)
    future = model.make_future_dataframe(periods=days_to_predict)
    forecast = model.predict(future)

    st.subheader("Forecast data")
    fig2 = plot_plotly(model, forecast)
    st.plotly_chart(fig2)


if __name__ == "__main__":
    main()
