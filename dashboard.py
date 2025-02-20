import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd

app = dash.Dash(__name__)

#df = pd.read_parquet("data/processed/msft_features.parquet")
df = pd.read_parquet("data/processed/aapl_features.parquet")
#df = pd.read_parquet("data/processed/amzn_features.parquet")

fig = px.line(df, x="date", y=["close", "ma_7"], title ="Stock Price Analysis")

app.layout = html.Div([
    html.H1("Financial Dashboard"),
    dcc.Graph(figure=fig)
])

if __name__ == "__main__":
    app.run_server(debug=True)