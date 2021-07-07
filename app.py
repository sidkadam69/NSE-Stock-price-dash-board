# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 09:17:42 2021

@author: siddh
"""
import dash
import numpy as np
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import warnings
external_scripts = ['/assets/style.css']
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARMA',
                        FutureWarning)
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARIMA',
                        FutureWarning)



app = dash.Dash(__name__,external_scripts=external_scripts)
server = app.server

df = pd.read_csv("stock_data.csv")
df.columns
df['Stock'].unique()
app.layout = html.Div([
    # Setting the main title of the Dashboard
    html.H1("NSE Data Analysis", style={"textAlign": "center"}),
    # Dividing the dashboard in tabs
    dcc.Tabs(id="tabs", children=[
        # Defining the layout of the first Tab
        dcc.Tab(label='Stock Prices', children=[
            html.Div([
                html.H1("Stocks High vs Lows", 
                        style={'textAlign': 'center'}),
                # Adding the first dropdown menu and the subsequent time-series graph
                dcc.Dropdown(id='my-dropdown',
                             options=[{'label': 'ICICIBANK', 'value': 'ICICI'},
                                      {'label': 'SBIBANK','value': 'SBIN'}, 
                                      {'label': 'HDFCBANK', 'value': 'HDFC'}, 
                                      {'label': 'AXISBANK','value': 'AXIS'}], 
                             multi=True,value=['HDFC'],
                             style={"display": "block", "margin-left": "auto", 
                                    "margin-right": "auto", "width": "60%"}),
                dcc.Graph(id='highlow'),
                html.H1("Market Volume Compare", style={'textAlign': 'center'}),
                # Adding the second dropdown menu and the subsequent time-series graph
                dcc.Dropdown(id='my-dropdown2',
                             options=[{'label': 'ICICIBANK', 'value': 'ICICI'},
                                      {'label': 'SBIBANK','value': 'SBIN'}, 
                                      {'label': 'HDFCBANK', 'value': 'HDFC'},
                                      {'label': 'AXISBANK','value': 'AXIS'}], 
                             multi=True,value=['HDFC'],
                             style={"display": "block", "margin-left": "auto", 
                                    "margin-right": "auto", "width": "60%"}),
                dcc.Graph(id='volume')
            ], className="container"),
        ]),
    dcc.Tab(label='Machine Learning', children=[
        html.Div([html.H1("Machine Learning", style={"textAlign": "center"}), html.H2("ARIMA Time Series Prediction", style={"textAlign": "left"}),
                  dcc.Dropdown(id='my-dropdowntest',options=[{'label': 'ICICIBANK', 'value': 'ICICI'},
                                      {'label': 'SBIBANK','value': 'SBIN'}, 
                                      {'label': 'HDFCBANK', 'value': 'HDFC'},
                                      {'label': 'AXISBANK','value': 'AXIS'}],
                               style={"display": "block", "margin-left": "auto", "margin-right": "auto", "width": "50%"}),
                  dcc.RadioItems(id="radiopred", value="High", labelStyle={'display': 'inline-block', 'padding': 10},
                                 options=[{'label': "High", 'value': "High"}, {'label': "Low", 'value': "Low"},
                                          {'label': "Volume", 'value': "Volume"}], style={'textAlign': "center", }),
                  dcc.Graph(id='traintest'), dcc.Graph(id='preds'),
                  ], className="container"),
    ]),
]),
])



@app.callback(Output('highlow', 'figure'),
              [Input('my-dropdown', 'value')])
def update_graph(selected_dropdown):
    dropdown = {"ICICI": "ICICIBANK","SBIN": "SBIBANK","HDFC": "HDFCBANK","AXIS": "AXISBANK",}
    trace1 = []
    trace2 = []
    for stock in selected_dropdown:
        trace1.append(
          go.Scatter(x=df[df["Stock"] == stock]["Date"],
                     y=df[df["Stock"] == stock]["High"],
                     mode='lines', opacity=0.7, 
                     name=f'High {dropdown[stock]}',textposition='bottom center'))
        trace2.append(
          go.Scatter(x=df[df["Stock"] == stock]["Date"],
                     y=df[df["Stock"] == stock]["Low"],
                     mode='lines', opacity=0.6,
                     name=f'Low {dropdown[stock]}',textposition='bottom center'))
    traces = [trace1, trace2]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', 
                                            '#FF7400', '#FFF400', '#FF0056'],
            height=600,
            title=f"High and Low Prices for {', '.join(str(dropdown[i]) for i in selected_dropdown)} Over Time",
            xaxis={"title":"Date",
                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'count': 6, 'label': '6M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                   'rangeslider': {'visible': True}, 'type': 'date'},
             yaxis={"title":"Price (INR)"})}
    return figure


@app.callback(Output('volume', 'figure'),
              [Input('my-dropdown2', 'value')])
def update_graph(selected_dropdown_value):
    dropdown = {"ICICI": "ICICIBANK","SBIN": "SBIBANK","HDFC": "HDFCBANK","AXIS": "AXISBANK",}
    trace1 = []
    for stock in selected_dropdown_value:
        trace1.append(
          go.Scatter(x=df[df["Stock"] == stock]["Date"],
                     y=df[df["Stock"] == stock]["Volume"],
                     mode='lines', opacity=0.7,
                     name=f'Volume {dropdown[stock]}', textposition='bottom center'))
    traces = [trace1]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data, 
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', 
                                            '#FF7400', '#FFF400', '#FF0056'],
            height=600,
            title=f"Market Volume for {', '.join(str(dropdown[i]) for i in selected_dropdown_value)} Over Time",
            xaxis={"title":"Date",
                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'count': 6, 'label': '6M',
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                   'rangeslider': {'visible': True}, 'type': 'date'},
             yaxis={"title":"Transactions Volume"})}
    return figure

@app.callback(Output('traintest', 'figure'),
              [Input('my-dropdowntest', 'value'), Input("radiopred", "value"),])
def update_graph(stock , radioval):
    dropdown = {"ICICI": "ICICIBANK","SBIN": "SBIBANK","HDFC": "HDFCBANK","AXIS": "AXISBANK",}
    radio = {"High": "High Prices", "Low": "Low Prices", "Volume": "Market Volume", }
    trace1 = []
    trace2 = []
    train_data = df[df['Stock'] == stock][-100:][0:int(100 * 0.8)]
    test_data = df[df['Stock'] == stock][-100:][int(100 * 0.8):]
    if (stock == None):
        trace1.append(
            go.Scatter(x= [0], y= [0],
                       mode='markers', opacity=0.7, textposition='bottom center'))
        traces = [trace1]
        data = [val for sublist in traces for val in sublist]
        figure = {'data': data,
                  'layout': go.Layout(colorway=['#FF7400', '#FFF400', '#FF0056'],
                                      height=600, title=f"{radio[radioval]}",
                                      paper_bgcolor='rgba(0,0,0,0)',
                                      plot_bgcolor='rgba(0,0,0,0)')}
    else:
        trace1.append(go.Scatter(x=train_data['Date'],y=train_data[radioval], mode='lines',
            opacity=0.7,name=f'Training Set',textposition='bottom center'))
        trace2.append(go.Scatter(x=test_data['Date'],y=test_data[radioval],mode='lines',
            opacity=0.6,name=f'Test Set',textposition='bottom center'))
        traces = [trace1, trace2]
        data = [val for sublist in traces for val in sublist]
        figure = {'data': data,
            'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
                height=600,title=f"{radio[radioval]} Train-Test Sets for {dropdown[stock]}",
                xaxis={"title":"Date",
                       'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 'step': 'month', 'stepmode': 'backward'},
                                                          {'count': 6, 'label': '6M', 'step': 'month', 'stepmode': 'backward'},
                                                          {'step': 'all'}])},
                       'rangeslider': {'visible': True}, 'type': 'date'},yaxis={"title":"Price (USD)"},     paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)')}
    return figure


@app.callback(Output('preds', 'figure'),
              [Input('my-dropdowntest', 'value'), Input("radiopred", "value"),])
def update_graph(stock, radioval):
    dropdown = {"ICICI": "ICICIBANK","SBIN": "SBIBANK","HDFC": "HDFCBANK","AXIS": "AXISBANK",}
    radio = {"High": "High Prices", "Low": "Low Prices", "Volume": "Market Volume", }
    dropdown = {"ICICI": "ICICIBANK","SBIN": "SBIBANK","HDFC": "HDFCBANK","AXIS": "AXISBANK",}
    trace1 = []
    trace2 = []
    if (stock == None):
        trace1.append(
            go.Scatter(x= [0], y= [0],
                       mode='markers', opacity=0.7, textposition='bottom center'))
        traces = [trace1]
        data = [val for sublist in traces for val in sublist]
        figure = {'data': data,
                  'layout': go.Layout(colorway=['#FF7400', '#FFF400', '#FF0056'],
                                      height=600, title=f"{radio[radioval]}",
                                      paper_bgcolor='rgba(0,0,0,0)',
                                      plot_bgcolor='rgba(0,0,0,0)')}
    else:
        test_data = df[df['Stock'] == stock][-100:][int(100 * 0.8):]
        train_data = df[df['Stock'] == stock][-100:][0:int(100 * 0.8)]
        train_ar = train_data[radioval].values
        test_ar = test_data[radioval].values
        history = [x for x in train_ar]
        predictions = list()
        for t in range(len(test_ar)):
            model = ARIMA(history, order=(3, 1, 0))
            model_fit = model.fit()
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
            obs = test_ar[t]
            history.append(obs)
        error = mean_squared_error(test_ar, predictions)
        trace1.append(go.Scatter(x=test_data['Date'],y=test_data['High'],mode='lines',
            opacity=0.6,name=f'Actual Series',textposition='bottom center'))
        trace2.append(go.Scatter(x=test_data['Date'],y= predictions, mode='lines',
            opacity=0.7,name=f'Predicted Series (MSE: {error})',textposition='bottom center'))
        traces = [trace1, trace2]
        data = [val for sublist in traces for val in sublist]
        figure = {'data': data,
            'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
                height=600,title=f"{radio[radioval]} ARIMA Predictions vs Actual for {dropdown[stock]}",
                xaxis={"title":"Date",
                       'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 'step': 'month', 'stepmode': 'backward'},
                                                          {'count': 6, 'label': '6M', 'step': 'month', 'stepmode': 'backward'},
                                                          {'step': 'all'}])},
                       'rangeslider': {'visible': True}, 'type': 'date'},yaxis={"title":"Price (INR)"},     paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)')}
    return figure

if __name__ == '__main__':
    app.run_server(debug=True)