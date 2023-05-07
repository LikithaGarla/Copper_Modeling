import pandas as pd
import numpy as np
import datetime as dt
import base64
import plotly.express as px
from PIL import Image
import xgboost as xb
import matplotlib.pyplot as mpy
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
import seaborn as sb
import calendar
import streamlit as st
from sklearn.model_selection import train_test_split



df=pd.DataFrame()
global it
global year

def predict_Sales(model):
    x=[]
    l=[]
    x.append(int(it))
    l.append(int(year))
    data = {'item':x,'year':l}
    new_df = pd.DataFrame(data)
    xgtest = xb.DMatrix(new_df)
    x=model.predict(xgtest)
    st.write("Sales is :",x[0])

def Add_Date_Features(df):
    print(df.head())
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df.date.dt.year
    df['month'] = df.date.dt.month
    df['day'] = df.date.dt.day
    df['dayofyear'] = df.date.dt.dayofyear
    df['dayofweek'] = df.date.dt.dayofweek
    df['weekofyear'] = df.date.dt.isocalendar().week
    df['quarterofyear'] = df.date.dt.quarter 
    Add_Rolling_sum(df)

def Add_Rolling_sum(df):
    res = df.groupby('item',as_index=False)['sales'].rolling(90).sum().shift(-89)
    df['rolling_sum'] = res['sales']
    PerformDataCleaning(df)

def XGBoost(X_train,y_train,X_test,Y_test):
    matrix_train = xb.DMatrix(X_train, label = y_train)
    matrix_test = xb.DMatrix(X_test, label = Y_test)
    model = xb.train(params={'objective':'reg:linear','eval_metric':'mae'}
                ,dtrain = matrix_train, num_boost_round = 500, 
                early_stopping_rounds = 20, evals = [(matrix_test,'test')])
    predict_Sales(model)
     

def Split(df):
    df_train, df_test = train_test_split(df, test_size=0.25)
    X_train = df_train.drop('rolling_sum', axis=1).dropna()
    y_train = df_train['rolling_sum']
    X_test = df_test.drop('rolling_sum', axis=1)
    Y_test=df_test['rolling_sum']
    XGBoost(X_train,y_train,X_test,Y_test)


def PerformDataCleaning(df):
    df=df.dropna()
    df.drop(['sales','dayofyear','dayofweek','weekofyear','quarterofyear','month','day','date'],axis=1,inplace=True)
    Split(df)

def Read_Dataset():
    df = pd.read_csv('train 2.csv')
    df = df.groupby(['date','item'],as_index=False).agg(sales=pd.NamedAgg(column='sales',aggfunc=sum))
    Add_Date_Features(df)
    

tab1, tab2 = st.tabs(["Display Trends", "Predict Sales"])

with tab2:
    d = st.date_input(
    "Select the date",max_value=dt.date(2090,12,31))
    year = d.year
    number = st.number_input('Enter the Item No.',min_value=1,format='%d',max_value=25)
    it = number

    if st.button('Predict Sales'):
        Read_Dataset()

with tab1:
    df = pd.read_csv('train 2.csv')
    df3 = df.groupby(['date','item'],as_index=False).agg(sales=pd.NamedAgg(column='sales',aggfunc=sum))
    df3['date'] = pd.to_datetime(df3['date'])
    df3['year'] = df3.date.dt.year
    df3['month'] = df3.date.dt.month
    df3['day'] = df3.date.dt.day
    df3['dayofyear'] = df3.date.dt.dayofyear
    df3['dayofweek'] = df3.date.dt.dayofweek
    df3['weekofyear'] = df3.date.dt.isocalendar().week
    df3['quarterofyear'] = df3.date.dt.quarter 
    res = df3.groupby('item',as_index=False)['sales'].rolling(90).sum().shift(-89)
    df3['rolling_sum'] = res['sales']
    df4=df3.dropna()
    df4.drop('date',axis=1,inplace=True)


    option = st.radio(
    "How do you want to visualize the data",
    ('Yearly Aggregated', 'Yearly Itemwise', 'Monthly for a specific year', 'Itemwise for a specific year'), horizontal=True)

    if(option=='Yearly Aggregated'):
        res = df4.groupby('year',as_index=False)['sales'].sum()
        fig = px.bar(res,x='year',y='sales')
        st.plotly_chart(fig)
    elif option=='Yearly Itemwise':
        res = df4.groupby(['year','item'],as_index=False)['sales'].sum()
        #res.head()
        fig = px.bar(res,x='year',y='sales',color='item')
        st.plotly_chart(fig)
    elif option=='Monthly for a specific year':
        year = st.selectbox(
    'Select an year', (2013, 2014,2015, 2016,2017))
        res = df4[df4['year'] == year].groupby(['month','item'],as_index=False)['sales'].sum()
        #res['month_name'] = df4['month'].dt.month_name
        res['month_name']=res['month'].apply(lambda x: calendar.month_name[x])
        #res.head()
        fig = px.bar(res,x='month_name',y='sales',color='item')
        #fig.update_xaxes(categoryorder='array', categoryarray= ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])
        st.plotly_chart(fig)
    elif option=='Itemwise for a specific year':
        year = st.selectbox(
    'Select an year', (2013, 2014,2015, 2016,2017))
        #item = 2
        #res = df4[(df4['year'] == year) & (df4['item']==item)].groupby(['month'],as_index=False)['sales'].sum()
        items = st.multiselect(
    'What are your favorite colors',
    [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50],[1])
        #items=[1,2,4,5,6,7,8,9,10]
        res = df4[(df4['year'] == year) & (df4.item.isin(items))].groupby(['month','item'],as_index=False)['sales'].sum()
        res['month_name']=res['month'].apply(lambda x: calendar.month_name[x])
        fig = px.line(res,x='month_name',y='sales',color='item')
        st.plotly_chart(fig)
        




