import streamlit as st
import pandas as pd
import numpy as np
import logging
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
import streamlit as st
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from callback.streamlit_callback import StreamHandler

st.set_page_config(layout="wide", page_icon="😊", page_title="Chat")
# st.title('🔗 Chat With Data')


province=""
city=""
sales_situation=""
stock_situation=""
price_situation=""
dataframe=pd.read_excel("tag-all.xlsx","Sheet1")
with st.sidebar:
    new_dataframe=dataframe.replace(" ", np.nan)  
    provinces=new_dataframe["Province"].drop_duplicates().dropna().values.tolist()
    province=st.selectbox("Province",provinces)
   
    new_dataframe=dataframe[dataframe["Province"]==province].replace(" ", np.nan)
    citys=new_dataframe["City"].drop_duplicates().dropna().values.tolist()
    city=st.selectbox("City",citys)

    new_dataframe=dataframe[(dataframe["Province"]==province)&(dataframe["City"]==city)].replace(" ", np.nan)
    date=new_dataframe["date"].drop_duplicates().dropna().values.tolist()
    date = [int(d) for d in date]
    date=sorted(date)
    start_date=st.selectbox("From",date)

    new_dataframe=dataframe[(dataframe["Province"]==province)&(dataframe["City"]==city)].replace(" ", np.nan)
    date=new_dataframe["date"].drop_duplicates().dropna().values.tolist()
        
    date = [int(d) for d in date]
    date=sorted(date)
    if len(date) >5:
        end_date=st.selectbox("To",date,index=5)
    else:
        end_date=st.selectbox("To",date)
     
    sales_situation=st.selectbox("HasSalesProblem",["Yes","N/A","No"],index=None)
    stock_situation=st.selectbox("HasInventoryProblem",["Yes","N/A","No"],index=None)
    point_situation=st.selectbox("HasViewPoint",["Yes","N/A","No"],index=None)
    result=st.button("Search")

if result:
    analysis_data=None
    analysis_data=dataframe[(dataframe["Province"]==province) & (dataframe["City"]==city) &((dataframe["date"]>=start_date)&(dataframe["date"]<=end_date))]
    if sales_situation:
        if sales_situation =="N/A":
            analysis_data=analysis_data[dataframe["HasSalesProblem"].isna()]
        else:
            analysis_data=analysis_data[(dataframe["HasSalesProblem"]==sales_situation)]
    if stock_situation:
            if stock_situation =="N/A":
                analysis_data=analysis_data[dataframe["HasInventoryProblem"].isna()]
            else:
                analysis_data=analysis_data[(dataframe["HasInventoryProblem"]==stock_situation)]
    if point_situation:
            if point_situation =="N/A":
                 analysis_data=analysis_data[dataframe["HasViewPoint"].isna()]
            else:
                analysis_data=analysis_data[(dataframe["HasViewPoint"]==point_situation)]
    analysis_data = analysis_data.sort_values(by='date', ascending=False)
    st.dataframe(analysis_data,width=3000,height=800)