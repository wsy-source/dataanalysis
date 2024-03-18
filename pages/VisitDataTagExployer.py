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

if st.session_state.get("data") is None:
    dataframe=pd.read_excel("tag-cn.xlsx","Sheet1")
    st.session_state["data"]=dataframe
    print(11111)
with st.sidebar:
    dataframe=st.session_state["data"]
    new_dataframe=dataframe.replace(" ", np.nan)  
    provinces=new_dataframe["省"].drop_duplicates().dropna().values.tolist()
    province=st.selectbox("Province",provinces)
   
    new_dataframe=dataframe[dataframe["省"]==province].replace(" ", np.nan)
    citys=new_dataframe["区"].drop_duplicates().dropna().values.tolist()
    citys.append("All")
    city=st.selectbox("City",citys,index=len(citys)-1)

    if city != "All":
        new_dataframe=dataframe[(dataframe["省"]==province)&(dataframe["区"]==city)].replace(" ", np.nan)
        date=new_dataframe["日期"].drop_duplicates().dropna().values.tolist()
        date = [int(d) for d in date]
        date=sorted(date)
        start_date=st.selectbox("From",date)

        new_dataframe=dataframe[(dataframe["省"]==province)&(dataframe["区"]==city)].replace(" ", np.nan)
        date=new_dataframe["日期"].drop_duplicates().dropna().values.tolist()
            
        date = [int(d) for d in date]
        date=sorted(date)
        if len(date) >5:
            end_date=st.selectbox("To",date,index=5)
        else:
            end_date=st.selectbox("To",date)
    else:
        new_dataframe=dataframe[(dataframe["省"]==province)].replace(" ", np.nan)
        date=new_dataframe["日期"].drop_duplicates().dropna().values.tolist()
        date = [int(d) for d in date]
        date=sorted(date)
        start_date=st.selectbox("From",date)

        new_dataframe=dataframe[(dataframe["省"]==province)].replace(" ", np.nan)
        date=new_dataframe["日期"].drop_duplicates().dropna().values.tolist()
            
        date = [int(d) for d in date]
        date=sorted(date)
        if len(date) >5:
            end_date=st.selectbox("To",date,index=5)
        else:
            end_date=st.selectbox("To",date)
     
    sales_situation=st.selectbox("HasSalesProblem",["Yes","N/A","No"],index=None)
    stock_situation=st.selectbox("HasInventoryProblem",["Yes","N/A","No"],index=None)
    point_situation=st.selectbox("HasViewPoint",["Yes","No"],index=None)
    result=st.button("Search")

if result:
    analysis_data=None
    if city != "All":
        analysis_data=dataframe[(dataframe["省"]==province) & (dataframe["区"]==city) &((dataframe["日期"]>=start_date)&(dataframe["日期"]<=end_date))]
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
        analysis_data = analysis_data.sort_values(by='日期', ascending=False)
        st.dataframe(analysis_data,width=3000,height=800)
    else:   
        analysis_data=dataframe[(dataframe["省"]==province)&((dataframe["日期"]>=start_date)&(dataframe["日期"]<=end_date))]
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
        analysis_data = analysis_data.sort_values(by='日期', ascending=False)
        st.dataframe(analysis_data,width=3000,height=800)