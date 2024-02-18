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
from langchain.prompts import PromptTemplate

prompt="""
  As a regional sales manager for Goodyear, you need to analyze the content discussed based on the visit data to analyze Analyze the topics and details discussed in the interview data to answer user questions.
  Use customer number instead of data number
  
  The questions asked by users may not have direct answers（No need to repeat the statement）. 
  You need to summarize them from the topics and detailed descriptions discussed in the interview data.
  Just tell me the result of your answer
  
  user question: {question}
  current data: 
  province: {province} city: {city} sales data 
  data: {data}

  answer in {languague}
"""

PROMPT=PromptTemplate.from_template(prompt)

st.set_page_config(layout="wide", page_icon="😊", page_title="Chat")
# st.title('🔗 Chat With Data')

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "memory" not in st.session_state:
    st.session_state["memory"]=ConversationBufferMemory(memory_key="history",return_messages=True,input_key="question")

province=""
city=""

dataframe=pd.read_excel("tag-data.xlsx","Sheet2")
with st.sidebar:
    new_dataframe=dataframe.replace(" ", np.nan)  
    provinces=new_dataframe["省"].drop_duplicates().dropna().values.tolist()
    province=st.selectbox("省",provinces,index=None)
   
    new_dataframe=dataframe[dataframe["省"]==province].replace(" ", np.nan)
    citys=new_dataframe["市"].drop_duplicates().dropna().values.tolist()
    city=st.selectbox("市",citys)

    new_dataframe=dataframe[(dataframe["省"]==province)&(dataframe["市"]==city)].replace(" ", np.nan)
    date=new_dataframe["日期"].drop_duplicates().dropna().values.tolist()
    date = [int(d) for d in date]
    date=sorted(date)
    start_date=st.selectbox("From",date)

    new_dataframe=dataframe[(dataframe["省"]==province)&(dataframe["市"]==city)].replace(" ", np.nan)
    date=new_dataframe["日期"].drop_duplicates().dropna().values.tolist()
        
    date = [int(d) for d in date]
    date=sorted(date)
    if len(date) >5:
        end_date=st.selectbox("To",date,index=5)
    else:
        end_date=st.selectbox("To",date)
     
    sales_situation=st.selectbox("销售情况",["好","一般","差"],index=None)
    stock_situation=st.selectbox("库存情况",["好","一般","差"],index=None)
    price_situation=st.selectbox("价格情况",["好","一般","差"],index=None)
    visit_situation=st.selectbox("拜访描述",["好","一般","差"],index=None)

    analysis_data=dataframe[(dataframe["省"]==province) & (dataframe["市"]==city) &((dataframe["日期"]>=start_date)&(dataframe["日期"]<=end_date)) & (dataframe["销售情况"]==sales_situation) & (dataframe["库存情况"]==stock_situation) & (dataframe["价格情况"]==price_situation) & (dataframe["拜访描述"]==visit_situation)] 
    analysis_data = analysis_data.sort_values(by='日期', ascending=False)

st.dataframe(analysis_data,width=3000)