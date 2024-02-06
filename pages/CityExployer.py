import streamlit as st
import pandas as pd
import numpy as np
import logging
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from retry import retry
import streamlit as st
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from callback.streamlit_callback import StreamHandler
from langchain.prompts import PromptTemplate
prompt="""
  As a regional sales manager for Goodyear,Need to answer user questions based on current regional sales data and chat history
  answer in chinese

  user question: {question}

  province: {province} city: {city} sales data 
  data: {data}

  current conversation history:
  {history}
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

dataframe=pd.read_excel("demo.xlsx")


with st.sidebar:
    new_dataframe=dataframe.replace(" ", np.nan)  
    provinces=new_dataframe["Province"].drop_duplicates().dropna().values.tolist()
    province=st.selectbox("Province",provinces,index=None)
   
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

analysis_data=dataframe[(dataframe["Province"]==province) & (dataframe["City"]==city) &((dataframe["date"]>=start_date)&(dataframe["date"]<=end_date))]
analysis_data = analysis_data.sort_values(by='date', ascending=False)


def call_llm(province,city,data,question,memory):
    llm = AzureChatOpenAI(azure_deployment="gpt-4106",streaming=True,
                          callbacks=[handler])
    
    chain=LLMChain(llm=llm,memory=memory,prompt=PROMPT)
    chain.invoke(input={
        "province": province,
        "city":city,
        "data": data,
        "question":question
    })
    st.session_state.messages=memory.chat_memory.messages



for msg in st.session_state.messages:

    if type(msg) == HumanMessage:
        st.chat_message("user").write(msg.content)
    if type(msg) == AIMessage:
        st.chat_message("assistant").write(msg.content)
        


if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    data=analysis_data.to_string()
    handler = StreamHandler(st.empty())
    st.dataframe(analysis_data)
    try:
        memory=st.session_state["memory"]
        call_llm(question=prompt,city=city,data=data,province=province,memory=memory)        
    except Exception as e:
        st.error("The token exceeds the maximum token supported by the model. Please reduce data input.")