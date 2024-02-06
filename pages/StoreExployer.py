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
  As a store sales manager for Goodyear, you need to analyze the content discussed based on the visit data to analyze Analyze the topics and details discussed in the interview data to answer user questions.
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

dataframe=None

with st.sidebar:
    languague=st.selectbox("Languague",["English","Chinese"])
    schema_info={
        "cn":{
            "province":"省",
            "city":"市",
            "date":"日期"
        },
        "en":{
            "province":"Province",
            "city":"City",
            "date":"date"
        }
    }

    if languague=="English":
        dataframe=pd.read_excel("demo_en.xlsx",dtype={"a": np.int32, "b": str},converters={'Customer Number':str})
        schema=schema_info["en"]
    else:
        dataframe=pd.read_excel("demo_cn.xlsx",dtype={"a": np.int32, "b": str},converters={'客户编号':str})
        schema=schema_info["cn"]


    new_dataframe=dataframe.replace(" ", np.nan)


    provinces=new_dataframe[schema["province"]].drop_duplicates().dropna().values.tolist()
    province=st.selectbox("Province",provinces,index=None)
   
    new_dataframe=dataframe[dataframe[schema["province"]]==province].replace(" ", np.nan)
    citys=new_dataframe[schema["city"]].drop_duplicates().dropna().values.tolist()
    city=st.selectbox("City",citys)


    new_dataframe=dataframe[(dataframe[schema["province"]]==province)&(dataframe[schema["city"]]==city)].replace(" ", np.nan)
    date=new_dataframe[schema["date"]].drop_duplicates().dropna().values.tolist()
    date = [int(d) for d in date]
    date=sorted(date)
    start_date=st.selectbox("From",date)


    new_dataframe=dataframe[(dataframe[schema["province"]]==province)&(dataframe[schema["city"]]==city)].replace(" ", np.nan)
    date=new_dataframe[schema["date"]].drop_duplicates().dropna().values.tolist()
    
    date = [int(d) for d in date]
    date=sorted(date)
    if len(date) >5:
        end_date=st.selectbox("To",date,index=5)
    else:
        end_date=st.selectbox("To",date)

    


analysis_data=dataframe[(dataframe[schema["province"]]==province) & (dataframe[schema["city"]]==city) &((dataframe[schema["date"]]>=start_date)&(dataframe[schema["date"]]<=end_date))]
analysis_data = analysis_data.sort_values(by=schema["date"], ascending=False)


def call_llm(province,city,data,question,memory):
    llm = AzureChatOpenAI(azure_deployment="gpt-4106",streaming=True,
                          callbacks=[handler])
    
    chain=LLMChain(llm=llm,memory=memory,prompt=PROMPT)
    chain.invoke(input={
        "province": province,
        "city":city,
        "data": data,
        "question":question,
        "languague": languague
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