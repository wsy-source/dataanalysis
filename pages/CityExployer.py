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
from openai import AzureOpenAI
from openai.types.chat import ChatCompletion
import time

st.set_page_config(layout="wide", page_icon="😊", page_title="Chat")
# st.title('🔗 Chat With Data')

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "example_messages" not in st.session_state:
    st.session_state["example_messages"] = [
        {"role":"system","content":"As the company's store sales manager, you need to analyze the content of discussions based on specific store visit data, analyze the topics and details discussed in the interview data, and answer users' questions. \n use customer number instead of data number \n The questions asked by users may not have direct answers（No need to repeat the statement. \n You are allowed to ask the user if the user's question is not clear"}
    ]

province=""
city=""

dataframe=None

with st.sidebar:
    languague=st.selectbox("Languague",["English","Chinese"])
    schema_info={
        "cn":{
            "province":"省",
            "city":"区",
            "date":"日期"
        },
        "en":{
            "province":"Province",
            "city":"City",
            "date":"date"
        }
    }

 
    dataframe=pd.read_excel("tag-cn.xlsx",dtype={"a": np.int32, "b": str},converters={'客户编号':str})
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




for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])





def generate(response:ChatCompletion):
    for chunk in response:
        if chunk.choices:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content
                time.sleep(0.015)


if prompt := st.chat_input():

    try:
        if analysis_data.empty:
            st.warning("No data found for this customer. Please select filter criteria in the left menu 👈")
        else:
            st.chat_message("user").write(prompt)
            # handler = StreamHandler(st.empty())
            client = AzureOpenAI()
            st.session_state.messages.append({"role":"user","content":prompt})
            messages = st.session_state.example_messages + st.session_state.messages
            all_data = analysis_data.to_dict(orient="records")
            messages.append({"role":"user", "content":f"answer question in {languague}"})
            messages.append({"role":"system", "content":all_data.__str__()})
            response = client.chat.completions.create(
                model="gpt-4106",
                stream=True,
                temperature=0.3,
                messages=messages,
            )
            result = st.chat_message("assistant").write_stream(generate(response))
            data=analysis_data.to_string()
            st.dataframe(analysis_data)
            st.session_state.messages.append({"role":"assistant", "content":result})
    except Exception as e:
        print(e.args.__str__())
        st.error("The token exceeds the maximum token supported by the model. Please reduce data input.")