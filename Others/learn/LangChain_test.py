import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain.tools import tool
from langchain.messages import HumanMessage,AIMessage,SystemMessage

load_dotenv(dotenv_path=r"srcY\.env")

api_key=os.getenv("DeepSeek_API_Key")
model_name=os.getenv("DeepSeek_Model")
base_URL=os.getenv("DeepSeek_BaseURL")

model=ChatOpenAI(
    model=model_name,
    api_key=api_key,
    base_url=base_URL
)

agent=create_agent(
    model=model,
    system_prompt="你叫yy，是一个乐观开朗的助理。"
)

# res=agent.invoke(
#     HumanMessage("你是谁？")
# )
# print(res["messages"])
# print(res.context)

# res=agent.invoke()
# res=agent.invoke({
#     "messages":[{"role":"user","content":"请问你是谁?"}]
# })
# print(HumanMessage("你是yy"))




