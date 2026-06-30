""" 学习langchain """
""" from langchain_openai import ChatOpenAI


llm=ChatOpenAI(
    model="deepseek-v4-flash",
    api_key="sk-7b3ecd71cc3b485f84a5565ab225b063",
    base_url="https://api.deepseek.com"
)

result=llm.invoke("你好，我是x")
print(result)

from langchain_core.messages import HumanMessage,SystemMessage

result=llm.invoke([
    SystemMessage("你是我的助手x"),
    HumanMessage("我是yy")
])
print(result) """

from langchain.tools import BaseTool
from memory.vector_memory import ManageMemory



class search_memory(BaseTool):
    name:str="search_memory"
    description:str="从当前项目的对话历史记忆中检索与用户问题相关的信息"
    memory_manager:ManageMemory

    def _run(self, query:str,k:int=3)->str:
        results=self.memory_manager.search(query,k)
        if not results:
            return "未找到相关历史记忆！"   #无记忆时触发
        return "\n---\n".join(results)


