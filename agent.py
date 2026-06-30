""" 
Agent类，主要设计agent.run()，实现reasoning acting,reflection等
"""
from openai import OpenAI
import json
from tools.base import Tool


class Agent():

    def __init__(self, tools: list[Tool], model: str, system_prompt: str, client: OpenAI):
        self.tools = tools
        self.model = model
        self.system_prompt = system_prompt
        self.tool_schemas = [t.to_openai_schema() for t in tools]
        self.client = client

    def _find_tool(self, name: str) -> Tool:
        for t in self.tools:
            if t.name == name:
                return t
        raise ValueError(f"未找到工具：{name}")

    def run(self, user_input: str) -> str:
        """
        ReAct 主循环（方案 B）。
        
        每次 run() 只带 system + 当前问题，不保留之前对话。
        AI 如果需要了解之前的内容，必须自己调 search_memory 工具。
        """
        # 每次重新构造，不跨轮次累积
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input}
        ]

        while True:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tool_schemas,
                tool_choice="auto"
            )

            msg = response.choices[0].message

            # AI 直接回答，本轮结束
            if response.choices[0].finish_reason == "stop":
                return msg.content

            # AI 要调工具
            elif response.choices[0].finish_reason == "tool_calls":
                # 把 AI 的 tool_call 请求加入本轮消息（保留 tool_calls 字段）
                messages.append(msg)

                for tool_call in msg.tool_calls:
                    tool = self._find_tool(tool_call.function.name)
                    arguments = json.loads(tool_call.function.arguments)
                    result = tool.execute(**arguments)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result
                    })
                # 继续 while 循环，AI 看到 tool 结果后继续思考
    
    # 以下函数并没有使用，仅供reflection参考
    def _reflect(self,user_input:str,draft:str)->str:
        """让 AI 审查自己的初步回答并改进"""
        messages = [
            {"role": "system", "content": "你是一个严谨的审查者。检查以下回答是否准确、完整。如果有问题，请给出改进版本。"},
            {"role": "user", "content": f"用户问题：{user_input}"},
            {"role": "assistant", "content": f"我的初步回答：{draft}"},
            {"role": "user", "content": "请审查上述回答，如有不准确或不完整之处，输出改进版本。如果没问题，直接输出原回答。"}
        ]
        response = self.client.chat.completions.create(model=self.model, messages=messages)
        return response.choices[0].message.content



           
