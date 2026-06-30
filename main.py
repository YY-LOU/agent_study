""" 组装所有组件 """
from memory.vector_memory import ManageMemory
from tools.search_memory import SearchMemoryTool
from agent import Agent
from openai import OpenAI

REACT_SYSTEM_PROMPT = """你是 AI 助手，遵循 ReAct（Reasoning + Acting）模式工作。

每次用户提问后，按以下步骤思考：

1. **分析**：分析用户需要什么，是否需要查找信息
2. **行动**：如果需要，选择适当的工具并调用
3. **观察**：查看工具返回的结果
4. **回答**：根据所有信息给出最终答案

可用的工具：
{tool_descriptions}

重要规则：
- 你不会看到之前的对话历史，每一次对话都是全新的开始。
- 如果需要了解用户之前说过什么（如名字、偏好、已讨论过的内容），
  必须调用 search_memory 工具来检索历史记忆。
- 如果不需要工具也能回答，直接回答即可。
"""

client = OpenAI(
    api_key="sk-7b3ecd71cc3b485f84a5565ab225b063",
    base_url="https://api.deepseek.com"
)


def main():
    # 1. 初始化记忆管理器
    manager = ManageMemory("memory")

    # 2. 初始化工具
    tools = [
        SearchMemoryTool(manager),
    ]

    # 3. 格式化 system prompt，把工具列表填进去
    tool_descriptions = "\n".join([f"- {t.name}: {t.description}" for t in tools])
    system_prompt = REACT_SYSTEM_PROMPT.format(tool_descriptions=tool_descriptions)

    # 4. 初始化 Agent
    agent = Agent(
        tools=tools,
        model="deepseek-v4-flash",
        system_prompt=system_prompt,
        client=client
    )

    # 5. 交互循环
    print("可用命令：/list, /creat 项目名, /use 项目名, /delete 项目名, /help")
    print()

    while True:
        user_input = input("你：")
        if user_input.lower() in ["exit", "quit"]:
            print("再见")
            break

        # ---- 命令处理（从 DSFirst.py 补回） ----
        if user_input.startswith("/"):
            parts = user_input.strip().split(maxsplit=1)
            cmd = parts[0]
            arg = parts[1] if len(parts) > 1 else ""

            if cmd == "/list":
                manager.list_projects()
            elif cmd == "/help":
                manager.show_help()
            elif cmd == "/creat":
                if arg:
                    manager.creat_project(arg)
                else:
                    print("用法：/creat 项目名")
            elif cmd == "/use":
                if arg:
                    manager.use_project(arg)
                else:
                    print("用法：/use 项目名")
            elif cmd == "/delete":
                if arg:
                    manager.delete_project(arg)
                else:
                    print("用法：/delete 项目名")
            else:
                print(f"未知命令：{cmd}，可以输入 /help 查看帮助文档")
            continue

        # ---- 项目选择检查（从 DSFirst.py 补回） ----
        if manager.current is None:
            print("当前未选择项目，请先使用 /use 项目名 选择一个项目")
            continue

        # ---- 对话 ----
        response = agent.run(user_input)

        # ---- 保存到记忆（从 DSFirst.py 补回） ----
        manager.add(f"用户输入：{user_input}\nAI回答：{response}")

        print(f"AI：{response}")
        print()


if __name__ == "__main__":
    main()