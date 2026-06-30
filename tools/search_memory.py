from tools.base import Tool
from memory.vector_memory import ManageMemory
from langchain.tools import BaseTool


class SearchMemoryTool(Tool):
    def __init__(self, memory_manager: ManageMemory):
        super().__init__(
            name="search_memory",
            description="从当前项目的对话历史记忆中检索与用户问题相关的信息",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "要检索的关键词或问题"
                    },
                    "k": {
                        "type": "integer",
                        "description": "返回结果数量，默认 3",
                        "default": 3
                    }
                },
                "required": ["query"]
            }
        )
        self.memory_manager = memory_manager

    def execute(self, query: str, k: int = 3) -> str:
        results = self.memory_manager.search(query, k)
        if not results:
            return "未找到相关历史记录"
        return "\n---\n".join(results)