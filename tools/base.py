""" tools抽象基类，每个工具继承基类，保证接口统一 """

class Tool:
    def __init__(self,name:str,description:str,parameters:dict):
        self.name=name
        self.description=description
        self.parameters=parameters
        

    def execute(self,**kwarge)->str:
        """ 执行工具逻辑，返回字符串结果 """
        raise NotImplementedError

    def to_openai_schema(self)->dict:
        """ 转换成OpenAI Tools参数的格式 """
        return {
            "type":"function",
            "function":{
                "name":self.name,
                "description":self.description,
                "parameters":self.parameters
            }
        }