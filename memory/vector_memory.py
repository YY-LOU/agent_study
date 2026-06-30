import json
from sentence_transformers import SentenceTransformer
import faiss
import os
import numpy as np



class VectorMemory:
    def __init__(self,save_path:str="memory"):
        self.save_path=save_path
        self.encoder=SentenceTransformer('all-MiniLM-L6-v2',local_files_only=True)  #选择模型all-MiniLM-L6-v2，离线，80mb
        self.dimension=384      #每个向量384维

        #Faiss索引本质是一个快速找邻居的数据结构，IndexFlatL2是暴力搜索，L2代表欧几里得距离
        # index存的数据就是向量索引，存了向量
        self.index=faiss.IndexFlatL2(self.dimension)
        self.texts=[]   #按顺序存原始文本（未转为向量的）
        # 加载历史记忆文件
        self._load()
    
    def _load(self):
        """ 从磁盘加载2个历史记忆文件 """
        index_file=f"{self.save_path}.faiss"
        texts_file=f"{self.save_path}.jsonl"
        # ctrl+v
        if os.path.exists(index_file) and os.path.exists(texts_file):
            self.index = faiss.read_index(index_file)
            with open(texts_file, "r", encoding="utf-8") as f:
                self.texts = [line.rstrip("\n") for line in f if line.strip()]
            print(f"🔄 已加载 {len(self.texts)} 条历史记忆")

    def _save(self):
        """ 
        写盘持久化保存函数
        在新对话加入到faiss索引index和上下文texts列表中时写，写穿策略
        """
        # 调用faiss封装的内置写函数，仅支持全量写，可选延迟写，现在是写穿策略
        faiss.write_index(self.index,f"{self.save_path}.faiss")

        # 增量写json格式的texts
        with open(f"{self.save_path}.jsonl","a",encoding='utf-8') as f:
            f.write(self.texts[-1]+"\n")


    def add(self,text:str):
        """ 
        把一段text文本存入记忆中，保存了原文和向量库中
        主要做2步：
        1.用内置离线embedding模型：all-MiniLM-L6-v2把text转为向量
        2.向量存入faiss的index索引self.index中，原文text存入texts[]列表中
        """

        # 转为Vector向量，.encode([text])返回（1，184）的二维向量
        # astype(np.float32)把二维向量转为Faiss要求的float32类型数据
        Vector=self.encoder.encode([text]).astype(np.float32)

        # 把Vector向量加入Faiss索引index，Faiss会自动从0开始为其分配内部id
        self.index.add(Vector)

        # 把原始Text文本加入texts[]中，texts[]和self.index是一一对应的
        # 这里对self.index的类型不是很了解，不知道faiss.IndexFlatL2(self.dimension)的返回值类型，等会打印输出看一下
        self.texts.append(text)
        
        # 记忆持久化保存
        self._save()

    def search(self,query:str,k:int=3)->list[str]:
        """
        根据query找到历史相近的k条信息并返回 
        """
        # 如果是首轮对话，texts[]列表为空，直接返回
        if len(self.texts)==0:
            return []
        
        #把待查询的query转为向量 
        queryVector=self.encoder.encode([query]).astype(np.float32)

        # 使用self.index.search查询与queryVector最接近的k个向量
        # 返回距离distances和索引值indices，其中distance不是很重要
        distance,indices=self.index.search(queryVector,min(k,len(self.texts)))
        # 这里indices类型内容和类型不理解
        #因为indices的形状是类似[[0,5,10]]，所以要套一层indices[0],代表索引号为0，5，10的最相关。
        # 索引号:0，5，10是index索引号，与texts的索引号相互对应，只不过texts存的是原文本，index存的是向量
        return [self.texts[i] for i in indices[0]]  


class ManageMemory:
    """ 
    多项目记忆管理器。
    目录结构：
    memory_root/
        ├── projects.json           ← 项目名列表
        ├── 项目A/
        │   ├── 项目A.faiss
        │   └── 项目A.jsonl
        └── 项目B/
            ├── 项目B.faiss
            └── 项目B.jsonl
    """
    def __init__(self,root:str="memory"):
        self.root=root                      #记忆根目录
        os.makedirs(root,exist_ok=True)     #确保根目录存在
        self.projects_file_path=os.path.join(root,"projects.json")    #projects_file.json文件存储json格式的项目名
        self.projects:list[str]=[]          #项目名列表
        self.current:str|None=None          #当前项目名
        self.memory:VectorMemory|None=None  #当前项目的VectoryMemory实例
        self._load_projects()               #加载项目列表
    
    """ 项目列表管理 """

    def _load_projects(self):
        """ 从projects.json中读取项目列表 """
        if os.path.exists(self.projects_file_path):
            with open(self.projects_file_path,'r',encoding='utf-8') as f:
                self.projects=json.load(f)
    
    def _save_projects(self):
        """ 把projects[]列表写入到文件projects_file.json文件中保存 """
        with open(self.projects_file_path,'w',encoding='utf-8') as f:
            json.dump(self.projects,f,ensure_ascii=False,indent=2)

    def list_projects(self):
        """ 列出所有项目名 """
        if len(self.projects)==0:
            print("暂无任何项目，请输入：/creat 项目名 创建项目")
        else:
            print("项目列表如下：")
            for p in self.projects:
                if p==self.current:
                    print(f"{p}"+"(当前项目)")
                else:
                    print(p)
    
    def creat_project(self,name:str):
        """ 创建新项目，项目名:name ,项目名不能重复"""
        if name in self.projects:
            print(f"创建失败，项目：{name}已存在，项目名不能重复,请重新创建")
            return
        # 先创建空项目文件，成功后才加入到projects中
        project_dir=os.path.join(self.root,name)
        os.makedirs(project_dir,exist_ok=True)
        self.projects.append(name)
        self._save_projects()
        print(f"已成功创建项目：{name}")
    
    def delete_project(self,name:str):
        """ 删除name项目 """
        # 先检查name项目是否存在
        if name not in self.projects:
            print(f"删除失败，项目：{name}不存在")
            return
        # 检查是否是当前项目
        if name==self.current:
            self.current=None
            self.memory=None
        # 先删除memory文件
        project_dir=os.path.join(self.root,name)
        if os.path.exists(os.path.join(project_dir)):
            for f in os.listdir(project_dir):
                os.remove(os.path.join(project_dir,f))
            os.rmdir(project_dir)
        
        # 更新projects并保存
        self.projects.remove(name)
        self._save_projects()

    def use_project(self,name:str):
        """ 使用项目：{name} """
        # 先检查项目是否存在
        if name not in self.projects:
            print(f"项目：{name}不存在，使用项目失败。可以使用命令：/creat 项目名 创建，也可以使用命令：/list 查看所有项目名")
            return
        self.current=name
        # 本项目文件路径前缀,类型待了解
        prefix=os.path.join(self.root,name,name)
        self.memory=VectorMemory(save_path=prefix)
        print(f"目录:{name}切换成功")

    def show_help(self):
        """ 输出帮助信息 """
        print("欢迎使用AI聊天助手，输入命令：\n/help 查看帮助文档\n/list 显示目录名列表\n/creat 项目名 创建项目\n/use 项目名 切换项目目录\n/delete 项目名 删除项目")

    """ 封装一层VectorMemory的add和search函数调用接口 """
    def add(self,text:str):
        if self.memory is None:
            print("请先使用：/use 项目名 选择一个项目")
            return
        self.memory.add(text)

    def search(self,query:str,k:int=3)->list[str]:
        if self.memory is None:
            return []
        return self.memory.search(query,k)
    