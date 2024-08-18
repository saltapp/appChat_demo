
from functools import update_wrapper
import time
import dotenv
import wrapt
dotenv.load_dotenv(dotenv.find_dotenv())

from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# 初始化向量数据库
embedding = OpenAIEmbeddings()

PATH=r"D:\appChat_demo\backend\data\docs\faiss"
faiss_vectorstore = FAISS.load_local(PATH, embedding, allow_dangerous_deserialization=True)





class PoweredVector:


    def __init__(self, func):

        def get_chroma():
            chroma_persist_directory = r'D:\appChat_demo\backend\data\vector_db'
            vectordb = Chroma(persist_directory=chroma_persist_directory, embedding_function=embedding)
            return vectordb


        update_wrapper(self, func)
        self.vectordb = get_chroma()
        self.func = func

    @wrapt.decorator
    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)
    
    def eager_call(self, *args, **kwargs):
        print(f"Not need to wait")
        return self.func(*args, **kwargs)
    

def change_vector(func):

    def get_chroma():
        chroma_persist_directory = r'D:\appChat_demo\backend\data\vector_db'
        vectordb = Chroma(persist_directory=chroma_persist_directory, embedding_function=embedding)
        return vectordb
    
    def decorated(*args, **kwargs):
        print(f"Wait for 1 second")
        decorated_args_list = []
        for arg in args:
            if isinstance(arg, FAISS):
                decorated_args_list.append(get_chroma())
            else:
                decorated_args_list.append(arg)
        print(f"args: {args}")
        print(f"kwargs: {kwargs}")
        return func(*args, **kwargs)
    return decorated

    
@change_vector
def query_simple(vectordb = None):
    print(vectordb.__class__)
    # query_text = "difference between Tesla and BYD"
    # docs = vectordb.similarity_search_with_score(query_text, k=5)

    # 输出结果
    # for doc, score in docs:
    #     print(f"Content: {doc.page_content}, Metadata: {doc.metadata}, Score: {score}")

if __name__ == "__main__":
    query_simple(vectordb =  faiss_vectorstore)