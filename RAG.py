from langchain_ollama import OllamaLLM
from langchain_core import document_loaders
import langchain_text_splitters
import langchain_chroma


# 调用LLM
model=OllamaLLM(model='deepseek-r1:7b')


# 加载文档，切分文档
loader=document_loaders.BaseLoader.load('')
chunks=langchain_text_splitters.TextSplitter(chunk_size=1000,chunk_overlap=100)

# 文本块存入向量库，建立索引
db=langchain_chroma.vectorstores()



# 检索相似文本块，组建prompt