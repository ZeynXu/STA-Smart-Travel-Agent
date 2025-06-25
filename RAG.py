from langchain_ollama import OllamaLLM,OllamaEmbeddings
from langchain_core import document_loaders
from langchain_community.document_loaders import PyPDFLoader
from langchain
import langchain_text_splitters
import langchain_chroma

# 调用LLM
model = OllamaLLM(model='deepseek-r1:7b')
embeddings=OllamaEmbeddings(model='nomic-embed-text:latest')

# 加载文档，切分文档
loader=PyPDFLoader('')

splitter = langchain_text_splitters.TextSplitter()
docs = document_loaders.BaseLoader.load_and_split('', splitter)

# 构造向量数据库和检索器



# 检索结果重排序


# 组装prompt，得到响应
