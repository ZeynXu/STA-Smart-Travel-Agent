from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_core import document_loaders
from langchain_docling import DoclingLoader
from langchain_milvus import Milvus
import langchain_text_splitters
import langchain_chroma

# 调用LLM
model = OllamaLLM(model='deepseek-r1:7b')
embeddings = OllamaEmbeddings(model='nomic-embed-text:latest')

# 加载文档，切分文档
loader = DoclingLoader(file_path='')
docs = loader.load()

splitter = langchain_text_splitters.TextSplitter()
docs = document_loaders.BaseLoader.load_and_split('', splitter)

# 构造向量数据库和检索器
URI = './milvus_vs.db'
vector_store = Milvus.from_documents(
    docs, embeddings,
    collection_name='',
    connection_args={'uri': URI},
)

retriever=vector_store.as_retriever()
retriever.invoke('')

# 检索结果重排序


# 组装prompt，得到响应
