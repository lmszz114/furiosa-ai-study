# rag11_1 카피

import os
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from langchain_chroma import Chroma
from dotenv import load_dotenv

import faiss    # 파이스에서 제공
from langchain_community.vectorstores import FAISS  # 랭체인에서 제공
from langchain_community.docstore.in_memory import InMemoryDocstore

load_dotenv()
api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = "https://monogpt.kr/api/monorouter/v1"

### 01. 데이터 불러오기
path = "./_data/rag_data/"
loader1 = TextLoader(path + "samsung_outlook.txt", encoding="utf-8")
loader2 = TextLoader(path + "nvidia_outlook.txt", encoding="utf-8")

### 02. 문서 자르기 (청킹)
Text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap = 100,
    separators = ["\n\n", "\n", " ", ""],
)

split_doc1 = loader1.load_and_split(Text_splitter)   # Text_splitter에서 정한대로 청크 300개, 오버랩을 100개로 자름 # samsung
split_doc2 = loader2.load_and_split(Text_splitter)   # nvidia

# 문서 개수 확인
# print(split_doc1)
print(len(split_doc1), len(split_doc2)) # 9 9       # document 단위 = 청크 단위

### 03. 임베딩
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(
    model = "text-embedding-3-small",
    api_key=api_key,
    base_url=base_url,
    # dimensions=5,
)

##################### FAISS #####################
faiss_index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
# faiss_index = faiss.IndexFlatL2(1536)
print("FAISS 인덱스 초기화 준비 완료")

# FAISS 벡터 저장소의 벡터 차원 수 (임베딩 차원 수)
print(faiss_index.d)    # 1536

faiss_db = FAISS(
    embedding_function=embeddings,
    index = faiss_index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

# 저장된 문서의 갯수 확인
print(faiss_db.index.ntotal)    # 0

######################## 준비 완료 ########################
#########################################################

db = FAISS.from_documents(
    documents=split_doc1 + split_doc2,
    embedding=embeddings,
)

DB_PATH = './_db/Faiss17/'

db.save_local(
    folder_path=DB_PATH,
    index_name='faiss_index17'
)