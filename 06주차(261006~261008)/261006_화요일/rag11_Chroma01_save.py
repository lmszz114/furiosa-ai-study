import os
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from langchain_chroma import Chroma
from dotenv import load_dotenv

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

### 04. VectorDB에 삽입하기
DB_PATH = "./_db/Chroma11/"
#저장
db = Chroma.from_documents(
    documents = split_doc1 + split_doc2,
    embedding = embeddings,
    persist_directory = DB_PATH,
    collection_name = "croma11",
)
print("Chroma 문서 저장 끝")