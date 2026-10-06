# rag11_1 카피

import os
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = "https://monogpt.kr/api/monorouter/v1"

from glob import glob

path = "./_data/rag_data/"
# 폴더에서 텍스트 파일 목록 가져오기
txt_files = glob(os.path.join(path, "*.txt"))

# print(txt_files)
# ['./_data/rag_data\\2026_AI_for_All.txt', './_data/rag_data\\nvidia_outlook.txt', './_data/rag_data\\samsung_outlook.txt']


### 01. 데이터 불러오기
# loader1 = TextLoader(path + "samsung_outlook.txt", encoding="utf-8")
# loader2 = TextLoader(path + "nvidia_outlook.txt", encoding="utf-8")

data = []
for text_file in txt_files:
    loader = TextLoader(text_file, encoding="utf-8")
    data += loader.load()

print("========================")
print(len(data))    # 3개의 데이터가 들어감
print(data[0].page_content)

char_count = [len(doc.page_content) for doc in data]
print(char_count)   # [8158, 2049, 1898] / 3개 파일 각각의 문장 개수


### 02. 문서 자르기 (청킹)
Text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap = 100,
    separators = ["\n\n", "\n", " ", ""],   # 통상 디폴트
)

texts = Text_splitter.split_documents(data)
print("생성된 텍스트 청크수 :", len(texts))     # 생성된 텍스트 청크수 : 59 <- 벡터가 이만큼 생길 예정임
print("각 청크의 길이 :", list(len(text.page_content) for text in texts))
# 각 청크의 길이 : [259, 154, 150, 282, 128, 276, 288, 258, 268, 262, 271, 226, 268, 182, 213, 281, 257, 182, 162, 208, 
# 259, 188, 225, 229, 219, 214, 258, 207, 284, 297, 198, 245, 177, 272, 215, 9, 269, 293, 290, 236, 289, 209, 222, 230, 
# 254, 249, 296, 181, 247, 243, 185, 219, 239, 235, 298, 299, 170, 187, 249]

print("첫번째 청크의 내용 :", texts[0].page_content)
# page_content 는 내용 / metadata 는 경로
# document ─ page_content
#          └ metadata
print("첫번째 청크의 내용 :", len(texts[0].page_content))
# 첫번째 청크의 내용 : 259


### 03. 임베딩
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(
    model = "text-embedding-3-small",
    api_key=api_key,
    base_url=base_url,
    # dimensions=5,
)

sample_text = "삼성전자의 창업자는 누구인가요?"
vector = embeddings.embed_query(sample_text)
# print(vector)
print(len(vector))  # 1536


### 04. VectorDB에 삽입하기
DB_PATH = "./_db/Chroma12/"
#저장
vector_store = Chroma.from_documents(
    documents = texts,
    embedding = embeddings,
    persist_directory = DB_PATH,
    collection_name = "croma12",
)
print(f"벡터 저장소에 저장된 문서 수 : {vector_store._collection.count()}")
# 벡터 저장소에 저장된 문서 수 :59 / 청크 갯수와 동일

query = "삼성전자의 창업자는 누구인가요?"
result = vector_store.similarity_search(query)

print(f"검색 결과의 길이 : {len(result)}")
# 검색 결과의 길이 : 4

############################ Retrievers ############################
############################ 검색기 ############################
retriever = vector_store.as_retriever(search_kwargs={"k":2})
print(retriever)
# tags=['Chroma', 'OpenAIEmbeddings'] vectorstore=<langchain_chroma.vectorstores.Chroma object at 0x000001CBE98D3750> 
# search_kwargs={'k': 2}
aaa = retriever.invoke(query)
print(f"검색된 관련 문서 수 : {len(aaa)}")
# 검색된 관련 문서 수 : 2
print(f"첫번째 관련 문서 내용 미리보기 : {aaa[0].page_content[:50]}...")
# 첫번째 관련 문서 내용 미리보기 : 삼성전자 사업 전망
# 삼성전자는 메모리 반도체, 파운드리, 스마트폰, 디스플레이와 가전 사...