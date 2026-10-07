# rag12_4 카피

import os
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = "https://monogpt.kr/api/monorouter/v1"

### 03. 임베딩
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(
    model = "text-embedding-3-small",
    api_key=api_key,
    base_url=base_url,
    # dimensions=5,
)

### 04. VectorDB에 삽입하기
DB_PATH = "./_db/Chroma12/"

vector_store = Chroma(
    embedding_function = embeddings,
    persist_directory = DB_PATH,
    collection_name = "croma12",
)

print(f"벡터 저장소에 저장된 문서 수 : {vector_store._collection.count()}")
query = "삼성전자의 창업자는 누구인가요?"
result = vector_store.similarity_search(query)
print(f"검색 결과의 길이 : {len(result)}")

############################ Retrievers ############################
############################ 검색기 ############################
retriever = vector_store.as_retriever(search_kwargs={"k":2})
print(retriever)
aaa = retriever.invoke(query)
print(f"검색된 관련 문서 수 : {len(aaa)}")
print(f"첫번째 관련 문서 내용 미리보기 : {aaa[0].page_content[:50]}...")

print("============================================================================")

############################ 모델 연결 ############################
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model = 'gpt-5-nano',
    temperature=0,  # 0 = 있는 그대로, 1 = 창의적으로
    max_tokens=1000,
    api_key=api_key,
    base_url=base_url,
)

response = model.invoke("삼성전자의 창업자는 누구인가요?")
print("model의 답변: ", response.content)
print("============================================================================")

############################ VectorDB + 모델 연결 ############################

query_with_context = f"""
    {aaa[0].page_content}\n\n
    위 내용에 근거하여 다음 질문에 답변하세요.\n\n{query}
"""

response = model.invoke(query_with_context)
print("model의 응답: ", response.content)