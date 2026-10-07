# [실습]
# FAISS로 gradio 챗봇 만들어보기

# rag17_1 카피

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

### 03. 임베딩
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(
    model = "text-embedding-3-small",
    api_key=api_key,
    base_url=base_url,
    # dimensions=5,
)

DB_PATH = './_db/Faiss17/'

# db.save_local(
#     folder_path=DB_PATH,
#     index_name='faiss_index17'
# )

db = FAISS.load_local(
    folder_path=DB_PATH,
    index_name="faiss_index17",
    embeddings=embeddings,
    allow_dangerous_deserialization=True
)

print("===========================================")
# 문서 저장소 ID 확인
print(db.index_to_docstore_id)
print("===========================================")
# 저장된 결과 확인
print(db.docstore._dict)
print("===========================================")
# 유사도 검색
aaa = db.similarity_search("삼성전자 창업주에 대해 알려줘", k=2)
print(aaa)

############################ Retrievers ############################
############################ 검색기 ############################
retriever = db.as_retriever(search_kwargs={"k":2})

############################ 모델 연결 ############################
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model = 'gpt-5.6-terra',
    temperature=0,  # 0 = 있는 그대로, 1 = 창의적으로
    max_tokens=1000,
    api_key=api_key,
    base_url=base_url,
)

from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

prompt = ChatPromptTemplate.from_template("""
다음 컨텍스트를 바탕으로 질문에 답변해 주세요. 
컨텍스트 관련 정보가 없다면, "주어진 정보로는 답변할 수 없습니다"
라고 말해주세요.

컨텍스트 : {context}
질문 : {input}
답변 :
""")

# 체인 생성
docu_chain = create_stuff_documents_chain(model, prompt)    # prompt | model (프롬프트와 모델 연결)
rag_chain = create_retrieval_chain(retriever, docu_chain)   # 검색 | docu_chain

######################## Gradio 챗봇 ########################
import gradio as gr

def answer_invoke(message, history):
    response = rag_chain.invoke({"input" : message})
    return response['answer']

# Gradio 인터페이스 만들기
demo = gr.ChatInterface(fn=answer_invoke, title='Test Chat Bot')

# Gradio 실행
demo.launch()
