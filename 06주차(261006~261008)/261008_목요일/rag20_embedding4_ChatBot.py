# [실습]
# bge-3, qwen3 활용해서 챗봇 구현

import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from langchain_chroma import Chroma
from dotenv import load_dotenv
import faiss
from langchain_community.vectorstores import FAISS  
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

load_dotenv()
api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = "https://monogpt.kr/api/monorouter/v1"


### 01. 데이터 불러오기
path = "./_data/"
pdf_loader = PyPDFLoader(path + "Attention_is_all_you_need.pdf")
pdf_docs = pdf_loader.load()


print(type(pdf_docs))   # <class 'list'>
print(len(pdf_docs))    # 15

### 02. 문서 자르기 (청킹)
Text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap = 100,
    separators = ["\n\n", "\n", " ", ""],
)

split_pdf_docs = Text_splitter.split_documents(pdf_docs)

# 문서 개수 확인
print(len(split_pdf_docs))  # 205


### 03. 임베딩
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(
    model_name = "BAAI/bge-m3",
    model_kwargs = {
        "device" : "cpu",
        # "local_files_only" : True
    }
)

##################### FAISS #####################
faiss_index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
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

######################## DB SAVE ########################
# db = FAISS.from_documents(
#     documents=split_pdf_docs,
#     embedding=embeddings,
# )

# DB_PATH = './_db/Faiss19/'

# db.save_local(
#     folder_path=DB_PATH,
#     index_name='faiss_index19'
# )
# 저장 완료 후 주석처리 함
#########################################################

######################## DB LOAD ########################
DB_PATH = './_db/Faiss19/'
db = FAISS.load_local(
    folder_path=DB_PATH,
    index_name="faiss_index19",
    embeddings=embeddings,
    allow_dangerous_deserialization=True
)
#########################################################

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