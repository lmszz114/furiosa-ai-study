# rag23_01 카피

from langchain_core.documents import Document

texts = [
    Document(page_content="삼성전자는 반도체를 생산하는 기업이다."),
    Document(page_content="삼성전자는 갤럭시 스마트폰을 판매한다."),
    Document(page_content="SK하이닉스는 메모리 반도체를 생산한다."),
    Document(page_content="엔비디아는 인공지능 연산용 GPU를 설계한다."),
    Document(page_content="애플은 아이폰을 개발하는 기업이다."),
    Document(page_content="테슬라는 전기자동차를 제조한다."),
    Document(page_content="삼성전자는 AI 반도체 기술을 연구한다."),
    Document(page_content="반도체는 컴퓨터의 핵심 부품이다."),
]

query = "삼성전자의 반도체 사업"

# ####################### BM25 검색 #######################
# # BM25는 키워드 기반 검색. Best Match 25
# from langchain_community.retrievers import BM25Retriever

# bm25_retriever = BM25Retriever.from_documents(texts)
# bm25_retriever.k = 3

# bm25_docs = bm25_retriever.invoke(query)

# print("=================== BM25 검색 결과 ===================")
# for doc in bm25_docs:
#     print(doc.page_content)

###############################################################
###############################################################
########################### 벡터 검색 ###########################

#rag22_3 내용 카피
import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from langchain_chroma import Chroma
from dotenv import load_dotenv
import faiss
from langchain_community.vectorstores import FAISS  
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.messages import HumanMessage, AIMessage

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

####################### DB SAVE ########################
db = FAISS.from_documents(
    documents=split_pdf_docs,
    embedding=embeddings,
)

DB_PATH = './_db/Faiss23/'

db.save_local(
    folder_path=DB_PATH,
    index_name='faiss_index23'
)

########################################################

######################## DB LOAD ########################
DB_PATH = './_db/Faiss23/'
vector_store = FAISS.load_local(
    folder_path=DB_PATH,
    index_name="faiss_index23",
    embeddings=embeddings,
    allow_dangerous_deserialization=True
)

vector_retriever = vector_store.as_retriever(
    search_kwargs={"k":3}
)
vector_docs = vector_retriever.invoke(query)


print("=================== 벡터디비 검색 결과 ===================")
for doc in vector_docs:
    print(doc.page_content)