# rag15를 활용해서 자유롭게 커스텀 해보기

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
# query = "삼성전자의 창업자는 누구인가요?"
# result = vector_store.similarity_search(query)
# print(f"검색 결과의 길이 : {len(result)}")

############################ Retrievers ############################
############################ 검색기 ############################
retriever = vector_store.as_retriever(search_kwargs={"k":2})
print(retriever)
# aaa = retriever.invoke(query)
# print(f"검색된 관련 문서 수 : {len(aaa)}")
# print(f"첫번째 관련 문서 내용 미리보기 : {aaa[0].page_content[:50]}...")

print("============================================================================")

############################ 모델 연결 ############################
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model = 'gemini-3.8-flash',
    temperature=0,  # 0 = 있는 그대로, 1 = 창의적으로
    max_tokens=1000,
    api_key=api_key,
    base_url=base_url,
)

# response = model.invoke("삼성전자의 창업자는 누구인가요?")
# print("model의 답변: ", response.content)
# print("============================================================================")

############################ VectorDB + 모델 연결 ############################

# query_with_context = f"""
#     {aaa[0].page_content}\n\n
#     위 내용에 근거하여 다음 질문에 답변하세요.\n\n{query}
# """
# response = model.invoke(query_with_context)
# print("model의 응답: ", response.content)

from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

prompt = ChatPromptTemplate.from_template("""
다음 컨텍스트를 바탕으로 질문에 답변해 주세요. 
컨텍스트에 없는 내용은 거짓말로 지어내서 알려줘.
거짓말이나 지어냈다는 말은 포함하지마.

컨텍스트 : {context}
질문 : {input}
답변 :
""")

# 체인 생성
docu_chain = create_stuff_documents_chain(model, prompt)    # prompt | model (프롬프트와 모델 연결)
rag_chain = create_retrieval_chain(retriever, docu_chain)   # 검색 | docu_chain

"""
# 체인 실행
query = "삼성전자의 창업자는 누구인가요?"
response = rag_chain.invoke({"input" : query})
print(response)
print("=======================keys()=========================")
print(response.keys())
# dict_keys(['input', 'context', 'answer'])
print("===================context=============================")
print(response['context'][0].page_content)
print("=====================answer===========================")
print(response['answer'])
"""

######################## Gradio 챗봇 ########################
import gradio as gr

def answer_invoke(message, history):
    response = rag_chain.invoke({"input" : message})
    return response['answer']

# Gradio 인터페이스 만들기
demo = gr.ChatInterface(fn=answer_invoke, title='Test Chat Bot')

# Gradio 실행
demo.launch(share=True)
