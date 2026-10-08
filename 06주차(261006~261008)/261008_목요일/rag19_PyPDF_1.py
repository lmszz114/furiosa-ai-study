from langchain_community.document_loaders import TextLoader, PyPDFLoader


path = "./_data/"
pdf_loader = PyPDFLoader(path + "Attention_is_all_you_need.pdf")
pdf_docs = pdf_loader.load()

print(type(pdf_docs))   # <class 'list'>
print(len(pdf_docs))    # 15
# print(pdf_docs)
print("========================================================")
print(pdf_docs[0])
print("========================================================")
