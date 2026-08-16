import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# %%
# 1. .env 파일에서 OPENAI_API_KEY 불러오기
load_dotenv()

# %%
# 2. 이전 단계 그대로: 문서 로드 + 청크 분할

loader = Docx2txtLoader("ex_02/doc/tag.docx")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200,
)
chunks = text_splitter.split_documents(documents)

print(f"쪼개진 청크 개수: {len(chunks)}\n")

# %%
# 3. 임베딩 모델 준비
# text-embedding-3-small: 저렴하고 성능 좋은 OpenAI 임베딩 모델

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
)

# %%
# 4. Chroma에 전체 청크 저장
# from_documents: 청크 리스트 전체를 받아서, 내부적으로 하나씩 임베딩 -> 벡터DB에 저장
# persist_directory: 이 경로에 디스크로 실제 저장됨 (지정 안 하면 메모리에만 있다가 프로그램 끄면 사라짐)
 
PERSIST_DIR = "ex_02/chroma_db"
 
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=PERSIST_DIR,
    collection_name="tag_docs",   # DB 안에서 이 데이터셋을 구분하는 이름
)
 
print(f"\nChroma 저장 완료! 경로: {PERSIST_DIR}")
print(f"저장된 벡터 개수: {vectorstore._collection.count()}")
 