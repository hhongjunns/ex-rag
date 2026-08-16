from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# %%
# 1. API 키 로드

load_dotenv()

# %%
# 2. 이전 단계에서 저장해둔 Chroma DB 다시 불러오기
# 주의: 여기서는 새로 embed하지 않고, 저장된 벡터를 그대로 불러오는 것
#       (embedding 객체는 "나중에 질문을 벡터화할 때" 쓰기 위해 필요)

PERSIST_DIR = "ex_02/chroma_db"

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
)

vectorstore = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings,
    collection_name="tag_docs",   # 저장할 때 썼던 이름과 동일해야 함
)

print(f"불러온 벡터 개수: {vectorstore._collection.count()}\n")

# %%
# 3. 질문으로 유사도 검색

query = "연봉 5000 만원 직장인의 소득세는 얼마인가요?"

# similarity_search: 질문과 가장 유사한 청크 k개를 반환
results = vectorstore.similarity_search(query, k=3)

print(f"[질문] {query}\n")
for i, doc in enumerate(results, start=1):
    print(f"--- 검색 결과 {i} ---")
    print(doc.page_content)
    print()

# %%
# 4. (참고) 유사도 점수까지 같이 보고 싶으면 이 함수 사용
# 점수가 낮을수록(거리가 가까울수록) 더 유사함

results_with_score = vectorstore.similarity_search_with_score(query, k=3)

print("=== 유사도 점수 포함 ===\n")
for i, (doc, score) in enumerate(results_with_score, start=1):
    print(f"--- 검색 결과 {i} (거리: {score:.4f}) ---")
    print(doc.page_content)
    print()