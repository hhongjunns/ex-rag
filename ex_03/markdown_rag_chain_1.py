# %%
"""
RAG 활용법
4. 질문이 있을 때, 벡터 데이터베이스에 유사도 검색
5. 유사도 검색으로 가져온 문서를 LLM 질문과 같이 전달
"""
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# %%
# 1. API 키 로드 + 저장된 Chroma DB 불러오기 (임베딩은 저장할 때와 동일한 모델 사용)

load_dotenv()

PERSIST_DIR = "ex_03/chroma_db"

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
)

vectorstore = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings,
    collection_name="tag_markdown_docs",
)

# %%
# 2. retriever 만들기

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3},   # 검색 결과 상위 3개
)

# %%
# 3. 검색된 문서 리스트(List[Document])를, LLM 프롬프트에 넣을 하나의 문자열로 합치는 함수

def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


# %%
# 4. 답변 생성용 프롬프트 (기존과 동일)

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "너는 대한민국 소득세법을 아는 전문가야. "
     "문서에 없는 내용은 모른다고 답해. 답변은 한국어로 해.\n\n"
     "문서 내용:\n{context}"),
    ("human", "{question}"),
])

# %%
# 5. LLM 준비 (답변 생성용)

llm = ChatOpenAI(
    model="gpt-5.4-nano",
    temperature=0,
)

# %%
# 6. [신규] 질문 재작성(Query Rewriting)용 프롬프트 + LLM
#
# 사용자의 자연어 질문을, 법조문/표 스타일 문서와 벡터 거리가 가까워지도록
# 검색하기 좋은 키워드 형태로 바꿔주는 별도의 작은 체인

rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "사용자의 질문을, 법조문에서 관련 세율표나 조항을 검색하기 좋은 "
     "핵심 키워드 형태로 바꿔줘. 다른 설명 없이 검색어만 출력해."),
    ("human", "{question}"),
])

rewrite_llm = ChatOpenAI(
    model="gpt-5.4-nano",
    temperature=0,
)

rewrite_chain = rewrite_prompt | rewrite_llm | StrOutputParser()

# %%
# 7. [신규] "질문 -> 재작성된 검색어 -> 검색 -> context" 로 이어지는 retrieval 파이프
#
# RunnableLambda(lambda x: {"question": x}): 문자열(질문)을 rewrite_chain이 원하는
#   dict 형태({"question": ...})로 감싸주는 어댑터
# 그 뒤에 rewrite_chain을 거쳐 재작성된 검색어가 나오면, 그걸 retriever에 넣어 검색

retrieval_chain = (
    RunnableLambda(lambda question: {"question": question})
    | rewrite_chain          # 원래 질문 -> 재작성된 검색어(문자열)
    | retriever               # 재작성된 검색어로 벡터 검색 -> Document 리스트
    | format_docs             # Document 리스트 -> 하나의 문자열(context)
)

# %%
# 8. 전체 체인을 LCEL로 조립
#
# {"context": retrieval_chain, "question": RunnablePassthrough()}
#   -> 원래 질문 하나를 입력하면, 동시에 두 갈래로 나뉨
#      (1) retrieval_chain 통과: 질문 재작성 -> 검색 -> context 문자열
#      (2) 그대로 question 으로 사용 (LLM에게는 원래 질문 그대로 보여줌)

rag_chain = (
    {"context": retrieval_chain, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# %%
# 9. 실행

if __name__ == "__main__":
    question = "연봉 5천만원의 직장인의 소득세는 얼마인가요?"

    print(f"[원래 질문] {question}\n")

    # 재작성된 검색어가 뭐였는지 별도로 확인 (디버깅용)
    rewritten_query = rewrite_chain.invoke({"question": question})
    print(f"[재작성된 검색어] {rewritten_query}\n")

    # 재작성된 검색어로 실제 검색된 문서 확인 (디버깅용)
    docs = retriever.invoke(rewritten_query)
    for i, doc in enumerate(docs, 1):
        print(f"--- 검색된 문서 {i} ---")
        print(doc.page_content)
        print()

    # 전체 체인 실행 (질문 재작성 -> 검색 -> 답변 생성까지 한 번에)
    answer = rag_chain.invoke(question)

    print(f"[답변]\n{answer}")