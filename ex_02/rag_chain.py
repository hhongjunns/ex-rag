from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# %%
# 1. API 키 로드 + 저장된 Chroma DB 불러오기 (임베딩은 저장할 때와 동일한 모델 사용)

load_dotenv()

PERSIST_DIR = "ex_02/chroma_db"

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
)

vectorstore = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings,
    collection_name="tag_docs",
)

# %%
# 2. retriever 만들기
# vectorstore.similarity_search()를 직접 호출하는 대신,
# "retriever"라는 Runnable 형태로 감싸두면 체인(|)에 바로 끼울 수 있음

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3},   # 검색 결과 상위 3개
)

# %%
# 3. 검색된 문서 리스트(List[Document])를, LLM 프롬프트에 넣을 하나의 문자열로 합치는 함수

def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


# %%
# 4. 프롬프트 정의 (ChatPromptTemplate.from_messages 사용)

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "너는 대한민국 소득세법을 아는 전문가야. "
     "문서에 없는 내용은 모른다고 답해. 답변은 한국어로 해.\n\n"
     "문서 내용:\n{context}"),
    ("human", "{question}"),
])

# %%
# 5. LLM 준비 (로컬 Ollama, 답변 생성은 무료)

llm = ChatOpenAI(
    model="gpt-5.4-nano",
    temperature=0,
)

# %%
# 6. 전체 체인을 LCEL로 조립
#
# RunnablePassthrough(): 입력값(질문)을 그대로 다음 단계로 흘려보냄
# {"context": retriever | format_docs, "question": RunnablePassthrough()}
#   -> 질문 하나를 입력하면, 동시에 두 갈래로 나뉘어서
#      (1) retriever에 넣어 검색 -> format_docs로 문자열 변환 -> context
#      (2) 그대로 question 으로 사용
#   -> 이 두 값을 prompt 의 {context}, {question} 자리에 채워넣음

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()   # LLM 응답(메시지 객체)에서 순수 텍스트만 뽑아냄
)

# %%
# 7. 실행

if __name__ == "__main__":
    question = "연봉 5천만원의 직장인의 소득세는 얼마인가요?"

    print(f"[질문] {question}\n")

    answer = rag_chain.invoke(question)
    docs = retriever.invoke(question)
    for i, doc in enumerate(docs, 1):
        print(f"--- 검색된 문서 {i} ---")
        print(doc.page_content)
        print()

    print(f"[답변]\n{answer}")