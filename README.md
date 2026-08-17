# ex_rag — RAG & LangChain 학습 프로젝트

LangChain(LCEL)과 RAG(Retrieval-Augmented Generation)를 단계별로 익히는 예제 모음.
`ex_01` → `ex_02` → `ex_03` 순서로 난이도가 올라간다.

## 환경

- Python 3.14 (`venv/`)
- 주요 패키지: `langchain` 1.3, `langchain-core`, `langchain-openai`, `langchain-ollama`,
  `langchain-community`, `langchain-chroma`, `chromadb`, `docx2txt`, `python-dotenv`, `pydantic`
- `.env`에 `OPENAI_API_KEY` 등 비밀키 보관 (git에는 커밋 안 됨, `.gitignore` 처리)
- 로컬 LLM 실험용으로 [Ollama](https://ollama.com/) 사용 (`ollama serve` + `ollama pull llama3.2:3b` 필요)

## 프로젝트 구조

```
ex_rag/
├── ex_01/
│   └── country_city_chain.py       # LCEL 체인 연결 + 구조화 출력 (RAG 없음)
├── ex_02/
│   ├── doc/tag.docx                 # 원본 문서 (소득세법 관련)
│   ├── load_and_split.py            # 1단계: 문서 로드 + 청크 분할
│   ├── embedding.py                 # 2단계: 임베딩 + Chroma 저장
│   ├── similarity_search.py         # 3단계: 유사도 검색
│   ├── rag_chain.py                 # 4단계: 검색 + LLM 답변 생성 (전체 RAG 체인)
│   └── chroma_db/                   # 벡터 DB 저장소 (git 제외)
└── ex_03/
    ├── doc/tax_with_markdown.docx   # 마크다운 표가 포함된 문서
    ├── markdown_embedding.py        # 마크다운 문서 임베딩 + 저장
    ├── markdown_rag_chain.py        # Query Rewriting 적용 RAG (용어 사전 포함)
    └── markdown_rag_chain_1.py      # Query Rewriting 적용 RAG (용어 사전 없음)
```

## ex_01 — LCEL 체인 연결 (RAG 이전 기초)

**파일:** `country_city_chain.py`

RAG 없이 LangChain의 핵심 개념인 **LCEL(`|` 연산자로 체인 연결)** 과 **구조화 출력**을 익히는 예제.

- `with_structured_output(PydanticModel)`로 LLM 응답을 Pydantic 스키마에 맞춰 강제 파싱
- 1단계 체인(나라 추천) 결과를 `RunnableLambda`로 가공해 2단계 체인(도시 추천)의 입력으로 전달
- 로컬 Ollama(`llama3.2:3b`) 모델 사용 → 무료 실험 가능

```python
full_chain = country_chain | RunnableLambda(to_city_input) | city_chain
```

## ex_02 — 기본 RAG 파이프라인

`.docx` 문서(소득세법)를 기반으로 질문에 답하는 RAG를 4단계로 나눠서 학습.

| 파일 | 단계 | 내용 |
|---|---|---|
| `load_and_split.py` | 1. 로드 + 분할 | `Docx2txtLoader`로 문서 로드 → `RecursiveCharacterTextSplitter`(chunk_size=1500, overlap=200)로 청크 분할 |
| `embedding.py` | 2. 임베딩 + 저장 | `OpenAIEmbeddings(text-embedding-3-small)`로 벡터화 → `Chroma.from_documents`로 `chroma_db/`에 영구 저장 |
| `similarity_search.py` | 3. 검색 | 저장된 Chroma DB를 재로드 → `similarity_search` / `similarity_search_with_score`로 질문과 유사한 청크 검색 |
| `rag_chain.py` | 4. 전체 체인 | retriever + `format_docs` + prompt + LLM(`gpt-5.4-nano`)을 LCEL로 조립한 완성형 RAG |

핵심 패턴 (`rag_chain.py`):

```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

- `RunnablePassthrough()`: 입력(질문)을 그대로 다음 단계로 흘려보냄
- 딕셔너리 형태의 러너블은 하나의 입력을 여러 갈래로 분기시켜 병렬 처리 후 각각 키에 매핑
- Chroma `persist_directory`를 지정하면 디스크에 영구 저장되어 재실행 시 재임베딩 불필요

## ex_03 — 마크다운 표 변환 + Query Rewriting

ex_02보다 발전된 버전. 문서에 **표(table)** 가 포함되어 있고, 사용자의 자연어 질문을
검색에 유리한 키워드로 **재작성(Query Rewriting)** 하는 단계가 추가됨.

| 파일 | 내용 |
|---|---|
| `markdown_embedding.py` | `tax_with_markdown.docx`(표 포함 문서)를 분할·임베딩. `separators=["\n\n제", "\n\n", "\n", " "]`로 법조문 "제○조" 단위/표 구조를 최대한 보존하며 분할 |
| `markdown_rag_chain.py` | Query Rewriting + **용어 사전(`TERM_GLOSSARY`)** 반영 버전 |
| `markdown_rag_chain_1.py` | Query Rewriting만 적용, 용어 사전 없는 버전 (비교용) |

### Query Rewriting 패턴

사용자가 일상어로 질문해도("연봉 5천만원인 사람") 법조문 용어("거주자")로 바꿔
벡터 검색 정확도를 높이는 별도의 작은 체인을 앞단에 붙인다.

```python
TERM_GLOSSARY = {"사람": "거주자"}

rewrite_chain = rewrite_prompt | rewrite_llm | StrOutputParser()

retrieval_chain = (
    RunnableLambda(lambda question: {"question": question})
    | rewrite_chain      # 원 질문 -> 검색용 키워드로 재작성
    | retriever           # 재작성된 검색어로 벡터 검색
    | format_docs
)

rag_chain = (
    {"context": retrieval_chain, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

- 검색(`context`)에는 **재작성된 질문**을 쓰고, 최종 답변 생성(`question`)에는 **원본 질문**을 그대로 전달 → 검색 정확도와 답변 자연스러움을 동시에 확보
- `markdown_rag_chain.py` vs `markdown_rag_chain_1.py`를 비교하면 용어 사전 유무가 검색 결과에 미치는 영향을 확인 가능

## 학습 흐름 요약

1. **ex_01**: RAG 없이 LCEL로 체인 두 개 잇기, 구조화 출력
2. **ex_02**: 문서 로드 → 분할 → 임베딩/저장 → 검색 → 답변 생성까지 RAG 기본기 전 단계
3. **ex_03**: 표가 있는 실전 문서 + Query Rewriting으로 검색 정확도 개선

## 실행 방법

```bash
source venv/bin/activate

# ex_01 (Ollama 필요: ollama serve && ollama pull llama3.2:3b)
python ex_01/country_city_chain.py

# ex_02 (최초 1회만 임베딩 필요)
python ex_02/load_and_split.py
python ex_02/embedding.py
python ex_02/similarity_search.py
python ex_02/rag_chain.py

# ex_03 (최초 1회만 임베딩 필요)
python ex_03/markdown_embedding.py
python ex_03/markdown_rag_chain.py
python ex_03/markdown_rag_chain_1.py
```

> `embedding.py` / `markdown_embedding.py`는 OpenAI API를 호출해 비용이 발생하므로,
> 이미 `chroma_db/`가 생성돼 있다면 다시 실행할 필요 없음.
