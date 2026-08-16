from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = Docx2txtLoader("ex_02/doc/tag.docx")
documents = loader.load()

# 문서를 일정 크기의 청크로 쪼개는 splitter 정의
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,      # 청크 하나의 최대 글자 수
    chunk_overlap=200,    # 청크끼리 겹치는 글자 수 (문맥 끊김 방지)
)

# 실제로 쪼개기 실행
chunks = text_splitter.split_documents(documents)

print(f"원본 Document 개수: {len(documents)}")
print(f"쪼개진 청크 개수: {len(chunks)}")
print("---")
print(f"첫 번째 청크 내용:\n{chunks[0].page_content}")
print("---")
print(f"두 번째 청크 내용:\n{chunks[1].page_content}")