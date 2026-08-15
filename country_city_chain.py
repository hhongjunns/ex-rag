# %%
"""
예제 1: LangChain + Ollama(Llama)로 나라 프롬프트 -> 도시 프롬프트를 연결하는 체인

흐름:
1. "특정 테마의 나라를 추천해줘" 프롬프트 실행 -> CountryInfo(Pydantic) 구조화 출력
2. 그 나라 이름을 이어받아 "이 나라의 도시를 추천해줘" 프롬프트 실행 -> CityInfo(Pydantic) 구조화 출력
3. 두 체인을 LCEL(|) 로 연결해서 한 번에 실행
"""

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field


# %%
# 1. 출력 구조를 Pydantic 모델로 정의
# LLM이 이 스키마에 맞춰서 JSON 형태로 답하도록 강제됨 (파싱 실패 걱정 없음)

class CountryInfo(BaseModel):
    country: str = Field(description="추천하는 나라 이름 (한글)")
    reason: str = Field(description="이 나라를 추천하는 이유, 한 문장")


class CityInfo(BaseModel):
    city: str = Field(description="추천하는 도시 이름 (한글)")
    description: str = Field(description="이 도시에 대한 간단한 소개, 한두 문장")


# %%
# 2. 로컬 Ollama 모델 연결
# ollama serve 가 백그라운드에서 돌고 있어야 하고, 모델은 미리 pull 되어 있어야 함
# ex) ollama pull llama3.2:3b

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0.7,
)

# with_structured_output: LLM의 응답을 Pydantic 모델 형태로 강제 파싱해줌
country_llm = llm.with_structured_output(CountryInfo)
city_llm = llm.with_structured_output(CityInfo)


# %%
# 3. 나라 프롬프트 (ChatPromptTemplate.from_messages 사용)

country_prompt = ChatPromptTemplate.from_messages([
    ("system", "너는 여행 전문가야. 사용자가 준 테마에 맞는 나라를 딱 하나만 추천해."),
    ("human", "테마: {theme}"),
])

country_chain = country_prompt | country_llm


# %%
# 4. 도시 프롬프트 (앞선 나라 결과를 입력으로 받음)

city_prompt = ChatPromptTemplate.from_messages([
    ("system", "너는 여행 전문가야. 주어진 나라 안에서 여행하기 좋은 도시를 딱 하나만 추천해."),
    ("human", "나라: {country}\n이 나라를 추천한 이유: {reason}"),
])

city_chain = city_prompt | city_llm


# %%
# 5. 두 체인을 LCEL로 연결
# country_chain 결과(CountryInfo 객체) -> dict로 변환 -> city_chain 입력으로 전달

def to_city_input(country_info: CountryInfo) -> dict:
    return {
        "country": country_info.country,
        "reason": country_info.reason,
    }


full_chain = country_chain | RunnableLambda(to_city_input) | city_chain


# %%
# 6. 실행

if __name__ == "__main__":
    theme = "겨울에 눈 구경하기 좋은 곳"

    print(f"[입력 테마] {theme}\n")

    # 중간 결과(나라)도 보고 싶으면 country_chain만 따로 실행해서 확인 가능
    country_result = country_chain.invoke({"theme": theme})
    print(f"[1단계: 나라 추천] {country_result.country}")
    print(f"  이유: {country_result.reason}\n")

    # 전체 체인 실행 (나라 -> 도시로 자동 연결)
    city_result = full_chain.invoke({"theme": theme})
    print(f"[2단계: 도시 추천] {city_result.city}")
    print(f"  소개: {city_result.description}")