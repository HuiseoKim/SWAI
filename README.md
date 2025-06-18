# SWAI (Smart Web AI) - 대학교 정보 검색 챗봇

## 프로젝트 소개
SWAI는 대학교 내의 정보를 자연어로 쉽고 빠르게 검색할 수 있는 챗봇 서비스입니다. 기존 에브리타임에서는 정확한 키워드나 검색어로만 정보를 찾을 수 있었던 한계를 극복하여, 과목명이나 용어를 줄여 쓰거나 추상적인 표현으로도 원하는 정보를 찾을 수 있습니다.

예를 들어, "컴아(컴퓨터아키텍쳐) 수업 난이도 어때?"와 같은 자연어 질문을 하면, 관련된 게시글들을 찾아 맥락에 맞는 답변을 제공합니다.

## 시스템 아키텍처

### 1. 데이터 수집 파이프라인
- **자동 크롤링 시스템**
  - 24시간 주기로 컴퓨터과학과 게시판 자동 크롤링
  - 새벽 시간대(한국 시간 기준) 실행으로 서버 부하 최소화
  - 증분 업데이트: 새로운 게시물만 필터링하여 효율적인 데이터 수집
  
- **데이터 저장 및 관리**
  - 게시물 메타데이터(작성시간, URL, 조회수 등) 함께 저장
  - Data concurrency 보장
  - 자동 백업 및 복구 시스템 구축

### 2. 데이터 처리 및 검색 시스템
- **텍스트 임베딩**
  - SFR-7B 모델 활용한 고성능 텍스트 임베딩
  - 문맥을 고려한 의미론적 벡터 변환
  - 다국어(한영) 처리 지원

- **검색 엔진**
  - FAISS 기반 벡터 유사도 검색
  - 코사인 유사도 메트릭 사용
  - Top-3 문서 선정
  - 빠른 응답 속도를 위한 reranking 생략
  - 일일 단위 인덱스 자동 업데이트

### 3. RAG(Retrieval-Augmented Generation) 시스템
- **문서 검색**
  - 사용자 질문 임베딩 변환
  - 실시간 유사도 기반 문서 검색
  - 컨텍스트 윈도우 최적화

- **답변 생성**
  - Llama 8B instruct 모델 사용
  - 검색된 문서 기반 답변 생성
  - 참고 문서 URL 자동 첨부
  - 답변 품질 모니터링 및 로깅

## 기술 스택
- **모델**
  - 임베딩: SFR-7B (Salesforce Research)
  - LLM: Llama 8B instruct
  - 벡터 DB: FAISS 사용

- **크롤링**
  - Beautiful Soup 4
  - Selenium
  - Chrome WebDriver

## 성능 분석 및 지표

### 1. 사용자 통계
- **접속 현황**
  - 총 접속자: 138명
  - 실사용자: 54명 (39.1%)
  - 3회 이상 재사용: 중간발표 대비 30% 증가
  - 재방문 사용자: 13명 (24% 재방문율)

- **사용 패턴**
  - 평균 세션 시간: 3분 30초
  - 사용자당 평균 질문 수: 2.7개
  - 피크 시간대: 오후 2시 ~ 6시

### 2. 퍼널 분석 (컴과 500명 기준)
- **Acquisition**: 27.6%

- **Activation**: 39%

- **Retention**: 24%

## 시스템 구성
- `rag_system.py`: RAG 기반 답변 생성 시스템
  - 임베딩 모델 관리
  - 문서 검색 로직
  - LLM 추론 파이프라인

- `question_monitor.py`: 질문 처리 시스템
  - 실시간 질문 모니터링
  - 답변 생성 및 전송
  - 로깅 및 분석

- `start_monitor.py`: 시스템 관리
  - 서비스 시작/중지
  - 상태 모니터링
  - 에러 핸들링

- `crawling/`: 크롤링 시스템
  - URL 수집기
  - 콘텐츠 추출기
  - 데이터 전처리

## 설치 및 실행 가이드

### 1. 환경 설정 (필수는 아님)
```bash
# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# 필수 패키지 설치
pip install -r requirements.txt
```

### 2. 에브리타임 계정 설정
```python
# crawling/everytime_config.py
USERNAME = "your_username"
PASSWORD = "your_password"
```

### 3. 데이터 수집
```bash
cd crawling

# 게시글 URL 수집
python get_url.py

# 게시글 상세 내용 크롤링
python get_detail.py
```

### 4. 임베딩 생성
```bash
# FAISS 인덱스 생성
python make_embedding.py
```

### 5. 서비스 실행
```bash
# 기본 실행
python start_monitor.py

# 디버그 모드
python start_monitor.py --debug
```

## 주의사항 및 제한사항
- **시스템 요구사항**
  - CUDA 지원 GPU (최소 24GB VRAM)
  - Python 3.8 이상
  - 최소 16GB RAM
  - 50GB 이상 저장공간

- **성능 고려사항**
  - GPU 서버에서 8B 규모의 LLM을 구동하므로 첫 응답에 5-10초 소요
  - 동시 사용자 처리를 위한 큐잉 시스템 구현
  - 캐시를 통한 반복 질문 최적화

## 향후 계획
- 다중 게시판 지원 확장
- 실시간 답변 속도 최적화
- 사용자 피드백 기반 답변 품질 개선
- 멀티 GPU 지원 추가 
