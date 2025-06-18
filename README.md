# SWAI (Smart Web AI) 질문 답변 시스템

## 프로젝트 소개
SWAI는 Google Sheets를 통해 입력된 질문을 자동으로 모니터링하고, RAG(Retrieval-Augmented Generation) 시스템을 활용하여 지능적인 답변을 생성하는 시스템입니다.

## 주요 기능
- Google Sheets 실시간 질문 모니터링
- RAG 기반 지능형 답변 생성 (SFR + FAISS + Llama 3.2 8b)
- 자동 답변 저장 및 백업
- 백그라운드 실행 지원
- 디버그 모드 지원

## 시스템 구성
- `rag_system.py`: RAG 기반 답변 생성 시스템
- `question_monitor.py`: Google Sheets 모니터링 및 답변 처리
- `start_monitor.py`: 시스템 실행 및 관리 스크립트
- `crawling/`: 크롤링 데이터 및 FAISS 인덱스 저장소

## 설치 방법

1. 필수 패키지 설치:
```bash
pip install -r requirements.txt
```

2. 에브리타임 계정 설정:
- `crawling/everytime_config.py` 파일에 에브리타임 계정 정보를 입력합니다:
```python
USERNAME = "your_username"
PASSWORD = "your_password"
```

3. 데이터 크롤링:
```bash
cd crawling
# 게시글 URL 수집
python get_url.py
# 게시글 상세 내용 크롤링
python get_detail.py
```

4. 임베딩 생성:
```bash
# 크롤링한 데이터로 FAISS 임베딩 생성
python make_embedding.py
```

## 사용 방법

### 기본 실행
```bash
python start_monitor.py
```

### 추가 실행 옵션
- 테스트 모드: `python start_monitor.py --test`
- 1회 실행: `python start_monitor.py --once`
- 디버그 모드: `python start_monitor.py --debug`
- 백그라운드 실행: `python start_monitor.py --daemon`
- 도움말 보기: `python start_monitor.py --help`

## 시스템 요구사항
- Python 3.8 이상
- CUDA 지원 GPU (RAG 시스템 사용 시)
- 필수 Python 패키지:
  - requests
  - torch
  - transformers (>=4.40.0)
  - sentence-transformers
  - faiss-cpu
  - accelerate
  - bitsandbytes
  - pandas
  - numpy
  - pickle-mixin

## 로그 및 백업
- 실행 로그: `question_monitor.log`
- 답변 백업: `answer_backup.jsonl`

## 문제 해결
1. 시스템 상태 확인: `python start_monitor.py --test`
2. 디버그 로그 확인: `python start_monitor.py --debug`
3. 백업 파일 확인: `answer_backup.jsonl`

## 프로세스 관리
- 실행 중인 프로세스 확인: `ps aux | grep start_monitor`
- 백그라운드 프로세스 종료: `kill $(cat monitor.pid)` 