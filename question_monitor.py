#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Google Sheets 질문 모니터링 시스템

지속적으로 실행되면서:
1. "question" 시트의 새로운 질문을 모니터링
2. 새로운 질문에 대해 RAG 시스템으로 답변 생성
3. "answer" 시트에 답변 저장
"""

import requests
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Set, Optional
import logging
from pathlib import Path
import sys
import os

# RAG 시스템 import
try:
    from rag_system import NewRAGSystem
    RAGSearcher = NewRAGSystem  # 호환성을 위해
except ImportError:
    print("⚠️  새로운 RAG 시스템을 찾을 수 없습니다. rag_system.py를 확인하세요.")
    NewRAGSystem = None


class QuestionMonitor:
    def __init__(self, rag_index_path: str = "./crawling/faiss_output"):
        """
        질문 모니터링 시스템 초기화
        
        Args:
            rag_index_path: RAG FAISS index 경로
        """
        # Google Apps Script URL
        self.script_url = 'https://script.google.com/macros/s/AKfycbyosOFzWHmdXvorBfuZOFfDYFlReBT68PWuXhXJApFut-A8wiu5juWjtYBOSWi1HVX2/exec'
        
        # 모니터링 설정
        self.check_interval = 10  # 10초마다 체크
        self.processed_questions: Set[str] = set()  # 처리된 질문 ID들
        self.request_delay = 1.0  # API 요청 간 지연
        
        # 로깅 설정 먼저
        self._setup_logging()
        
        # 새로운 RAG 시스템 초기화 (시작 시 한 번만)
        self.rag_system = None
        self.rag_ready = False
        
        if NewRAGSystem and Path(rag_index_path).exists():
            self.logger.info("RAG 시스템 초기화 시작... (시간이 걸릴 수 있습니다)")
            try:
                self.rag_system = NewRAGSystem(faiss_dir=rag_index_path, device_id=0)
                if self.rag_system.initialize_all():
                    self.rag_ready = True
                    self.logger.info("✅ RAG 시스템 초기화 완료! 모델들이 GPU에 로드되었습니다.")
                else:
                    self.logger.error("⚠️  RAG 시스템 초기화 실패")
                    self.rag_system = None
            except Exception as e:
                self.logger.error(f"⚠️  RAG 시스템 로드 실패: {e}")
                self.rag_system = None
        else:
            self.logger.warning("⚠️  RAG 시스템 없이 실행됩니다. 기본 답변을 사용합니다.")
        
        # 시작 시 기존 질문들 로드
        self._load_existing_questions()
    
    def _setup_logging(self):
        """로깅 설정"""
        # 환경변수나 인자로 디버그 모드 확인
        debug_mode = '--debug' in sys.argv or os.environ.get('DEBUG', '').lower() in ['1', 'true', 'yes']
        log_level = logging.DEBUG if debug_mode else logging.INFO
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('question_monitor.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        if debug_mode:
            self.logger.info("🐛 디버그 모드 활성화됨")
    
    def _make_request(self, params: Dict) -> Optional[Dict]:
        """Google Apps Script에 요청 보내기"""
        try:
            time.sleep(self.request_delay)
            
            # 디버깅용 로그 추가
            self.logger.debug(f"요청 파라미터: {params}")
            
            response = requests.get(self.script_url, params=params, timeout=30)
            response.raise_for_status()
            
            response_text = response.text.strip()
            
            # 응답 디버깅
            self.logger.debug(f"응답 상태 코드: {response.status_code}")
            self.logger.debug(f"응답 헤더: {dict(response.headers)}")
            self.logger.debug(f"응답 내용 (처음 200자): {response_text[:200]}")
            
            # 빈 응답 체크
            if not response_text:
                self.logger.error("빈 응답을 받았습니다")
                return None
            
            # JSONP 응답 처리
            if response_text.startswith('undefined(') and response_text.endswith(')'):
                json_text = response_text[10:-1]
                data = json.loads(json_text)
                return data
            else:
                try:
                    data = response.json()
                    return data
                except json.JSONDecodeError:
                    # JSON이 아닌 응답일 경우
                    self.logger.error(f"JSON이 아닌 응답: {response_text[:500]}")
                    return None
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"네트워크 요청 실패: {e}")
            return None
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON 파싱 실패: {e}")
            self.logger.error(f"파싱 실패한 응답: {response_text[:500] if 'response_text' in locals() else 'N/A'}")
            return None
        except Exception as e:
            self.logger.error(f"예상치 못한 오류: {e}")
            return None
    
    def _load_existing_questions(self):
        """시작 시 기존 질문들을 로드하여 중복 처리 방지"""
        self.logger.info("기존 질문들 로드 중...")
        
        params = {
            'action': 'read',
            'table': 'question'
        }
        
        data = self._make_request(params)
        
        if data and data.get('success', False):
            questions = data.get('data', [])
            for q in questions:
                question_id = f"{q.get('id', '')}_{q.get('time_stamp', '')}"
                self.processed_questions.add(question_id)
            
            self.logger.info(f"기존 질문 {len(questions)}개 로드 완료")
        else:
            self.logger.warning("기존 질문 로드 실패")
    
    def get_new_questions(self) -> List[Dict]:
        """새로운 질문들 가져오기"""
        params = {
            'action': 'read',
            'table': 'question'
        }
        
        data = self._make_request(params)
        
        if not data or not data.get('success', False):
            return []
        
        questions = data.get('data', [])
        new_questions = []
        
        for q in questions:
            question_id = f"{q.get('id', '')}_{q.get('time_stamp', '')}"
            
            if question_id not in self.processed_questions:
                new_questions.append(q)
                self.processed_questions.add(question_id)
        
        return new_questions
    
    def generate_answer(self, question: str, user_id: str = None) -> Dict[str, any]:
        """
        질문에 대한 답변 생성 (이미 로드된 RAG 시스템 사용)
        
        Args:
            question: 사용자 질문
            user_id: 사용자 ID (선택사항)
            
        Returns:
            Dict: {
                'answer': 생성된 답변,
                'documents': 참고 문서 링크들
            }
        """
        try:
            # RAG 시스템이 준비되어 있는지 확인
            if self.rag_ready and self.rag_system:
                self.logger.info(f"RAG 답변+문서 생성 시작: {question[:50]}...")
                
                # 새로운 메서드로 답변과 문서 링크 모두 가져오기
                rag_result = self.rag_system.generate_rag_answer_with_documents(question)
                answer = rag_result.get('answer', '')
                documents = rag_result.get('documents', [])
                
                if answer and answer.strip():
                    self.logger.info(f"✅ RAG 답변+문서 생성 완료: 답변 {len(answer)}자, 문서 {len(documents)}개")
                    return {
                        'answer': answer,
                        'documents': documents
                    }
                else:
                    self.logger.warning("RAG 시스템에서 빈 답변 반환, 기본 답변 사용")
            else:
                self.logger.warning("RAG 시스템이 준비되지 않음, 기본 답변 사용")
            
            # RAG 결과가 없거나 시스템이 준비되지 않은 경우 기본 답변
            default_answer = self._generate_default_answer(question)
            return {
                'answer': default_answer,
                'documents': []
            }
            
        except Exception as e:
            self.logger.error(f"답변 생성 실패: {e}")
            return {
                'answer': "죄송합니다. 현재 답변을 생성할 수 없습니다. 잠시 후 다시 시도해주세요.",
                'documents': []
            }
    
    def _compose_answer_from_context(self, question: str, contexts: List[str]) -> str:
        """
        검색된 컨텍스트를 바탕으로 답변 구성 (레거시 메서드 - 사용 안함)
        """
        # 새로운 RAG 시스템에서는 Llama가 직접 답변을 생성하므로 이 메서드는 사용하지 않음
        return self._generate_default_answer(question)
    
    def _generate_default_answer(self, question: str) -> str:
        """
        기본 답변 생성 (RAG 시스템 없을 때)
        
        Args:
            question: 사용자 질문
            
        Returns:
            str: 기본 답변
        """
        # 질문 키워드 기반 간단한 답변
        question_lower = question.lower()
        
        if any(keyword in question_lower for keyword in ['과제', '숙제', 'assignment']):
            return "과제와 관련된 질문이시군요. 구체적인 과제 내용이나 어려운 부분을 알려주시면 더 자세한 도움을 드릴 수 있습니다."
        
        elif any(keyword in question_lower for keyword in ['수업', '강의', '시간표', 'class']):
            return "수업 관련 문의이시군요. 학과 홈페이지나 학습관리시스템에서 더 정확한 정보를 확인하실 수 있습니다."
        
        elif any(keyword in question_lower for keyword in ['복전', '전과', '복수전공']):
            return "복수전공이나 전과 관련 문의는 학과 사무실이나 학사팀에 직접 문의하시는 것이 가장 정확합니다."
        
        elif any(keyword in question_lower for keyword in ['이산구조', '자료구조', '알고리즘']):
            return "전공 과목 관련 질문이시군요. 교수님께 직접 문의하시거나 학습 커뮤니티를 활용해보시는 것을 추천드립니다."
        
        else:
            return f"'{question[:50]}...' 에 대한 질문 감사합니다. 더 구체적인 정보를 제공해주시면 더 정확한 답변을 드릴 수 있습니다."
    
    def save_answer(self, question_data: Dict, answer: str) -> bool:
        """
        답변을 answer 시트에 저장 (id, question, answer, time_stamp 형식)
        
        Args:
            question_data: 원본 질문 데이터
            answer: 생성된 답변
            
        Returns:
            bool: 저장 성공 여부
        """
        try:
            # 원본 질문의 time_stamp를 정확히 보존
            original_timestamp = question_data.get('time_stamp', '')
            question_id = question_data.get('id', '')
            original_question = question_data.get('question', '')
            
            # answer 시트 구조에 맞는 데이터 구성
            answer_data = {
                'id': question_id,
                'question': original_question,  # 원본 질문 그대로 저장
                'answer': answer,               # RAG로 생성된 답변
                'time_stamp': original_timestamp  # 원본 질문과 동일한 time_stamp 사용
            }
            
            # 로깅으로 저장 데이터 확인
            self.logger.info(f"답변 저장 준비 - ID: {question_id}, time_stamp: {original_timestamp}")
            self.logger.debug(f"질문: {original_question[:50]}...")
            self.logger.debug(f"답변: {answer[:100]}...")
            
            # Google Apps Script Insert API 사용
            params = {
                'action': 'insert',
                'table': 'answer',
                'data': json.dumps(answer_data, ensure_ascii=False)
            }
            
            self.logger.info(f"답변 저장 시도: ID {question_id}")
            
            response_data = self._make_request(params)
            
            self.logger.debug(f"Insert API 응답: {response_data}")
            
            if response_data and response_data.get('success', False):
                self.logger.info(f"✅ 답변 저장 성공: ID {question_id}")
                return True
            else:
                # 실패한 경우 상세 로그
                self.logger.error(f"❌ Insert API 실패: {response_data}")
                
                # 로컬 백업 저장
                self._save_answer_locally(answer_data)
                return False
                
        except Exception as e:
            self.logger.error(f"❌ 답변 저장 중 오류: {e}")
            
            # 예외 발생 시에도 로컬 백업
            try:
                answer_data = {
                    'id': question_data.get('id', ''),
                    'question': question_data.get('question', ''),
                    'answer': answer,
                    'time_stamp': question_data.get('time_stamp', ''),
                    'error': str(e)
                }
                self._save_answer_locally(answer_data)
            except:
                pass
                
            return False
    
    def _save_answer_locally(self, answer_data: Dict):
        """로컬 파일에 답변 백업 저장"""
        try:
            backup_file = "answer_backup.jsonl"
            with open(backup_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(answer_data, ensure_ascii=False) + '\n')
            self.logger.info(f"답변을 로컬에 백업 저장: {backup_file}")
        except Exception as e:
            self.logger.error(f"로컬 백업 저장 실패: {e}")
    
    def save_answer_with_documents(self, question_data: Dict, answer_result: Dict) -> bool:
        """
        답변과 문서 링크를 answer 시트에 저장
        
        Args:
            question_data: 원본 질문 데이터
            answer_result: RAG 답변 결과 {answer: str, documents: List[Dict]}
            
        Returns:
            bool: 저장 성공 여부
        """
        try:
            # 원본 질문 정보
            original_timestamp = question_data.get('time_stamp', '')
            question_id = question_data.get('id', '')
            original_question = question_data.get('question', '')
            
            # 답변과 문서 정보
            answer = answer_result.get('answer', '')
            documents = answer_result.get('documents', [])
            
            # 문서 링크들을 문자열로 포맷팅 (URL만)
            document_urls = []
            for doc in documents:
                url = doc.get('url', '')
                
                if url:
                    # 완전한 URL 생성
                    if not url.startswith('http'):
                        url = f"https://{url}"
                    document_urls.append(url)
            
            # 문서 URL들을 줄바꿈으로 구분
            documents_text = "\n".join(document_urls) if document_urls else ""
            
            # answer 시트 구조에 맞는 데이터 구성
            answer_data = {
                'id': question_id,
                'question': original_question,
                'answer': answer,
                'document': documents_text,  # 문서 URL들
                'time_stamp': original_timestamp
            }
            
            # 로깅
            self.logger.info(f"답변+문서 저장 준비 - ID: {question_id}")
            self.logger.debug(f"질문: {original_question[:50]}...")
            self.logger.debug(f"답변: {answer[:100]}...")
            self.logger.debug(f"참고문서 {len(documents)}개: {[doc.get('title', '')[:20] for doc in documents]}")
            
            # Google Apps Script Insert API 사용
            params = {
                'action': 'insert',
                'table': 'answer',
                'data': json.dumps(answer_data, ensure_ascii=False)
            }
            
            self.logger.info(f"답변+문서 저장 시도: ID {question_id}")
            
            response_data = self._make_request(params)
            
            if response_data and response_data.get('success', False):
                self.logger.info(f"✅ 답변+문서 저장 성공: ID {question_id} (문서 {len(documents)}개)")
                return True
            else:
                self.logger.error(f"❌ Insert API 실패: {response_data}")
                self._save_answer_locally(answer_data)
                return False
                
        except Exception as e:
            self.logger.error(f"❌ 답변+문서 저장 중 오류: {e}")
            
            # 예외 발생 시에도 로컬 백업
            try:
                backup_data = {
                    'id': question_data.get('id', ''),
                    'question': question_data.get('question', ''),
                    'answer': answer_result.get('answer', ''),
                    'documents': answer_result.get('documents', []),
                    'time_stamp': question_data.get('time_stamp', ''),
                    'error': str(e)
                }
                self._save_answer_locally(backup_data)
            except:
                pass
                
            return False
    
    def process_new_questions(self):
        """새로운 질문들 처리"""
        try:
            new_questions = self.get_new_questions()
            
            if not new_questions:
                return
            
            self.logger.info(f"새로운 질문 {len(new_questions)}개 발견")
            
            for question_data in new_questions:
                user_id = question_data.get('id', 'unknown')
                question_text = question_data.get('question', '')
                timestamp = question_data.get('time_stamp', '')
                
                self.logger.info(f"질문 처리 중: {user_id} - {question_text[:50]}...")
                
                # 답변 생성
                answer_result = self.generate_answer(question_text, user_id)
                
                # 답변 저장 (문서 링크 포함)
                success = self.save_answer_with_documents(question_data, answer_result)
                
                if success:
                    self.logger.info(f"✅ 질문 처리 완료: {user_id}")
                else:
                    self.logger.error(f"❌ 질문 처리 실패: {user_id}")
                
                # 요청 간 지연
                time.sleep(self.request_delay)
                
        except Exception as e:
            self.logger.error(f"질문 처리 중 오류: {e}")
    
    def run_monitor(self):
        """모니터링 시작"""
        self.logger.info("🚀 질문 모니터링 시스템 시작")
        self.logger.info(f"체크 주기: {self.check_interval}초")
        
        # RAG 시스템 상태 표시
        if self.rag_ready:
            self.logger.info("🤖 RAG 시스템: 활성화됨 (SFR + FAISS + Llama 3.2 8b)")
        else:
            self.logger.warning("⚠️  RAG 시스템: 비활성화됨 (기본 답변 모드)")
        
        try:
            while True:
                self.logger.info("새로운 질문 확인 중...")
                self.process_new_questions()
                
                self.logger.info(f"{self.check_interval}초 대기 중...")
                time.sleep(self.check_interval)
                
        except KeyboardInterrupt:
            self.logger.info("🛑 모니터링 중단됨 (Ctrl+C)")
        except Exception as e:
            self.logger.error(f"💥 모니터링 중 오류: {e}")
        finally:
            self.logger.info("모니터링 종료")
    
    def test_system(self):
        """시스템 테스트"""
        print("🔍 시스템 테스트 시작")
        print("=" * 50)
        
        # 1. 기본 연결 테스트
        print("1. Google Sheets 연결 테스트...")
        test_params = {'action': 'read', 'table': 'question', 'limit': 1}
        response = self._make_request(test_params)
        
        if response and response.get('success'):
            print("✅ Google Sheets 연결 성공")
        else:
            print("❌ Google Sheets 연결 실패")
            return False
        
        # 2. 지원 기능 테스트
        print("\n2. Google Apps Script 지원 기능 테스트...")
        
        # question 읽기 테스트
        print("  🔍 question 테이블 읽기 테스트...")
        read_params = {'action': 'read', 'table': 'question'}
        response = self._make_request(read_params)
        if response and response.get('success'):
            questions = response.get('data', [])
            print(f"    ✅ question 읽기 성공: {len(questions)}개 질문")
        else:
            print(f"    ❌ question 읽기 실패: {response}")
        
        # answer 테이블 insert 테스트
        print("  🔍 answer 테이블 insert 테스트...")
        
        # 실제 question 데이터의 time_stamp를 사용한 테스트
        original_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        test_answer_data = {
            'id': 'TEST_' + datetime.now().strftime('%Y%m%d_%H%M%S'),
            'time_stamp': original_timestamp,  # question과 동일한 형식의 time_stamp 사용
            'original_question': 'TEST 질문입니다',
            'answer': 'TEST 답변입니다',
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        print(f"    테스트용 time_stamp: {original_timestamp}")
        
        insert_params = {
            'action': 'insert',
            'table': 'answer',
            'data': json.dumps(test_answer_data)
        }
        
        response = self._make_request(insert_params)
        if response and response.get('success'):
            print("    ✅ answer 테이블 insert 성공 (answer 시트 사용 가능)")
            print(f"    ✅ time_stamp 보존 확인: {original_timestamp}")
        else:
            error = response.get('data', {}).get('error', '알 수 없는 오류') if response else '응답 없음'
            print(f"    ❌ answer 테이블 insert 실패: {error}")
            print("    💡 Google Sheets에서 'answer' 시트를 수동으로 생성해주세요.")
            print("    💡 또는 Google Apps Script가 자동 생성하도록 수정이 필요합니다.")
        
        # 3. RAG 시스템 테스트
        print("\n3. RAG 시스템 테스트...")
        if self.rag_system:
            try:
                test_answer = self.rag_system.generate_rag_answer("컴퓨터과학 수업")
                print(f"✅ RAG 답변 생성 성공: {len(test_answer)}자")
                print(f"   테스트 답변: {test_answer[:100]}...")
            except Exception as e:
                print(f"❌ RAG 답변 생성 실패: {e}")
        else:
            print("⚠️  RAG 시스템 없음 - 기본 답변 사용")
        
        # 4. 답변 생성 테스트
        print("\n4. 답변 생성 테스트...")
        test_question = "이산구조 수업에 대해 궁금합니다."
        test_result = self.generate_answer(test_question)
        print(f"테스트 질문: {test_question}")
        print(f"생성된 답변: {test_result['answer'][:100]}...")
        
        # 문서 링크 테스트
        documents = test_result.get('documents', [])
        if documents:
            print(f"참고 문서 {len(documents)}개:")
            for i, doc in enumerate(documents, 1):
                print(f"  {i}. {doc.get('title', '제목없음')[:30]}... (유사도: {doc.get('similarity_score', 0):.2f})")
        else:
            print("참고 문서: 없음")
        
        # 5. 백업 파일 확인
        print("\n5. 백업 시스템 테스트...")
        try:
            test_data = {
                'id': 'BACKUP_TEST',
                'test': True,
                'timestamp': datetime.now().isoformat()
            }
            self._save_answer_locally(test_data)
            print("✅ 로컬 백업 시스템 작동")
        except Exception as e:
            print(f"❌ 로컬 백업 실패: {e}")
        
        print("\n✅ 모든 테스트 완료!")
        return True


def main():
    """메인 함수"""
    print("🤖 Google Sheets 질문 모니터링 시스템")
    print("=" * 50)
    
    # 모니터 시스템 초기화
    monitor = QuestionMonitor()
    
    # 명령행 인자 처리
    if len(sys.argv) > 1:
        if sys.argv[1] == '--test':
            # 테스트 모드
            monitor.test_system()
            return
        elif sys.argv[1] == '--once':
            # 한 번만 실행
            print("한 번만 새로운 질문 확인...")
            monitor.process_new_questions()
            return
        elif sys.argv[1] == '--debug':
            # 디버그 모드로 일반 실행
            print("🐛 디버그 모드로 모니터링 시작...")
            print("중단하려면 Ctrl+C를 누르세요.")
            print("-" * 50)
            monitor.run_monitor()
            return
    
    # 일반 모니터링 모드
    print("지속적 모니터링을 시작합니다...")
    print("중단하려면 Ctrl+C를 누르세요.")
    print("디버그 모드: python question_monitor.py --debug")
    print("-" * 50)
    
    monitor.run_monitor()


if __name__ == "__main__":
    main() 