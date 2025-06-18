#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
새로운 RAG 시스템
SFR + FAISS + Llama 3.2 8b를 사용한 질문 답변 시스템
"""

import os
import json
import pickle
import numpy as np
import torch
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import List, Dict, Optional
import logging
import re


class NewRAGSystem:
    def __init__(self, faiss_dir: str = "./crawling/faiss_output", device_id: int = 0):
        """
        새로운 RAG 시스템 초기화
        
        Args:
            faiss_dir: FAISS 인덱스 디렉토리
            device_id: 사용할 GPU ID (0번 GPU 고정)
        """
        self.faiss_dir = faiss_dir
        self.device = f"cuda:{device_id}"
        
        # 모델들
        self.sfr_model = None
        self.llama_tokenizer = None
        self.llama_model = None
        
        # FAISS 관련
        self.faiss_index = None
        self.texts = []
        self.metadata = []
        self.config = {}
        
        # 로깅 설정
        self.logger = logging.getLogger(__name__)
        
    def load_embedding_model(self):
        """SFR 임베딩 모델 로드"""
        self.logger.info("SFR 임베딩 모델 로딩 중...")
        
        # SFR 모델 (Salesforce/SFR-Embedding-Mistral)
        model_name = "Salesforce/SFR-Embedding-Mistral"
        self.sfr_model = SentenceTransformer(model_name, device=self.device)
        
        self.logger.info(f"✅ SFR 모델 로드 완료: {model_name}")
    
    def load_llama_model(self):
        """Llama 3.2 모델 로드 (더 가벼운 버전 사용)"""
        self.logger.info("Llama 3.2 모델 로딩 중...")
        
        # 더 가벼운 모델 사용 (1B 또는 3B)
        model_name = "meta-llama/Meta-Llama-3-8B-Instruct"  # 8B 대신 1B 사용
        
        # 양자화 설정 (메모리 효율성을 위해)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        # 토크나이저 로드
        self.llama_tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        if self.llama_tokenizer.pad_token is None:
            self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        
        # 모델 로드 (0번 GPU에 고정)
        self.llama_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map={"": self.device},  # 0번 GPU에 고정
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        
        self.logger.info(f"✅ Llama 3.2 1B 모델 로드 완료 (GPU: {self.device})")
    
    def load_faiss_index(self):
        """FAISS 인덱스와 관련 데이터 로드"""
        self.logger.info("FAISS 인덱스 로딩 중...")
        
        # 설정 파일 로드
        config_path = os.path.join(self.faiss_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        
        # FAISS 인덱스 로드
        index_path = os.path.join(self.faiss_dir, "faiss_index.bin")
        self.faiss_index = faiss.read_index(index_path)
        
        # 텍스트 데이터 로드
        texts_path = os.path.join(self.faiss_dir, "texts.pkl")
        with open(texts_path, 'rb') as f:
            self.texts = pickle.load(f)
        
        # 메타데이터 로드
        metadata_path = os.path.join(self.faiss_dir, "metadata.pkl")
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        
        self.logger.info(f"✅ FAISS 인덱스 로드 완료: {self.faiss_index.ntotal}개 벡터, {len(self.texts)}개 텍스트")
    
    def initialize_all(self):
        """모든 컴포넌트 초기화"""
        self.logger.info("RAG 시스템 초기화 시작...")
        
        try:
            self.load_embedding_model()
            self.load_faiss_index() 
            self.load_llama_model()
            
            self.logger.info("✅ RAG 시스템 초기화 완료!")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ RAG 시스템 초기화 실패: {e}")
            return False
    
    def search_similar_documents(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        쿼리에 대한 유사 문서 검색
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 상위 결과 수
            
        Returns:
            검색된 문서 리스트
        """
        if self.sfr_model is None or self.faiss_index is None:
            raise ValueError("RAG 시스템이 초기화되지 않았습니다!")
        
        # 쿼리를 SFR로 임베딩
        query_embedding = self.sfr_model.encode([query])
        
        # FAISS 검색
        distances, indices = self.faiss_index.search(
            query_embedding.astype('float32'), top_k
        )
        
        # 검색 결과 구성
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.texts):
                result = {
                    'rank': i + 1,
                    'text': self.texts[idx],
                    'metadata': self.metadata[idx] if idx < len(self.metadata) else {},
                    'distance': float(distance),
                    'similarity_score': 1 / (1 + distance)
                }
                results.append(result)
        
        return results
    
    def generate_answer_with_llama(self, question: str, context_docs: List[Dict]) -> str:
        """
        검색된 문서를 바탕으로 Llama 3.2 8b로 답변 생성
        
        Args:
            question: 사용자 질문
            context_docs: 검색된 관련 문서들
            
        Returns:
            생성된 답변
        """
        if self.llama_model is None or self.llama_tokenizer is None:
            raise ValueError("Llama 모델이 로드되지 않았습니다!")
        
        # 컨텍스트 구성
        context_text = ""
        for i, doc in enumerate(context_docs[:3], 1):  # 상위 3개만 사용
            text = doc['text'][:300]  # 텍스트 길이 제한
            context_text += f"[참고자료 {i}]\n{text}\n\n"
        
        # 개선된 프롬프트 구성
        prompt = f"""당신은 대학생들을 돕는 친근한 AI 상담사입니다. 아래 참고 정보를 바탕으로 학생의 질문에 자연스럽고 도움이 되는 답변을 한국어로 제공해주세요.

중요한 규칙:
0. 질문에 대한 답변을 제공하세요.
1. 반드시 한국어로만 답변하세요
2. 절대로 코드, 프로그래밍 언어, 함수, 변수명 등을 포함하지 마세요
3. 일반적인 대화체로 자연스럽게 답변하세요
4. 참고 정보의 내용을 참고해서 질문에 대한 답변을 제공하세요
5. 친근하고 이해하기 쉽게 설명하세요
6. 답변은 완전한 문장으로 구성하세요

참고 정보:
***
{context_text}
***

질문: {question}

답변:"""
        
        # 토크나이저로 변환
        inputs = self.llama_tokenizer(
            prompt, 
            return_tensors="pt", 
            max_length=2048,
            truncation=True
        ).to(self.device)
        
        # 답변 생성
        with torch.no_grad():
            outputs = self.llama_model.generate(
                **inputs,
                max_new_tokens=300,  # 더 짧게 조정
                temperature=0.3,  # 더 보수적으로 변경
                do_sample=True,
                top_p=0.8,  # 더 보수적으로
                top_k=40,
                repetition_penalty=1.2,  # 더 엄격하게
                pad_token_id=self.llama_tokenizer.eos_token_id,
                eos_token_id=self.llama_tokenizer.eos_token_id
            )
        
        # 응답 디코딩
        generated_text = self.llama_tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        ).strip()
        
        # 답변 후처리
        generated_text = self._post_process_answer(generated_text)
        
        return generated_text
    
    def _post_process_answer(self, text: str) -> str:
        """
        생성된 답변을 후처리하여 품질을 개선
        
        Args:
            text: 원본 텍스트
            
        Returns:
            후처리된 텍스트
        """
        if not text or len(text.strip()) < 3:
            return "관련 정보를 찾을 수 없습니다."
        
        # 기본적인 정리
        text = text.strip()
        
        # 코드 블록 제거 (```로 둘러싸인 부분)
        text = re.sub(r'```[\s\S]*?```', '', text)
        text = re.sub(r'`[^`]*`', '', text)  # 인라인 코드 제거
        
        # 프로그래밍 관련 키워드가 포함된 줄 제거
        programming_keywords = [
            'import', 'def ', 'class ', 'return', 'if __name__',
            'python', 'function', 'variable', '()', '{','}', 
            'def(', 'return(', 'import ', 'from ', 'print(',
            '= [', '= {', '= (', 'lambda', 'yield'
        ]
        
        lines = text.split('\n')
        clean_lines = []
        
        for line in lines:
            line = line.strip()
            # 빈 줄이나 너무 짧은 줄 건너뛰기
            if not line or len(line) < 3:
                continue
                
            # 프로그래밍 키워드가 포함된 줄 제거
            contains_code = any(keyword.lower() in line.lower() for keyword in programming_keywords)
            
            # 영어로만 구성된 줄 제거 (한국어가 없는 경우)
            has_korean = any('\uac00' <= char <= '\ud7af' for char in line)
            
            if not contains_code and (has_korean or len([c for c in line if c.isalpha()]) < len(line) * 0.5):
                clean_lines.append(line)
        
        # 정리된 텍스트 재구성
        text = ' '.join(clean_lines)
        
        # 특수 문자나 이상한 패턴 제거
        text = re.sub(r'[(){}\[\]<>]', '', text)  # 괄호 제거
        text = re.sub(r'\s+', ' ', text)  # 여러 공백을 하나로
        text = text.strip()
        
        # 너무 짧거나 의미 없는 답변 처리
        if len(text) < 10:
            return "죄송합니다. 명확한 답변을 드리기 어렵습니다. 좀 더 구체적으로 질문해 주시겠어요?"
        
        # 길이 제한 (400자 이내)
        if len(text) > 400:
            # 마지막 완전한 문장까지만 자르기
            sentences = text.split('.')
            result = ""
            for sentence in sentences:
                if len(result + sentence + '.') <= 400:
                    result += sentence + '.'
                else:
                    break
            text = result if result else text[:400] + "..."
        
        # 마지막 정리
        if not text.endswith(('.', '!', '?', '요', '다', '니다', '습니다')):
            if '.' not in text[-10:]:
                text += "."
        
        return text
    
    def generate_rag_answer(self, question: str) -> str:
        """
        RAG 파이프라인: 검색 + 생성
        
        Args:
            question: 사용자 질문
            
        Returns:
            RAG로 생성된 답변
        """
        try:
            # 1. FAISS 검색
            search_results = self.search_similar_documents(question, top_k=3)
            
            if not search_results:
                return "죄송합니다. 관련 정보를 찾을 수 없습니다."
            
            # 2. Llama로 답변 생성
            answer = self.generate_answer_with_llama(question, search_results)
            
            # 3. 로깅
            self.logger.info(f"RAG 답변 생성 완료 - 질문: {question[:50]}...")
            for i, result in enumerate(search_results, 1):
                self.logger.debug(f"  참고자료 {i}: 유사도 {result['similarity_score']:.3f}")
            
            return answer
            
        except Exception as e:
            self.logger.error(f"RAG 답변 생성 실패: {e}")
            return "죄송합니다. 답변 생성 중 오류가 발생했습니다."

    def generate_rag_answer_with_documents(self, question: str) -> Dict[str, any]:
        """
        RAG 파이프라인: 검색 + 생성 + 문서 링크 반환
        
        Args:
            question: 사용자 질문
            
        Returns:
            Dict: {
                'answer': 생성된 답변,
                'documents': 참고 문서 링크들
            }
        """
        try:
            # 1. FAISS 검색
            search_results = self.search_similar_documents(question, top_k=3)
            
            if not search_results:
                return {
                    'answer': "죄송합니다. 관련 정보를 찾을 수 없습니다.",
                    'documents': []
                }
            
            # 2. Llama로 답변 생성
            answer = self.generate_answer_with_llama(question, search_results)
            
            # 3. 문서 링크들 추출
            document_links = []
            for result in search_results:
                metadata = result.get('metadata', {})
                if isinstance(metadata, dict) and 'metadata' in metadata:
                    # 메타데이터가 중첩된 구조인 경우
                    inner_metadata = metadata['metadata']
                    url = inner_metadata.get('url', '')
                    title = inner_metadata.get('title', '제목 없음')
                else:
                    # 직접 구조인 경우
                    url = metadata.get('url', '')
                    title = metadata.get('title', '제목 없음')
                
                if url:
                    # 완전한 URL 생성
                    if not url.startswith('http'):
                        url = f"https://{url}"
                    
                    document_links.append({
                        'title': title,
                        'url': url,
                        'similarity_score': result.get('similarity_score', 0)
                    })
            
            # 4. 로깅
            self.logger.info(f"RAG 답변 + 문서 생성 완료 - 질문: {question[:50]}...")
            self.logger.info(f"참고 문서 {len(document_links)}개: {[doc['title'][:20] for doc in document_links]}")
            
            return {
                'answer': answer,
                'documents': document_links
            }
            
        except Exception as e:
            self.logger.error(f"RAG 답변+문서 생성 실패: {e}")
            return {
                'answer': "죄송합니다. 답변 생성 중 오류가 발생했습니다.",
                'documents': []
            }


def test_rag_system():
    """RAG 시스템 테스트"""
    print("🧪 새로운 RAG 시스템 테스트")
    print("=" * 50)
    
    # RAG 시스템 초기화
    rag = NewRAGSystem(faiss_dir="./crawling/faiss_output", device_id=0)
    
    if not rag.initialize_all():
        print("❌ RAG 시스템 초기화 실패")
        return
    
    # 테스트 질문들
    test_questions = [
        "컴퓨터과학과 과제가 너무 어려워요",
        "이산구조 수업은 어떤가요?",
        "복수전공 신청 방법을 알려주세요"
    ]
    
    print("\n🔍 RAG 답변 테스트")
    print("-" * 30)
    
    for question in test_questions:
        print(f"\n질문: {question}")
        answer = rag.generate_rag_answer(question)
        print(f"답변: {answer}")
        print("-" * 30)


if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    test_rag_system() 