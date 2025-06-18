#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
에브리타임 컴퓨터학과 데이터를 SFR 모델로 임베딩하여 FAISS 인덱스 생성
./everytime_computer_data.json 파일이 존재할 때 실행됩니다.
"""

import json
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os
import sys
from datetime import datetime

class EverytimeRAGIndexBuilder:
    def __init__(self, model_name="Salesforce/SFR-Embedding-Mistral"):
        """
        에브리타임 데이터를 위한 FAISS index 및 embedding 생성기
        
        Args:
            model_name (str): SFR embedding 모델 이름
        """
        # CUDA 장치 설정 (필요시 수정)
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        
        self.model_name = model_name
        self.model = None
        self.texts = []
        self.metadata = []
        self.embeddings = None
        self.index = None
        
    def load_model(self):
        """SFR embedding 모델 로드"""
        print(f"🚀 SFR 모델 로딩 중: {self.model_name}")
        try:
            self.model = SentenceTransformer(self.model_name)
            print("✅ 모델 로딩 완료!")
        except Exception as e:
            print(f"❌ 모델 로딩 실패: {e}")
            raise
        
    def process_everytime_data(self, json_path):
        """에브리타임 JSON 데이터를 RAG 형식으로 변환 (포스트+댓글 통합)"""
        print(f"📂 에브리타임 데이터 로딩 중: {json_path}")
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = []
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            post = json.loads(line)
                            data.append(post)
                        except json.JSONDecodeError as e:
                            print(f"⚠️  라인 {line_num}에서 JSON 파싱 오류: {e}")
                            continue
        except Exception as e:
            print(f"❌ 파일 읽기 실패: {e}")
            raise
            
        print(f"📊 총 {len(data)}개의 포스트 발견")
        
        # 각 포스트를 댓글과 함께 하나의 텍스트로 변환
        for idx, post in enumerate(data):
            # 포스트와 모든 댓글을 통합한 텍스트 구성
            combined_text = self._create_post_text(post)
            post_id = f"post_{idx}"
            
            # 통합된 포스트 추가
            self.texts.append(combined_text)
            self.metadata.append({
                'id': post_id,
                'metadata': {
                    'type': 'post_with_comments',
                    'post_index': idx,
                    'title': post.get('title', ''),
                    'url': post.get('url', ''),
                    'likes': post.get('likes', '0'),
                    'comments_count': post.get('comments_count', '0'),
                    'scraps': post.get('scraps', '0'),
                    'timestamp': post.get('timestamp', ''),
                    'total_comments': len(post.get('comments', []))
                }
            })
                
        print(f"✅ 데이터 변환 완료! 총 {len(self.texts)}개의 텍스트 생성")
        print(f"   - 포스트(댓글 포함): {len(data)}개")
        
    def _create_post_text(self, post):
        """포스트 데이터와 모든 댓글을 하나의 검색용 텍스트로 변환"""
        title = post.get('title', '')
        detail = post.get('detail', '')
        timestamp = post.get('timestamp', '')
        likes = post.get('likes', '0')
        comments_count = post.get('comments_count', '0')
        
        # 메인 포스트 텍스트 구성
        text = f"제목: {title}\n내용: {detail}\n"
        if timestamp:
            text += f"작성시간: {timestamp}\n"
        text += f"좋아요: {likes}, 댓글수: {comments_count}\n"
        
        # 모든 댓글을 포스트 텍스트에 포함
        comments = post.get('comments', [])
        if comments:
            text += "\n[댓글들]\n"
            for comment_idx, comment in enumerate(comments, 1):
                comment_text = comment.get('Comment', '')
                author = comment.get('Author', '')
                comment_timestamp = comment.get('Timestamp', '')
                vote_count = comment.get('Vote Count', '0')
                comment_type = comment.get('Type', '')
                parent_author = comment.get('Parent Author', '')
                
                # 댓글 정보 추가
                text += f"\n댓글{comment_idx}"
                if comment_type == 'child':
                    text += f"(답글 to {parent_author})"
                text += f": {comment_text}"
                
                if author:
                    text += f" - {author}"
                if comment_timestamp:
                    text += f" ({comment_timestamp})"
                if vote_count and vote_count != '0':
                    text += f" [추천:{vote_count}]"
        
        return text
        
    def create_embeddings(self, batch_size=32):
        """텍스트를 embedding으로 변환"""
        print("🔄 Embedding 생성 중...")
        
        if self.model is None:
            self.load_model()
            
        # 배치 단위로 embedding 생성
        embeddings_list = []
        
        for i in tqdm(range(0, len(self.texts), batch_size), desc="Embedding 진행"):
            batch_texts = self.texts[i:i+batch_size]
            try:
                batch_embeddings = self.model.encode(batch_texts, show_progress_bar=False)
                embeddings_list.extend(batch_embeddings)
            except Exception as e:
                print(f"❌ 배치 {i//batch_size + 1} 처리 중 오류: {e}")
                raise
                
        self.embeddings = np.array(embeddings_list)
        print(f"✅ Embedding 생성 완료! Shape: {self.embeddings.shape}")
        
    def create_faiss_index(self):
        """FAISS index 생성"""
        print("🔍 FAISS index 생성 중...")
        
        if self.embeddings is None:
            raise ValueError("먼저 embedding을 생성해주세요!")
            
        # L2 거리 기반 FAISS index 생성
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        
        # embedding 추가
        self.index.add(self.embeddings.astype('float32'))
        
        print(f"✅ FAISS index 생성 완료! 총 {self.index.ntotal}개 벡터 추가됨")
        
    def save_index_and_data(self, output_dir="./faiss_output"):
        """FAISS index와 관련 데이터 저장"""
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"💾 결과 저장 중: {output_dir}")
        
        # FAISS index 저장
        index_path = os.path.join(output_dir, "faiss_index.bin")
        faiss.write_index(self.index, index_path)
        print(f"  📁 FAISS index 저장됨: {index_path}")
        
        # 텍스트와 메타데이터 저장
        texts_path = os.path.join(output_dir, "texts.pkl")
        with open(texts_path, 'wb') as f:
            pickle.dump(self.texts, f)
        print(f"  📁 텍스트 데이터 저장됨: {texts_path}")
        
        metadata_path = os.path.join(output_dir, "metadata.pkl")
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        print(f"  📁 메타데이터 저장됨: {metadata_path}")
        
        # 임베딩 저장 (옵션)
        embeddings_path = os.path.join(output_dir, "embeddings.npy")
        np.save(embeddings_path, self.embeddings)
        print(f"  📁 Embedding 데이터 저장됨: {embeddings_path}")
        
        # 설정 정보 저장
        config = {
            'model_name': self.model_name,
            'num_texts': len(self.texts),
            'embedding_dimension': self.embeddings.shape[1],
            'created_at': datetime.now().isoformat(),
            'source_file': './everytime_computer_data.json',
            'data_type': 'everytime_computer_science_posts'
        }
        
        config_path = os.path.join(output_dir, "config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        print(f"  📁 설정 정보 저장됨: {config_path}")


def check_input_file():
    """입력 파일 존재 여부 확인"""
    input_file = "./everytime_computer_data.json"
    
    if not os.path.exists(input_file):
        print(f"❌ 입력 파일을 찾을 수 없습니다: {input_file}")
        print("   ./everytime_computer_data.json 파일이 존재하는지 확인해주세요.")
        return False
        
    # 파일 크기 확인
    file_size = os.path.getsize(input_file)
    if file_size == 0:
        print(f"❌ 입력 파일이 비어있습니다: {input_file}")
        return False
        
    print(f"✅ 입력 파일 확인됨: {input_file} ({file_size:,} bytes)")
    return True


def main():
    """메인 실행 함수"""
    print("🎯 에브리타임 컴퓨터학과 데이터 임베딩 생성기")
    print("=" * 60)
    
    # 1. 입력 파일 확인
    if not check_input_file():
        sys.exit(1)
    
    # 경로 설정
    input_file = "./everytime_computer_data.json"
    output_dir = "faiss_output"
    
    # RAG Index Builder 초기화
    builder = EverytimeRAGIndexBuilder()
    
    try:
        # 2. 에브리타임 데이터 로드 및 변환
        builder.process_everytime_data(input_file)
        
        # 3. 모델 로드
        builder.load_model()
        
        # 4. Embedding 생성
        print("\n" + "=" * 60)
        builder.create_embeddings(batch_size=16)  # 메모리 상황에 따라 조절 가능
        
        # 5. FAISS index 생성
        print("\n" + "=" * 60)
        builder.create_faiss_index()
        
        # 6. 결과 저장
        print("\n" + "=" * 60)
        builder.save_index_and_data(output_dir)
        
        print("\n" + "=" * 60)
        print("🎉 RAG Index 생성 완료!")
        print(f"📂 출력 디렉토리: {output_dir}")
        print(f"📊 총 텍스트 수: {len(builder.texts):,}")
        print(f"🔢 Embedding 차원: {builder.embeddings.shape[1]}")
        print(f"💾 인덱스 크기: {builder.index.ntotal:,} 벡터")
        
        # 생성된 파일들 확인
        print(f"\n📁 생성된 파일들:")
        for filename in ['faiss_index.bin', 'texts.pkl', 'metadata.pkl', 'embeddings.npy', 'config.json']:
            filepath = os.path.join(output_dir, filename)
            if os.path.exists(filepath):
                size = os.path.getsize(filepath)
                print(f"   - {filename}: {size:,} bytes")
        
        print("\n✅ 처리 완료! 이제 RAG 시스템에서 검색을 사용할 수 있습니다.")
        
    except KeyboardInterrupt:
        print("\n⚠️  사용자에 의해 중단되었습니다.")
        return False
    except Exception as e:
        print(f"\n❌ 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
        
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 