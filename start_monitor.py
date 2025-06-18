#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
질문 모니터링 시스템 런처

사용법:
- python start_monitor.py           # 일반 모니터링 시작
- python start_monitor.py --test    # 시스템 테스트
- python start_monitor.py --once    # 한 번만 실행
- python start_monitor.py --daemon  # 백그라운드 실행
"""

import sys
import os
import subprocess
from datetime import datetime

def print_banner():
    """시작 배너 출력"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                 🤖 질문 모니터링 시스템                       ║
    ║                                                              ║
    ║  Google Sheets에서 새로운 질문을 모니터링하고                ║
    ║  RAG 시스템으로 자동 답변을 생성합니다.                      ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_requirements():
    """필수 요구사항 확인"""
    print("🔍 시스템 요구사항 확인 중...")
    
    # Python 패키지 확인
    required_packages = ['requests', 'pandas']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ 누락된 패키지: {', '.join(missing_packages)}")
        print("다음 명령으로 설치하세요:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    # RAG 시스템 확인
    rag_path = "./crawling/faiss_output"
    if not os.path.exists(rag_path):
        print("⚠️  RAG 시스템이 설정되지 않았습니다.")
        print("기본 답변 모드로 실행됩니다.")
        print("RAG 설정은 다음 명령으로 하세요:")
        print("cd RAG && python create_embeddings.py")
    else:
        print("✅ RAG 시스템 감지됨")
    
    print("✅ 기본 요구사항 확인 완료")
    return True

def show_help():
    """도움말 출력"""
    help_text = """
🚀 질문 모니터링 시스템 사용법

기본 명령:
  python start_monitor.py           # 일반 모니터링 시작
  python start_monitor.py --test    # 시스템 테스트
  python start_monitor.py --once    # 한 번만 실행
  python start_monitor.py --debug   # 디버그 모드 실행
  python start_monitor.py --daemon  # 백그라운드 실행
  python start_monitor.py --help    # 이 도움말 보기

상세 설명:
  --test    : 시스템 연결 및 기능 테스트
  --once    : 새로운 질문을 한 번만 확인하고 종료
  --debug   : 상세한 디버그 로그와 함께 실행
  --daemon  : 백그라운드에서 지속적으로 실행
  --help    : 이 도움말 출력

설정 파일:
  monitor_config.py : 시스템 설정 (체크 주기, URL 등)
  
로그 파일:
  question_monitor.log : 실행 로그 확인
  answer_backup.jsonl : 저장 실패 시 백업 파일

문제 해결:
  1. 먼저 --test 옵션으로 시스템 상태 확인
  2. --debug 옵션으로 상세한 로그 확인
  3. answer_backup.jsonl 파일에서 저장 실패한 답변 확인

중단 방법:
  Ctrl+C : 포어그라운드 실행 중단
  
백그라운드 프로세스 확인:
  ps aux | grep start_monitor
    """
    print(help_text)

def run_daemon():
    """백그라운드에서 실행"""
    print("🔄 백그라운드 모드로 시작...")
    
    # 백그라운드 실행을 위한 nohup 명령 구성
    cmd = [
        'nohup',
        sys.executable,
        'question_monitor.py',
        '&'
    ]
    
    try:
        # 백그라운드 프로세스 시작
        process = subprocess.Popen(
            [sys.executable, 'question_monitor.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE
        )
        
        print(f"✅ 백그라운드 프로세스 시작됨 (PID: {process.pid})")
        print("📄 로그 확인: tail -f question_monitor.log")
        print("⏹️  중단하려면: kill " + str(process.pid))
        
        # PID 파일 저장
        with open('monitor.pid', 'w') as f:
            f.write(str(process.pid))
        
        return True
        
    except Exception as e:
        print(f"❌ 백그라운드 실행 실패: {e}")
        return False

def main():
    """메인 함수"""
    print_banner()
    
    # 명령행 인자 처리
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        
        if arg in ['--help', '-h']:
            show_help()
            return
        
        elif arg == '--daemon':
            if not check_requirements():
                return
            run_daemon()
            return
    
    # 기본 요구사항 확인
    if not check_requirements():
        print("\n❌ 요구사항을 먼저 설치해주세요.")
        return
    
    print(f"\n🚀 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    try:
        # question_monitor.py 실행
        if len(sys.argv) > 1:
            # 인자를 그대로 전달
            cmd = [sys.executable, 'question_monitor.py'] + sys.argv[1:]
        else:
            cmd = [sys.executable, 'question_monitor.py']
        
        subprocess.run(cmd, check=True)
        
    except KeyboardInterrupt:
        print("\n🛑 사용자에 의해 중단됨")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 실행 오류: {e}")
    except Exception as e:
        print(f"\n💥 예상치 못한 오류: {e}")
    finally:
        print(f"\n⏰ 종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("감사합니다! 👋")

if __name__ == "__main__":
    main() 