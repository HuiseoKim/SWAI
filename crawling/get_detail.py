import json

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options

from datetime import datetime
import time
from bs4 import BeautifulSoup
import random
from html import unescape
import everytime_config

from tqdm import tqdm

ID = everytime_config.ID
PW = everytime_config.PW

chrome_options = Options()
#chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-gpu')
#chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3')

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

everytime_url=[]
driver.get("https://yonsei.everytime.kr/442356/v/")
time.sleep(5+ random.uniform(1, 5))
WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.NAME, 'id'))).send_keys(ID)
time.sleep(3+ random.uniform(1, 5))
driver.find_element(By.NAME, 'password').send_keys(PW)
time.sleep(4+ random.uniform(1, 5))
driver.find_element(By.XPATH, '//input[@value="에브리타임 로그인"]').click()
time.sleep(2+ random.uniform(1, 5))

def extract_text(tags):
    full_text = ""
    for tag in tags:
        # 태그에서 텍스트 추출
        text = tag.get_text()

        # HTML 엔티티를 일반 텍스트로 변환 (예: &amp; -> &)
        text = unescape(text)

        # 공백 제거 및 정리
        text = text.strip()

        # 텍스트 추가
        full_text += text + " "  # 단어 사이에 공백 추가

    return full_text.strip()

def extract_comments(soup):
    comments = []
    last_parent_author = None

    for article in soup.find_all('article'):
        comment_type = None  # 여기에 초기값 설정
        author = ""
        comment_text = ""
        timestamp = ""
        vote_count = 0

        h3_tag = article.find('h3', class_='medium')
        p_tag = article.find('p', class_='large')
        time_tag = article.find('time', class_='medium')
        vote_tag = article.find('li', class_='vote')

        if h3_tag:
            author = h3_tag.text.strip()
        if p_tag:
            comment_text = p_tag.text.strip()
        if time_tag:
            timestamp = time_tag.text.strip()
        if vote_tag:
            vote_count = vote_tag.text.strip()

        if 'parent' in article['class']:
            last_parent_author = author
            comment_type = 'parent'
        elif 'child' in article['class']:
            comment_type = 'child'

        if comment_type:
            comment_dict = {
                "Type": comment_type,
                "Author": author,
                "Comment": comment_text,
                "Timestamp": timestamp,
                "Vote Count": vote_count
            }

            # 대댓글일 경우 부모 댓글의 작성자 추가
            if comment_type == 'child' and last_parent_author:
                comment_dict["Parent Author"] = last_parent_author

            comments.append(comment_dict)

    return comments

def process_url(url, existing_data):
    try:
        driver.get("https://yonsei." + url)
        time.sleep(3 + random.uniform(1, 5))
        html_content = driver.page_source
        soup = BeautifulSoup(html_content, 'html.parser')
        title = extract_text(soup.find_all('h2', class_='large'))
        detail = extract_text(soup.find('p', class_='large'))
        likes = extract_text(soup.find('li', class_= "vote"))
        comments_count = extract_text(soup.find('li', class_="comment"))
        scraps = extract_text(soup.find('li', class_='scrap'))
        comments = extract_comments(soup)
        timestamp = extract_text(soup.find('time', class_='large'))
        json_object = {
            "title": title,
            "detail": detail,
            "likes": likes,
            "comments_count": comments_count,
            "scraps": scraps,
            "url": url,
            "comments": comments,
            "timestamp": timestamp
        }
        return json_object
    except Exception as e:
        print(f"Error processing URL {url}: {str(e)}")
        return None

# 기존 데이터 로드
existing_data = []
with open("everytime_computer_data.json", 'r', encoding='UTF-8') as f:
    for line in f:
        existing_data.append(json.loads(line))

# URL 데이터 로드
with open("url.json", 'r', encoding='UTF-8') as f:
    urls = json.load(f)
    
new_data=[]
for url in tqdm(urls, "Processing URLs"):
    existing_url_data = None
    for data in existing_data:
        if data['url'] == url:
            existing_url_data = data
            break
    updated_json_object = process_url(url, existing_data)
    if updated_json_object:
        if existing_url_data:
            existing_url_data.update(updated_json_object)
            new_data.append(updated_json_object)
        else:
            existing_data.append(updated_json_object)
            new_data.append(updated_json_object)

# Write updated data back to the file
with open("everytime_computer_data.json", 'w', encoding='UTF-8') as f:
    for data in existing_data:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')

with open("new_data.json", 'w', encoding='UTF-8') as f:
    for data in new_data:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')

driver.close()