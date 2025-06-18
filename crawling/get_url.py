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

ID = everytime_config.ID
PW = everytime_config.PW

chrome_options = Options()
#chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-gpu')
#chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3')

#driver = webdriver.Chrome(ChromeDriverManager().install(), options=chrome_options)
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

everytime_url=[]
driver.get("https://yonsei.everytime.kr/442356/p/")
time.sleep(3)
WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.NAME, 'id'))).send_keys(ID)
time.sleep(3)
driver.find_element(By.NAME, 'password').send_keys(PW)
driver.find_element(By.XPATH, '//input[@value="에브리타임 로그인"]').click()
time.sleep(3)

for i in range(6,20):
    driver.get('https://yonsei.everytime.kr/442356/p/'+str(i+1))
    time.sleep(3)
    time.sleep(random.uniform(1,5))
    
    html = driver.page_source

    soup = BeautifulSoup(html, 'html.parser')

    articles = soup.find_all('article', class_='list')

    for article in articles:
        a_tag = article.find('a', class_='article')
        if a_tag and 'href' in a_tag.attrs:
            href_value = a_tag['href']
            everytime_url.append('everytime.kr'+href_value)

driver.close()

file_path="url.json"
with open(file_path, 'w') as json_file:
    json.dump(everytime_url, json_file, indent=2)