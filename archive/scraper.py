from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time
import random
import re
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).parents[1] / 'data'

df = pd.read_csv(DATA_DIR / 'asins_allbutclothing_filtered_less_than_90.csv')
already_done_df=pd.read_csv(DATA_DIR / 'scripted_review_counts.csv')

asins=df['asin'].tolist()
already_done=set(already_done_df['asin'].tolist())

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
    "source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
})

def get_review_count(soup):
    # case 1: no reviews (format 1)
    if soup.find('div', {'id': 'cm-cr-dp-no-reviews-message'}):
        return 0
    
    # case 2: no reviews (format 2)
    title = soup.find('span', {'data-hook': 'top-customer-reviews-title'})
    if title and 'no customer reviews' in title.get_text(strip=True).lower():
        return 0
    
    # case 3: has reviews
    span = soup.find('span', {'data-hook': 'total-review-count'})
    if span:
        return int(re.sub(r'[^\d]', '', span.get_text(strip=True)))
    
    return None  # blocked or unexpected structure

results = {}
output_path = DATA_DIR / 'scripted_review_counts.csv'
output_path.parent.mkdir(parents=True, exist_ok=True)

try:
    for i, asin in enumerate(asins):
        if asin in already_done:
            print('{asin} already collected, skip')
            continue
        else:
            driver.get(f"https://www.amazon.com/dp/{asin}")
            time.sleep(random.uniform(2, 5))
            html = driver.page_source
            if 'robot' in driver.current_url or 'captcha' in html.lower():
                print(f"Blocked at {asin}, stopping")
                break
            soup = BeautifulSoup(html, 'html.parser')
            results[asin] = get_review_count(soup)
            print(f"{asin}: {results[asin]}")

        if (i + 1) % 100 == 0:
            pd.DataFrame(list(results.items()), columns=['asin', 'review_count']) \
              .to_csv(output_path, mode='a', header=not output_path.exists(), index=False)
            print(f"Checkpoint saved at {i + 1} ASINs")
            results = {}

    if results:
        pd.DataFrame(list(results.items()), columns=['asin', 'review_count']) \
          .to_csv(output_path, mode='a', header=not output_path.exists(), index=False)
finally:
    driver.quit()



