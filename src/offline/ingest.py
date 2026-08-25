import requests, json, sqlite3, time, math
import pandas as pd
from datetime import datetime, timedelta
from typing import List
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('KEEPA_API_KEY')

KEEPA_EPOCH = datetime(2011, 1, 1)


def convert_to_keepa_mins(date: str) -> int:
    """
    Converts a date string to Keepa time (minutes since 2011-01-01).
    date: str '2025-03-01'
    """
    target_date = datetime.strptime(date, '%Y-%m-%d')
    return int((target_date - KEEPA_EPOCH).total_seconds() / 60)


def check_tokens() -> int:
    response = requests.get(f"https://api.keepa.com/token?key={api_key}") # from keepa documents
    tokens = response.json()['tokensLeft']
    print(f"Tokens left: {tokens}")
    return tokens


def get_asin_on_month(starting_date: str, ending_date: str) -> List[str]:
    """
    Returns a list of ASINs launched between starting_date and ending_date.
    date format: '2025-03-01'
    """
    starting_time = convert_to_keepa_mins(starting_date)
    ending_time = convert_to_keepa_mins(ending_date)

    # use product finder function keepa https://keepa.com/#!discuss/t/product-finder/5473
    url_query = f"https://api.keepa.com/query?key={api_key}&domain=1"

    selection = {
        "hasParentASIN": False,
        "variationReviewCount_gte": 0,
        "variationReviewCount_lte": 200,
        "current_BUY_BOX_SHIPPING_gte": 1500,
        "current_BUY_BOX_SHIPPING_lte": 10000,
        "buyBoxSellerId": ["-ATVPDKIKX0DER", "-A3SLTBYT1P4ASM"],
        "totalOfferCount_gte": 0,
        "totalOfferCount_lte": 1,
        "listedSince_gte": starting_time,
        "listedSince_lte": ending_time,
        "rootCategory": [
            "✜10677469011", "✜133140011", "✜13727921011", "✜16333372011",
            "✜163856011", "✜18145289011", "✜18981045011", "✜2238192011",
            "✜2625373011", "✜283155", "✜2858778011", "✜3561432011",
            "✜5174", "✜599858"
        ],
        "sort": [["current_SALES", "asc"], ["monthlySold", "desc"]],
        "productType": [0, 1, 2],
        "perPage": 5000, # default is 50, less than 10,000
        "page": 0 # page index
    }

    # use data instead of json= and converts to a json string so keepa can parse it
    response = requests.post(url_query, data={'selection': json.dumps(selection)}) # json.dumps convert dictionary to string
    return response.json().get('asinList', [])


def get_or_cache_asins(conn, start: str, end: str) -> List[str]:
    """
    Returns ASIN list for a month from db cache, or fetches from Keepa if not cached.
    Saves to asin_list table so Keepa is only called once per month.
    """
    # use a table to save fetched asin list,primary_key is asin, month
    # each month, use asin and month to query the table and get the asin list again
    # if not in the db, we go fetch it
    cached = pd.read_sql(
        "SELECT asin FROM asin_list WHERE month=?", conn, params=(start,)
    )['asin'].tolist()

    if cached:
        print(f"  Using cached ASIN list for {start} ({len(cached)} ASINs)")
        return cached

    print(f"  Fetching ASIN list for {start} from Keepa...")
    results = get_asin_on_month(start, end)
    print(f"  {len(results)} ASINs found")

    # save asin list to db so we never need to fetch it again
    for asin in results:
        conn.execute(
            "INSERT OR IGNORE INTO asin_list(month, asin) VALUES(?,?)",
            (start, asin)
        )
    conn.commit()
    return results


def fetch_and_save_products(conn, start: str, asins: List[str]):
    """
    Fetches product data from Keepa in batches of 100 and saves to products table.
    Stops early if tokens run low.
    Flow is:

    Call Keepa for each month's Asin -> save into db asin_list
    """
    # check which ASINs are already saved in products for this month to handle partial saves
    saved_asins = pd.read_sql(
        "SELECT asin FROM products WHERE month=?", conn, params=(start,)
    )['asin'].tolist()
    remaining = [asin for asin in asins if asin not in saved_asins]
    print(f"  {len(remaining)} ASINs remaining to fetch")

    if not remaining:
        print(f"  Skipping {start} — already complete")
        return

    for i in range(0, len(remaining), 100): # limit is 100 per batch https://keepa.com/#!discuss/t/products/110
        # check tokens before each batch to avoid going negative
        tokens = check_tokens()
        if tokens < 200:  # one batch costs ~190 tokens (100 ASINs * 2)
            wait_min = math.ceil((200 - tokens) / 20)  # 20 tokens/min refill rate
            print(f"Low tokens ({tokens}) — waiting {wait_min} min for refill...")
            conn.commit()
            time.sleep(wait_min * 60) # program wait for * mins to rerun

        asin_list = ','.join(remaining[i:i+100]) # asin is expecting a comma separated string
        url_product = f"https://api.keepa.com/product?key={api_key}&domain=1&asin={asin_list}"
        response_product = requests.get(url_product, params={'history': 1, 'rating': 1})
        resp_json = response_product.json()

        if 'products' not in resp_json:
            print(f"  Error at batch {i}: {resp_json.get('error')}")
            conn.commit()
            return

        for product in resp_json['products']:
            cat = product.get('categoryTree')
            category = cat[0]['name'] if cat else None
            conn.execute(
                "INSERT OR IGNORE INTO products(asin, month, title, cat, raw_data) VALUES(?,?,?,?,?)",
                (product['asin'], start, product.get('title'), category, json.dumps(product)) # convert dict to string so sqlite can store
            )

    conn.commit() # commit after each month so data is not lost if error occurs
    print(f"  Saved {start}")

def fetch_and_save_buybox(conn, start: str, asins: List[str]):
    """
    Fetches product data from Keepa in batches of 100 and saves to products table.
    Stops early if tokens run low.
    Flow is:

    Call Keepa for each month's Asin -> save into db asin_list
    """
    # check which ASINs are already saved in products for this month to handle partial saves
    saved_asins = pd.read_sql(
        "SELECT asin FROM products WHERE month=? AND buybox IS NOT NULL", conn, params=(start,)
    )['asin'].tolist()
    remaining = [asin for asin in asins if asin not in saved_asins]
    print(f"  {len(remaining)} ASINs remaining to fetch buybox")

    if not remaining:
        print(f"  Skipping {start} — already complete")
        return

    for i in range(0, len(remaining), 100): # limit is 100 per batch https://keepa.com/#!discuss/t/products/110
        # check tokens before each batch to avoid going negative
        tokens = check_tokens()
        while check_tokens() < 400:  # one batch costs ~300 tokens (100 ASINs * 3)
            wait_min = math.ceil((400 - tokens) / 20)  # 20 tokens/min refill rate
            print(f"Low tokens ({check_tokens()}) — waiting {wait_min} min for refill...")
            conn.commit()
            time.sleep(wait_min * 60) # program wait for * mins to rerun
            print(f"after wait ({check_tokens()})")

        print(f"after wait ({check_tokens()}), fetching...")
        asin_list = ','.join(remaining[i:i+100]) # asin is expecting a comma separated string
        url_product = f"https://api.keepa.com/product?key={api_key}&domain=1&asin={asin_list}"
        response_product = requests.get(url_product, params={'buybox': 1})
        resp_json = response_product.json()

        if 'products' not in resp_json:
            print(f"  Error at batch {i}: {resp_json.get('error')}")
            conn.commit()
            return

        for product in resp_json['products']:
            conn.execute(
                "UPDATE products SET buybox=? WHERE asin=? and month=?",
                (json.dumps(product),product['asin'], start) # convert dict to string so sqlite can store
            )

    conn.commit() # commit after each month so data is not lost if error occurs
    print(f"  Saved {start}")



def save_data_to_db(start_month: str = '2024-01-01', end_month: str = '2026-04-01', db_path: str = 'data/raw/product_launch.db'):
    """
    Main function. Iterates through months, fetches ASINs and product data, saves to SQLite.
    """
    with sqlite3.connect(db_path) as conn: # with block ensures connection closes even if error occurs
        conn.execute("""
            CREATE TABLE IF NOT EXISTS products(
                asin TEXT, month TEXT, title TEXT, cat TEXT, raw_data TEXT,
                PRIMARY KEY(asin, month)
            )
        """)
        # store month -> asin list so we don't call keepa every time for the same list
        conn.execute("""
            CREATE TABLE IF NOT EXISTS asin_list(
                month TEXT, asin TEXT,
                PRIMARY KEY(month, asin)
            )
        """)

        # update the db with buybox column UPDATE stopped at 2025.10.1
        try:
            conn.execute("ALTER TABLE products ADD COLUMN buybox TEXT ")
        except Exception:
            pass

        months = pd.date_range(start_month, end_month, freq='MS') # create a list of dates [2024-01-01,2024-02-01,2024-03-01,..]
        date_pairs = [(str(months[i].date()), str(months[i+1].date())) for i in range(len(months)-1)] # [('2024-01-01','2024-02-01'),..]

        for start, end in date_pairs:
            print(f"\n--- {start} ---")
            asins = get_or_cache_asins(conn, start, end)
            # fetch_and_save_products(conn, start, asins)
            fetch_and_save_buybox(conn, start, asins)

          # update the db with RATING column 
        try:
            conn.execute("ALTER TABLE products ADD COLUMN rating TEXT ")
        except Exception:
            pass

        months = pd.date_range(start_month, end_month, freq='MS') # create a list of dates [2024-01-01,2024-02-01,2024-03-01,..]
        date_pairs = [(str(months[i].date()), str(months[i+1].date())) for i in range(len(months)-1)] # [('2024-01-01','2024-02-01'),..]

        for start, end in date_pairs:
            print(f"\n--- {start} ---")
            asins = get_or_cache_asins(conn, start, end)
            # fetch_and_save_products(conn, start, asins)
            fetch_and_save_buybox(conn, start, asins)


if __name__ == '__main__':
    save_data_to_db()
