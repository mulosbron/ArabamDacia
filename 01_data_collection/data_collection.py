import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import os
import pandas as pd
import time
import re

load_dotenv()

headers = {
    "User-Agent": os.getenv("USER_AGENT")
}
base_url = os.getenv("BASE_URL")
url_path = os.getenv("URL_PATH")
list_url = f"{base_url}{url_path}"
listings = []

if os.path.exists("arabam_listings.csv"):
    df_existing = pd.read_csv("arabam_listings.csv", dtype={'İlan No': str})
    existing_listing_nos = set(df_existing['İlan No'])
    print(f"{len(existing_listing_nos)} existing listings loaded.")
else:
    df_existing = pd.DataFrame()
    existing_listing_nos = set()
    print("No existing listings found, a new file will be created.")

for page in range(1, 7):
    print(f"Processing page {page}...")
    page_url = f"{list_url}?page={page}"
    response = requests.get(page_url, headers=headers)
    if response.status_code != 200:
        print(f"Page {page} could not be retrieved. Status Code: {response.status_code}")
        continue
    soup = BeautifulSoup(response.content, "html.parser")
    listing_rows = soup.find_all("tr", class_="listing-list-item")
    for row in listing_rows:
        a_tag = row.find("a", href=True)
        if a_tag:
            listing_url = base_url + a_tag['href']
            listings.append(listing_url)
    time.sleep(2)
print(f"Total {len(listings)} listings found.")


def get_listing_details(url):
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"URL could not be retrieved: {url}")
        return None
    soup = BeautifulSoup(response.content, "html.parser")
    details = {}
    details_div = soup.find("div", class_="product-properties-details")
    if not details_div:
        print(f"Details not found: {url}")
        return None
    properties = details_div.find_all("div", class_="property-item")
    for item in properties:
        key_div = item.find("div", class_="property-key")
        value_div = item.find("div", class_="property-value")
        if key_div and value_div:
            key = key_div.get_text(strip=True).replace(":", "")
            value = value_div.get_text(strip=True)
            details[key] = value
    price_container = soup.find("div", class_="product-price-container")
    if price_container:
        price_div = price_container.find("div", {"data-testid": "desktop-information-price"})
        if price_div:
            price = price_div.get_text(strip=True)
            details["Fiyat"] = price
        else:
            print(f"Price info not found in container: {url}")
            details["Fiyat"] = "Unknown"
    else:
        print(f"Price container not found: {url}")
        details["Fiyat"] = "Unknown"
    details["URL"] = url
    listing_no_match = re.search(r'/(\d+)$', url)
    if listing_no_match:
        details['İlan No'] = listing_no_match.group(1)
    else:
        details['İlan No'] = None
    return details


new_listings = []
for idx, listing_url in enumerate(listings, 1):
    print(f"Processing listing {idx}: {listing_url}")
    details = get_listing_details(listing_url)
    if details:
        listing_no = details.get('İlan No')
        if listing_no and listing_no not in existing_listing_nos:
            new_listings.append(details)
            existing_listing_nos.add(listing_no)
            print(f"New listing added: Listing No {listing_no}")
        else:
            print(f"Listing already exists: Listing No {listing_no}")
    time.sleep(2)

if new_listings:
    df_new = pd.DataFrame(new_listings)
    if not df_existing.empty:
        df = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv("arabam_listings.csv", index=False, encoding='utf-8-sig')
    print(f"Total {len(new_listings)} new listings added. Data updated.")
else:
    print("No new listings found. Data is up to date.")
