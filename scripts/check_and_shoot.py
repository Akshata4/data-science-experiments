"""Load a running Streamlit app in headless Chromium, click through its
CRISP-DM tabs, fail loudly on any Streamlit exception banner, and save
one full-page screenshot per tab into <outdir>.

Usage:
    python3 scripts/check_and_shoot.py <url> <outdir> <tab-prefix> [n_tabs]
"""
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

url, outdir, prefix = sys.argv[1], sys.argv[2], sys.argv[3]
n_tabs = int(sys.argv[4]) if len(sys.argv) > 4 else 6
Path(outdir).mkdir(parents=True, exist_ok=True)

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1440, "height": 1000})
    page.goto(url, wait_until="networkidle", timeout=60000)
    time.sleep(3)

    err = page.locator("text=Traceback (most recent call last)")
    if err.count() > 0:
        print("STREAMLIT EXCEPTION DETECTED")
        print(page.locator("div.stException").first.inner_text() if page.locator("div.stException").count() else page.content()[:3000])
        browser.close()
        sys.exit(1)

    tabs = page.locator('div[data-testid="stTab"]')
    count = tabs.count()
    print(f"found {count} tabs")
    for i in range(min(count, n_tabs)):
        tabs.nth(i).click()
        time.sleep(2.2)
        page.screenshot(path=f"{outdir}/{prefix}_{i:02d}.png", full_page=True)
        print(f"shot {i} ok")

    if count == 0:
        page.screenshot(path=f"{outdir}/{prefix}_00.png", full_page=True)

    browser.close()
print("DONE")
