import asyncio
import os
import platform

from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig
from crawl4ai.content_filter_strategy import BM25ContentFilter
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from crawl4ai.deep_crawling import BestFirstCrawlingStrategy
from crawl4ai.deep_crawling.filters import (
    ContentRelevanceFilter,
    ContentTypeFilter,
    FilterChain,
    URLPatternFilter,
)
from crawl4ai.deep_crawling.scorers import KeywordRelevanceScorer
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By

# ---------------------------------------------------------------------------
# Selenium / Chrome setup
# ---------------------------------------------------------------------------
# The container image used in production includes the Chromium binary at
# /usr/bin/chromium and a matching chromedriver at /usr/bin/chromedriver.  When
# developing locally on macOS (or any environment where Chrome is installed in
# the standard location) we should **not** override the binary path – Selenium
# Manager can discover the correct driver automatically.  The helper below
# prepares a consistent, headless `Options` instance for all platforms and
# chooses the appropriate driver path.


def _make_headless_options() -> Options:
    """Return a headless `selenium.webdriver.ChromeOptions` instance."""

    opts = Options()
    opts.add_argument("--headless")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")

    # Use the Chromium binary shipped in the Linux container when running on
    # Linux servers.  On macOS/Windows we leave `binary_location` unset so that
    # the system installation of Chrome/Chromium is used.
    if platform.system() == "Linux":
        opts.binary_location = "/usr/bin/chromium"

    return opts


def _create_webdriver() -> webdriver.Chrome:
    """Create a cross‑platform headless Chrome driver.

    Priority:
    1. Use the explicit CHROMEDRIVER_PATH env var if provided and valid.
    2. On Linux try the well‑known location used in the container.
    3. Fallback to Selenium Manager auto‑download (requires Selenium ≥4.6).
    """

    opts = _make_headless_options()

    # 1. Explicit override via env
    driver_path = os.getenv("CHROMEDRIVER_PATH")

    # 2. Container default on Linux
    if not driver_path and platform.system() == "Linux":
        driver_path = "/usr/bin/chromedriver"

    # 3. If no path or file missing, rely on Selenium Manager
    if driver_path and os.path.exists(driver_path):
        return webdriver.Chrome(service=Service(driver_path), options=opts)

    return webdriver.Chrome(options=opts)


class AdvancedCrawler:
    """
    A configurable asynchronous crawler for retrieving PDFs from websites.
    """

    def __init__(self, url_patterns=None, scorer_params=None, include_external=False):
        self.filter_chain = None
        self.keyword_scorer = None
        if url_patterns:
            self.filter_chain = FilterChain([URLPatternFilter(patterns=url_patterns)])
        if scorer_params:
            self.keyword_scorer = KeywordRelevanceScorer(**scorer_params)
        self.include_external = include_external

    def _get_pagination_if_exists(self, url):
        driver = _create_webdriver()

        try:
            driver.get(url)
            driver.implicitly_wait(5)

            # Attempt to find pagination elements
            pagination_items = driver.find_elements(
                By.CSS_SELECTOR, ".pagination__item a"
            )
            page_numbers = [
                int(el.text.strip())
                for el in pagination_items
                if el.text.strip().isdigit()
            ]

            if page_numbers:
                max_page = max(page_numbers)
                print(f"Pagination detected, collected {max_page} pages.")
                return [f"{url}&page_no={i}" for i in range(1, max_page + 1)]
            else:
                print("No pagination detected.")
                return []
        finally:
            driver.quit()

    async def _crawl(self, url, max_depth=1):
        config = CrawlerRunConfig(
            deep_crawl_strategy=BestFirstCrawlingStrategy(
                max_depth=max_depth,
                include_external=self.include_external,
                # filter_chain=self.filter_chain,
                # url_scorer=self.keyword_scorer,
            ),
            scraping_strategy=LXMLWebScrapingStrategy(),
            stream=True,
            verbose=False,
            cache_mode=CacheMode.BYPASS,
        )

        results = []
        async with AsyncWebCrawler() as crawler:
            async for result in await crawler.arun(url, config=config):
                if result.success and "application/pdf" in result.response_headers.get(
                    "content-type", ""
                ):
                    filename = result.response_headers.get(
                        "content-disposition", "No title found"
                    )
                    if filename != "No title found":
                        filename = filename.split(";")[1].split("=")[1]
                    print(
                        f"Name: {filename if filename else 'No name found'}\nUrl: {result.url}\n"
                    )
                    results.append(
                        {
                            "url": result.url,
                            "title": filename,
                            "trust": url,
                        }
                    )
        return results

    async def deep_crawl(self, url):
        results = []
        pagination_links = self._get_pagination_if_exists(url)
        if not pagination_links:
            pagination_links = [url]

        for page in pagination_links:
            print(f"Crawling page: {page}")
            papers = await self._crawl(page)
            if len(papers) < 3:
                print("Didn't find sufficient results, trying with depth 2")
                papers = await self._crawl(page, max_depth=2)
            results.extend(papers)

        return results
