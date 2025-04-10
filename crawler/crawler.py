import numpy as np
import polars as pl
import asyncio
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from crawl4ai.deep_crawling.filters import (
    FilterChain,
    URLPatternFilter,
    ContentTypeFilter,
    ContentRelevanceFilter
)
from crawl4ai.deep_crawling.scorers import KeywordRelevanceScorer
from crawl4ai.deep_crawling import BestFirstCrawlingStrategy
from crawl4ai.content_filter_strategy import BM25ContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager


class AdvancedCrawler:
    """
    A configurable asynchronous crawler for retrieving PDFs from websites.
    
    Parameters:
        url_patterns (list): List of URL patterns to include. Defaults to patterns likely related to board meeting papers.
        max_depth (int): The maximum crawling depth. Defaults to 6.
        include_external (bool): Whether to include external links. Defaults to False.
        stream (bool): Whether to stream results from the crawler. Defaults to True.
        verbose (bool): If True, enables verbose logging. Defaults to False.
    """
    def __init__(self, url_patterns=None, scorer_params=None, include_external=False):
        self.filter_chain = None
        self.keyword_scorer = None
        if url_patterns:
            self.filter_chain = FilterChain([
                URLPatternFilter(patterns=url_patterns)
            ])
        if scorer_params:
            self.keyword_scorer = KeywordRelevanceScorer(**scorer_params)
        self.include_external = include_external


    def _get_pagination_if_exists(self, url):
        # Setup Selenium WebDriver
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")  # Run in headless mode (no GUI)
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        
        # Open the URL
        base_url = url
        driver.get(base_url)
        
        # Wait for the page to fully load
        driver.implicitly_wait(5)
        
        # Locate pagination elements
        pagination_items = driver.find_elements(By.CSS_SELECTOR, ".pagination__item a")
        
        # Extract numeric page numbers from available links
        page_numbers = [int(el.text.strip()) for el in pagination_items if el.text.strip().isdigit()]
        driver.quit()
        
        # Get the highest page number
        if page_numbers:
            max_page = max(page_numbers)
            print(f"Pagination detected, collected {max_page} pages.")
            
            all_page_links = [f"{base_url}&page_no={i}" for i in range(1, max_page + 1)]
            return all_page_links
            
        else:
            print("No pagination detected.")
            return []
    

    async def _crawl(self, url, max_depth=1):
        """
        Crawls the given URL and returns results for PDFs whose content type is "application/pdf".

        Args:
            url (str): The starting URL to begin crawling.
            
        Returns:
            list: A list of dictionaries containing the trust URL and the PDF URL.
        """
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
            cache_mode=CacheMode.BYPASS
        )
        results = []
        async with AsyncWebCrawler() as crawler:
            async for result in await crawler.arun(url, config=config):
                if result.success and ("application/pdf" in result.response_headers["content-type"]):
                    filename = result.response_headers.get("content-disposition", "No title found")
                    if filename != "No title found":
                        filename = filename.split(";")[1].split("=")[1]
                    print(f"Name: {filename if filename else 'No name found'}\nUrl: {result.url}\n")
                    results.append({
                        "url": result.url,
                        "title": filename,
                        "trust": url,
                    })
        return results
    

    async def deep_crawl(self, url):
        results = []
        pagination_links = self._get_pagination_if_exists(url)
        if not pagination_links:
            pagination_links = [url]
        for page in pagination_links:
            print(f"Crawling page: {page}")
            papers = await(self._crawl(page))
            if len(papers) < 3:
                print("Didn't find sufficient results, trying with depth 2")
                papers = await(self._crawl(page, max_depth=2))
            results.extend(papers)
        return results