import os
import re
import json
import time
import asyncio
from typing import List, Dict, Any
from datetime import datetime
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, LLMConfig
from crawl4ai.extraction_strategy import LLMExtractionStrategy
import urllib.parse
from urllib.parse import urljoin, urlparse
import warnings

from utils.config import Config

class NHSCrawler:
    def __init__(self):
        """Initialize the NHS crawler and set up patterns for board paper recognition."""
        # Flask app is passed to avoid circular imports
        self.use_simulation = False

        # Check if OpenAI API key is valid, if not, use simulation mode
        openai_api_key = os.getenv('OPENAI_API_KEY', '')
        if not openai_api_key or openai_api_key == 'sk-your-openai-api-key' or Config.SIMULATE_CRAWLER:
            self.use_simulation = True
            print("Using simulation mode for board paper detection.")

        # Initialize organization info dictionary
        self.organization_info = {}
        
        # Directory to save the list of NHS organizations
        self.data_dir = Config.DATA_DIR
        os.makedirs(self.data_dir, exist_ok=True)
        self.orgs_file = os.path.join(self.data_dir, "nhs_organizations.json")
        
        # NHS URL sources
        self.nhs_trust_list_url = "https://www.england.nhs.uk/about/nhs-health-organisations/nhs-trusts/"
        self.icb_list_url = "https://www.england.nhs.uk/integratedcare/integrated-care-in-your-area/"

        # Common patterns to identify board papers
        self.board_paper_patterns = [
            r'board[-_\s]?papers?',
            r'board[-_\s]?meeting',
            r'board[-_\s]?pack',
            r'governing[-_\s]?body',
            r'minutes',
            r'agenda',
            r'papers[-_\s]?for[-_\s]?board',
            r'trust[-_\s]?board',
            r'public[-_\s]?board',
            r'committee[-_\s]?papers',
            r'annual[-_\s]?report',
            r'icb[-_\s]?board',
            r'board[-_\s]?of[-_\s]?directors',
            # Additional patterns for more variations
            r'meeting[-_\s]?of[-_\s]?the[-_\s]?board',
            r'board[-_\s]?of[-_\s]?directors[-_\s]?meeting',
            r'trust[-_\s]?board[-_\s]?meeting',
            r'board[-_\s]?meeting[-_\s]?papers',
            r'public[-_\s]?board[-_\s]?meeting',
            r'directors[-_\s]?meeting',
            r'governing[-_\s]?body[-_\s]?meeting',
            r'board[-_\s]?report'
        ]
        
        # Common date patterns in board paper titles
        self.date_patterns = [
            r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[-_\s]?20\d\d',
            r'20\d\d[-_\s]?(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)',
            r'(january|february|march|april|may|june|july|august|september|october|november|december)[-_\s]?20\d\d',
            r'20\d\d[-_\s]?(january|february|march|april|may|june|july|august|september|october|november|december)',
            r'\d{1,2}[-_\s]?(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[-_\s]?20\d\d',
            r'\d{1,2}[-_\s]?(january|february|march|april|may|june|july|august|september|october|november|december)[-_\s]?20\d\d'
        ]

        # Keep track of already visited URLs to prevent loops
        self.visited_urls = set()
        
    def _get_extraction_schema(self) -> dict:
        """Return the JSON schema for LLM extraction of board papers"""
        return {
            "type": "object",
            "properties": {
                "is_board_papers_page": {
                    "type": "boolean",
                    "description": "Whether this page lists board papers or meeting minutes"
                },
                "board_papers": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {
                                "type": "string",
                                "description": "Title of the board paper or meeting document"
                            },
                            "url": {
                                "type": "string",
                                "description": "URL of the document"
                            },
                            "date": {
                                "type": "string",
                                "description": "Date of the meeting or document (if available)"
                            }
                        },
                        "required": ["title", "url"]
                    }
                },
                "potential_navigation_links": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "The text of the navigation link"
                            },
                            "url": {
                                "type": "string",
                                "description": "URL of the navigation link"
                            }
                        },
                        "required": ["text", "url"]
                    }
                }
            },
            "required": ["is_board_papers_page"]
        }
        
    def _get_extraction_instructions(self) -> str:
        """Return the instruction prompt for the LLM extraction"""
        return """
        You are tasked with finding board papers on NHS Trust and Integrated Care Board (ICB) websites. 
        
        Board papers are typically found in sections labeled as:
        - "Board meetings"
        - "Governing Body meetings"
        - "Trust Board"
        - "Board papers"
        - "Papers" 
        - "Governance"
        - "About us" -> "Board" or "Our board"
        - "Publications"
        - "Freedom of Information"
        - "Joint Health & Wellbeing Board"
        - "Integrated Care Partnership" or "ICP"
        - "Integrated Care Board" or "ICB"
        
        Board papers are usually PDF documents that contain:
        - Meeting minutes
        - Meeting agendas
        - Financial reports
        - Performance reports
        - Strategy documents
        - Executive reports
        
        Analyze this webpage and provide the following:
        1. Is this page listing board papers or meeting minutes? Respond with true or false.
        2. If true, extract all links to board papers or meeting documents on this page.
           For each document, provide the title, URL, and date (if available).
        3. If false but you see promising navigation links that might lead to board papers,
           list those links in detail.
        
        Be thorough and check for any links that might lead to or contain board papers.
        """
    
    def _get_nhs_organizations(self) -> List[Dict[str, str]]:
        """Get a list of NHS Trusts and ICBs with their website URLs"""
        # Check if we already have the data cached
        if os.path.exists(self.orgs_file):
            with open(self.orgs_file, 'r') as f:
                return json.load(f)
        
        organizations = []
        
        # Scrape NHS Trusts
        print("Scraping NHS Trusts information...")
        try:
            response = requests.get(self.nhs_trust_list_url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # NHS Trusts are typically listed in sections with h2 headers for categories
            # and ul/li elements for the actual trusts
            for section in soup.find_all(['div', 'section']):
                for li in section.find_all('li'):
                    link = li.find('a')
                    if link and link.text:
                        organizations.append({
                            "name": link.text.strip(),
                            "type": "Trust",
                            "url": link['href'] if link['href'].startswith('http') else None
                        })
        except Exception as e:
            print(f"Error scraping NHS Trusts: {e}")
        
        # Scrape ICBs
        print("Scraping ICBs information...")
        try:
            response = requests.get(self.icb_list_url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # ICBs are typically listed in a similar format
            for section in soup.find_all(['div', 'section']):
                for li in section.find_all('li'):
                    link = li.find('a')
                    if link and link.text and 'ICB' in link.text:
                        organizations.append({
                            "name": link.text.strip(),
                            "type": "ICB",
                            "url": link['href'] if link['href'].startswith('http') else None
                        })
        except Exception as e:
            print(f"Error scraping ICBs: {e}")
        
        # Save the data for future use
        with open(self.orgs_file, 'w') as f:
            json.dump(organizations, f, indent=2)
            
        return organizations

    def should_follow_link(self, link_text: str, link_url: str = "") -> bool:
        """Determine if we should follow a navigation link based on its text and URL"""
        # Skip links that are likely not useful
        if not link_text and not link_url:
            return False
            
        # Skip social media and external links
        skip_domains = ['twitter.com', 'facebook.com', 'youtube.com', 'linkedin.com', 'instagram.com']
        if any(domain in str(link_url).lower() for domain in skip_domains):
            return False
            
        # Always follow links that are likely to lead to board papers
        high_priority_keywords = [
            "board", "papers", "meeting", "governance", "minutes", "agenda", 
            "trust board", "board of directors", "publications"
        ]
        
        # Convert to lowercase for case-insensitive matching
        link_text_lower = link_text.lower() if link_text else ""
        link_url_lower = link_url.lower() if link_url else ""
        
        # Check for pagination links
        if self.is_pagination_link(link_text_lower, link_url_lower):
            return True
        
        # Check for high priority keywords
        for keyword in high_priority_keywords:
            if keyword in link_text_lower or keyword in link_url_lower:
                return True
        
        # Medium priority keywords
        medium_priority_keywords = [
            "committee", "publication", "foi", "freedom of information",
            "annual report", "corporate", "council", "icb", "icp", "strategy",
            "about us", "about", "who we are", "what we do"
        ]
        
        for keyword in medium_priority_keywords:
            if keyword in link_text_lower or keyword in link_url_lower:
                return True
                
        return False
        
    def is_pagination_link(self, link_text: str, link_url: str) -> bool:
        """Detect if a link is for pagination"""
        # Check for common pagination patterns in text
        pagination_texts = ['next', 'previous', 'page', '»', '«', '>', '<']
        if any(p in link_text for p in pagination_texts):
            return True
            
        # Check for page numbers in text
        if link_text.isdigit():
            return True
            
        # Check URL for pagination patterns
        pagination_patterns = [
            r'page=\d+', r'p=\d+', r'/page/\d+', r'/p/\d+', 
            r'offset=\d+', r'start=\d+', r'limit=\d+'
        ]
        
        for pattern in pagination_patterns:
            if re.search(pattern, link_url):
                return True
                
        return False

    def extract_date_from_title(self, title: str) -> str:
        """Try to extract a date from a board paper title"""
        # Try different date patterns
        for pattern in self.date_patterns:
            match = re.search(pattern, title.lower())
            if match:
                return match.group(0)
        return ""  # Return empty string if no date found

    def is_likely_board_paper(self, url: str, title: str) -> bool:
        """Determine if a URL is likely a board paper based on the URL and title"""
        # Convert to lowercase for case-insensitive matching
        url_lower = url.lower()
        title_lower = title.lower()
        
        # Check if it's a document
        document_extensions = ['.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx']
        is_document = any(url_lower.endswith(ext) for ext in document_extensions)
        
        # Also check for URLs that contain these extensions but might have additional parameters
        if not is_document:
            is_document = any(f"{ext}?" in url_lower or f"{ext}&" in url_lower for ext in document_extensions)
            
        # Also check for links that say "papers" or "agenda" even if they don't have a document extension
        # (they might link to a page with the actual documents)
        if not is_document and ("papers" in url_lower or "agenda" in url_lower or "minutes" in url_lower):
            return True
        
        # Key terms that indicate board papers
        key_terms = [
            "board", "director", "meeting", "trust", "governing body", "committee",
            "minutes", "agenda", "papers", "pack", "report", "icb", "integrated care board",
            "public board", "trust board", "board of directors", "governing body",
            "board meeting", "board papers", "board pack", "meeting papers",
            "board minutes", "trust minutes", "committee papers", "committee minutes"
        ]
        
        # Month indicators to help identify meeting documents
        months = [
            "january", "february", "march", "april", "may", "june", "july",
            "august", "september", "october", "november", "december",
            "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec"
        ]
        
        # Check for date patterns that often indicate meeting documents
        has_date = bool(
            re.search(r'\d{1,2}[._-]\d{1,2}[._-](?:20)?\d{2}', url_lower) or
            re.search(r'\d{1,2}[._-]\d{1,2}[._-](?:20)?\d{2}', title_lower) or
            any(month in title_lower for month in months)
        )
        
        # Check for year indicators (2023, 2024, or 2025)
        has_recent_year = any(str(year) in url_lower or str(year) in title_lower for year in [2023, 2024, 2025])
        
        # Strong indicators - if any of these combinations are present, it's very likely a board paper
        if has_recent_year:
            if any(term in url_lower or term in title_lower for term in key_terms):
                print(f"Found likely recent board document: {title} ({url})")
                return True
        
        # Check for combinations of key terms that strongly indicate a board paper
        term_combinations = [
            ["board", "paper"],
            ["board", "meeting"],
            ["trust", "board"],
            ["governing", "body"],
            ["board", "agenda"],
            ["board", "minutes"],
            ["director", "meeting"],
            ["committee", "paper"],
            ["board", "pack"],
            ["public", "board"],
            ["icb", "meeting"],
            ["icb", "papers"],
            ["integrated", "care", "board"],
            ["please find", "papers"],
            ["please find", "agenda"],
            ["meeting", "papers"],
            ["meeting", "pack"],
            ["board", "report"],
            ["trust", "minutes"],
            ["governing", "papers"]
        ]
        
        for combo in term_combinations:
            if all(term in url_lower or term in title_lower for term in combo):
                print(f"Found likely board document (term combination): {title} ({url})")
                return True
        
        # If it has a date and key terms, it's likely a board paper
        if has_date:
            if any(term in url_lower or term in title_lower for term in key_terms):
                print(f"Found likely dated board document: {title} ({url})")
                return True
        
        # Check board paper patterns
        for pattern in self.board_paper_patterns:
            if re.search(pattern, url_lower) or re.search(pattern, title_lower):
                print(f"Found likely board paper (matched pattern): {title} ({url})")
                return True
        
        # If it's a document with certain keywords in the path
        if is_document:
            path_keywords = ["board", "meeting", "public", "trust", "papers", "minutes", "agenda"]
            if any(keyword in url_lower.split('/')[-2:] for keyword in path_keywords):
                print(f"Found likely board document (path keywords): {title} ({url})")
                return True
        
        return False

    def normalize_url(self, base_url: str, link_url: str) -> str:
        """Convert a relative URL to an absolute URL"""
        if not link_url:
            return None
            
        if not link_url.startswith('http'):
            # Handle different formats of relative URLs
            if link_url.startswith('/'):
                # Get domain from base_url
                parsed_base = urllib.parse.urlparse(base_url)
                domain = f"{parsed_base.scheme}://{parsed_base.netloc}"
                return domain + link_url
            else:
                # Handle relative path without leading slash
                return urllib.parse.urljoin(base_url, link_url)
        return link_url

    async def extract_links_from_html(self, html_content: str, base_url: str) -> Dict[str, Any]:
        """Extract all links from HTML content and categorize them"""
        if not html_content:
            return {"board_papers": [], "navigation_links": []}
            
        soup = BeautifulSoup(html_content, 'html.parser')
        result = {
            "board_papers": [],
            "navigation_links": []
        }
        
        # Find all links
        for link in soup.find_all('a', href=True):
            url = link['href']
            title = link.get_text().strip()
            
            # Skip empty links, javascript, mailto, etc.
            if not url or url.startswith(('javascript:', 'mailto:', 'tel:', '#')):
                continue
                
            # Normalize the URL
            normalized_url = self.normalize_url(base_url, url)
            if not normalized_url:
                continue
                
            # Check if it's a board paper
            if self.is_likely_board_paper(normalized_url, title):
                date = self.extract_date_from_title(title)
                result["board_papers"].append({
                    "title": title if title else os.path.basename(normalized_url),
                    "url": normalized_url,
                    "date": date
                })
            # Check if it's a navigation link we should follow
            elif self.should_follow_link(title, normalized_url):
                result["navigation_links"].append({
                    "text": title,
                    "url": normalized_url
                })
        
        return result

    async def crawl_for_documents(self, url: str, crawler: AsyncWebCrawler) -> Dict[str, Any]:
        """Directly crawl a page looking for document links"""
        result = {
            "board_papers": [],
            "navigation_links": []
        }
        
        try:
            # Simple run config
            run_config = CrawlerRunConfig(
                word_count_threshold=1  # Ensure we don't filter out content
            )
            
            # Fetch the page
            crawl_result = await crawler.arun(
                url=url,
                config=run_config
            )
            
            if not crawl_result.success:
                return result
                
            # Check if this page is an archive page with 2023 papers
            if "2023" in url:
                print(f"Examining potential 2023 archive page more closely: {url}")
                
            # Parse HTML with BeautifulSoup
            soup = BeautifulSoup(crawl_result.cleaned_html, 'html.parser')
            
            # Look for any elements with "2023" to help identify archive pages
            year_elements = soup.find_all(string=lambda text: "2023" in text if text else False)
            if year_elements:
                print(f"Found {len(year_elements)} elements with '2023' text on {url}")
            
            # Look for all links
            for link in soup.find_all('a', href=True):
                link_url = link['href']
                link_text = link.get_text().strip()
                
                # Skip empty links, javascript, mailto, etc.
                if not link_url or link_url.startswith(('javascript:', 'mailto:', 'tel:', '#')):
                    continue
                    
                # Normalize the URL
                normalized_url = self.normalize_url(url, link_url)
                if not normalized_url:
                    continue
                
                # Special handling for potential 2023 links
                if "2023" in normalized_url or "2023" in link_text:
                    # Prioritize this as a navigation link if it's not a document
                    if not any(normalized_url.lower().endswith(ext) for ext in ['.pdf', '.doc', '.docx', '.xls', '.xlsx']):
                        result["navigation_links"].append({
                            "text": link_text,
                            "url": normalized_url
                        })
                        continue
                
                # Check if it's a document (PDF, DOC, etc.)
                if any(normalized_url.lower().endswith(ext) for ext in ['.pdf', '.doc', '.docx', '.xls', '.xlsx']):
                    # Check if it's likely a board paper
                    if self.is_likely_board_paper(normalized_url, link_text):
                        # Extract date if available
                        date = self.extract_date_from_title(link_text)
                        
                        # Add to result
                        result["board_papers"].append({
                            "title": link_text if link_text else os.path.basename(normalized_url),
                            "url": normalized_url,
                            "date": date
                        })
                    # If not a board paper but worth navigating to
                    elif self.should_follow_link(link_text, normalized_url):
                        result["navigation_links"].append({
                            "text": link_text,
                            "url": normalized_url
                        })
                # If not a document but worth navigating to
                elif self.should_follow_link(link_text, normalized_url):
                    result["navigation_links"].append({
                        "text": link_text,
                        "url": normalized_url
                    })
            
            # Special case for tables that often contain board papers
            for table in soup.find_all('table'):
                for row in table.find_all('tr'):
                    cells = row.find_all(['td', 'th'])
                    if len(cells) >= 2:  # Likely a table with title and link
                        title_cell = cells[0].get_text().strip()
                        
                        # Look for links in any cell
                        links = row.find_all('a', href=True)
                        for link in links:
                            link_url = link['href']
                            link_text = link.get_text().strip() or title_cell
                            
                            # Normalize the URL
                            normalized_url = self.normalize_url(url, link_url)
                            if not normalized_url:
                                continue
                                
                            # Special check for 2023 papers in tables
                            if "2023" in normalized_url or "2023" in link_text or "2023" in title_cell:
                                if any(normalized_url.lower().endswith(ext) for ext in ['.pdf', '.doc', '.docx', '.xls', '.xlsx']):
                                    print(f"Found likely 2023 board paper in table: {link_text or title_cell} ({normalized_url})")
                                    date = self.extract_date_from_title(link_text or title_cell)
                                    result["board_papers"].append({
                                        "title": link_text or title_cell,
                                        "url": normalized_url,
                                        "date": date
                                    })
                                    continue
                                
                            # Check if it's a document
                            if any(normalized_url.lower().endswith(ext) for ext in ['.pdf', '.doc', '.docx', '.xls', '.xlsx']):
                                # Check if it's likely a board paper
                                if self.is_likely_board_paper(normalized_url, link_text):
                                    # Extract date if available
                                    date = self.extract_date_from_title(link_text or title_cell)
                                    
                                    # Add to result
                                    result["board_papers"].append({
                                        "title": link_text or title_cell,
                                        "url": normalized_url,
                                        "date": date
                                    })
            
        except Exception as e:
            print(f"Error in document crawling for {url}: {e}")
            
        return result
        
    async def _find_board_papers_for_org(self, org: Dict[str, str], max_pages: int = 30) -> List[Dict[str, Any]]:
        """Find board papers for a single NHS organization"""
        if not org.get("url"):
            print(f"No URL available for {org['name']}")
            return []
            
        papers = []
        
        # Reset visited URLs for each organization
        self.visited_urls = set()
        
        try:
            # Configure the browser and crawler - use more conservative settings
            browser_config = BrowserConfig(
                headless=True,
                verbose=True
            )
            
            # Set up the LLM extraction strategy if not using simulation
            run_config = None
            if not self.use_simulation:
                extraction_strategy = LLMExtractionStrategy(
                    llm_config=LLMConfig(
                        provider="openai/gpt-3.5-turbo",
                        api_token=Config.OPENAI_API_KEY
                    ),
                    schema=self._get_extraction_schema(),
                    extraction_type="schema",
                    instruction=self._get_extraction_instructions()
                )
                
                # Configure the crawler run
                run_config = CrawlerRunConfig(
                    extraction_strategy=extraction_strategy,
                    word_count_threshold=1  # Ensure we don't filter out too much content
                )
            else:
                # Simple run config for simulation mode
                run_config = CrawlerRunConfig(
                    word_count_threshold=1  # Ensure we don't filter out content
                )
            
            # Create a single crawler instance for the entire organization
            try:
                async with AsyncWebCrawler(config=browser_config) as crawler:
                    print(f"Successfully created crawler instance for {org['name']}")
                    
                    # Define more specific paths for board papers/meetings
                    base_url = org["url"]
                    common_paths = [
                        "",  # Root URL
                        "/about-us/board", 
                        "/about-us/board-of-directors",
                        "/about-us/board-meetings",
                        "/about-us/publications",
                        "/publications",
                        "/board",
                        "/board-meetings",
                        "/board-papers",
                        "/trust-board",
                        "/about/board",
                        "/about/board-meetings",
                        "/about-us/our-board",
                        "/about-us/trust-board",
                        "/about/our-board",
                        "/about/trust-board",
                        "/about-us",
                        "/about",
                        "/corporate-information/board-papers",
                        "/corporate-information/board-meetings",
                        "/trust-information/board-meetings",
                        "/about-us/corporate-information",
                        "/our-trust/who-we-are/board-of-directors",
                        "/about-us/who-we-are/board-of-directors",
                        "/corporate-information/board",
                        "/corporate-information",
                        "/about-us/corporate",
                        "/directors",
                        "/who-we-are/board-of-directors"
                    ]
                    
                    # Try the most likely pages first
                    paths_checked = 0
                    for path in common_paths:
                        if paths_checked >= max_pages:
                            break
                            
                        # Construct full URL
                        current_url = self.normalize_url(base_url, path)
                        if not current_url or current_url in self.visited_urls:
                            continue
                            
                        self.visited_urls.add(current_url)
                        paths_checked += 1
                        
                        print(f"Checking {current_url} for {org['name']} - {paths_checked}/{len(common_paths)}")
                        
                        try:
                            # First, try direct document crawling approach
                            document_result = await self.crawl_for_documents(current_url, crawler)
                            
                            # Add any board papers found directly
                            if document_result.get("board_papers"):
                                print(f"Direct crawling found {len(document_result['board_papers'])} board papers on {current_url}")
                                for paper in document_result.get("board_papers", []):
                                    paper_url = paper.get("url", "")
                                    if not paper_url:
                                        continue
                                    
                                    # Add found paper to the list
                                    papers.append({
                                        "title": paper.get("title", "Unknown Title"),
                                        "url": paper_url,
                                        "date": paper.get("date", ""),
                                        "organization": org["name"],
                                        "org_type": org["type"],
                                        "org_url": org["url"],
                                        "found_page": current_url
                                    })
                            
                            # If few papers found, look for navigation links to explore
                            if document_result.get("navigation_links"):
                                nav_links = document_result.get("navigation_links", [])
                                print(f"Found {len(nav_links)} navigation links to explore")
                                
                                # Check if any link text suggests board papers or meetings
                                board_related_links = []
                                for link in nav_links:
                                    link_text = link.get("text", "").lower()
                                    link_url = link.get("url", "").lower()
                                    
                                    # Check if link is likely to lead to board papers
                                    if any(pattern in link_text or pattern in link_url for pattern in ["board", "meeting", "minute", "agenda", "paper", "director"]):
                                        board_related_links.append(link)
                                
                                # Prioritize board-related links
                                explore_links = board_related_links[:8] if board_related_links else nav_links[:5]
                                
                                # Explore a limited number of navigation links
                                nav_links_explored = 0
                                for link in explore_links:
                                    if paths_checked >= max_pages:
                                        break
                                        
                                    link_url = link.get("url")
                                    if not link_url or link_url in self.visited_urls:
                                        continue
                                    
                                    self.visited_urls.add(link_url)
                                    paths_checked += 1
                                    nav_links_explored += 1
                                    
                                    print(f"Exploring navigation link {nav_links_explored}: {link_url}")
                                    
                                    # Try to find papers on this linked page
                                    sub_result = await self.crawl_for_documents(link_url, crawler)
                                    
                                    if sub_result.get("board_papers"):
                                        print(f"Found {len(sub_result['board_papers'])} more papers on {link_url}")
                                        for paper in sub_result.get("board_papers", []):
                                            paper_url = paper.get("url", "")
                                            if not paper_url:
                                                continue
                                            
                                            papers.append({
                                                "title": paper.get("title", "Unknown Title"),
                                                "url": paper_url,
                                                "date": paper.get("date", ""),
                                                "organization": org["name"],
                                                "org_type": org["type"],
                                                "org_url": org["url"],
                                                "found_page": link_url
                                            })
                        
                        except Exception as e:
                            print(f"Error processing {current_url}: {e}")
                            continue
                            
                    print(f"Finished crawling {org['name']} - checked {paths_checked} pages, found {len(papers)} papers")
            
            except Exception as e:
                print(f"Failed to create crawler instance for {org['name']}: {e}")
                
        except Exception as e:
            print(f"Error in crawling setup for {org['name']} ({org['url']}): {e}")
            
        return papers
    
    def find_board_papers(self) -> List[Dict[str, Any]]:
        """Find board papers across all NHS organizations"""
        print("Starting to find board papers...")
        
        # Define some common NHS URLs to test
        test_urls = [
            "https://www.hct.nhs.uk/",
            "https://www.imperial.nhs.uk/",
            "https://www.guysandstthomas.nhs.uk/",
            "https://www.england.nhs.uk/",
            "https://www.bedfordshirehospitals.nhs.uk/",
            "https://bedfordshirelutonandmiltonkeynes.icb.nhs.uk/"
        ]
        
        # Use a direct scraping approach instead of browser automation
        import requests
        from bs4 import BeautifulSoup
        import time
        
        all_papers = []
        
        # Process each URL
        for url in test_urls:
            try:
                print(f"Processing {url}...")
                papers_found = []
                
                # Extract domain name for organization name
                domain = url.split('//')[-1].split('/')[0]
                org_name = domain.replace('www.', '').replace('.org.uk', '').replace('.nhs.uk', '')
                org_name = org_name.upper()
                
                # Common paths that might contain board papers
                common_paths = [
                    "",  # Root URL
                    "/about-us/board", 
                    "/about-us/board-of-directors",
                    "/about-us/board-meetings",
                    "/about-us/publications",
                    "/publications",
                    "/board",
                    "/board-meetings",
                    "/board-papers",
                    "/trust-board",
                    "/about/board",
                    "/about/board-meetings"
                ]
                
                # Process each path
                for path in common_paths:
                    try:
                        path_url = urllib.parse.urljoin(url, path)
                        print(f"Checking {path_url}")
                        
                        response = requests.get(path_url, timeout=10)
                        if response.status_code == 200:
                            soup = BeautifulSoup(response.text, 'html.parser')
                            
                            # Find all links
                            for link in soup.find_all('a', href=True):
                                href = link['href']
                                text = link.get_text().strip()
                                
                                # Skip empty, javascript or anchor links
                                if not href or href.startswith(('javascript:', '#', 'mailto:', 'tel:')):
                                    continue
                                    
                                # Normalize URL
                                full_url = urllib.parse.urljoin(path_url, href)
                                
                                # Check if it's a likely board paper
                                if self.is_likely_board_paper(full_url, text):
                                    date = self.extract_date_from_title(text)
                                    print(f"Found board paper: {text} ({full_url})")
                                    papers_found.append({
                                        "title": text if text else "Unknown",
                                        "url": full_url,
                                        "date": date,
                                        "organization": org_name,
                                        "org_type": "Trust" if "nhs.uk" in url else "ICB",
                                        "org_url": url,
                                        "found_page": path_url
                                    })
                    except Exception as e:
                        print(f"Error checking path {path_url}: {str(e)}")
                
                print(f"Found {len(papers_found)} papers for {org_name}")
                all_papers.extend(papers_found)
                
                # Add a delay between sites
                time.sleep(Config.CRAWL_DELAY)
                
            except Exception as e:
                print(f"Error processing {url}: {str(e)}")
                continue
        
        # Filter out duplicate papers
        filtered_papers = self.filter_duplicate_papers(all_papers)
        
        print(f"Total board papers found: {len(filtered_papers)} (after removing duplicates)")
        return filtered_papers

    def normalize_board_paper_url(self, url: str) -> str:
        """Normalize board paper URLs to help with duplicate detection"""
        if not url:
            return ""
        
        # Remove query parameters and fragments
        url = url.split('?')[0].split('#')[0]
        
        # Normalize file extensions
        if url.lower().endswith('.pdf'):
            # For PDFs, keep the full path as it's likely unique
            return url
        
        return url
        
    def filter_duplicate_papers(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter out duplicate papers based on normalized URLs and titles"""
        if not papers:
            return []
            
        print(f"Filtering duplicates from {len(papers)} papers...")
        unique_papers = {}
        
        for paper in papers:
            # Get paper URL and normalize it
            paper_url = paper.get("url", "")
            normalized_url = self.normalize_board_paper_url(paper_url)
            
            # Get paper title and normalize it
            paper_title = paper.get("title", "").lower().strip()
            
            # Create a unique key based on normalized URL and title
            # For PDFs, the URL is often sufficient as a unique identifier
            if normalized_url.lower().endswith('.pdf'):
                key = normalized_url
            else:
                key = f"{normalized_url}|{paper_title}"
                
            # If we haven't seen this paper before, add it
            if key not in unique_papers:
                unique_papers[key] = paper
                
        result = list(unique_papers.values())
        print(f"After filtering: {len(result)} unique papers (removed {len(papers) - len(result)} duplicates)")
        return result

    async def find_board_papers_for_urls(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Find board papers for specific URLs provided"""
        print(f"Starting to find board papers for {len(urls)} specific URLs...")
        
        # Use a direct scraping approach instead of browser automation
        import requests
        from bs4 import BeautifulSoup
        import time
        from urllib.parse import urljoin, urlparse
        import warnings
        from datetime import datetime
        warnings.filterwarnings('ignore', message='Unverified HTTPS request')
        
        all_papers = []
        visited_urls = set()
        
        def extract_date(text: str, url: str) -> str:
            """Extract date from text or URL using various patterns"""
            # Common date formats
            date_patterns = [
                r'(\d{1,2})[-/_](\d{1,2})[-/_](20\d{2})',  # dd-mm-yyyy
                r'(20\d{2})[-/_](\d{1,2})[-/_](\d{1,2})',  # yyyy-mm-dd
                r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[-/_\s](20\d{2})',  # month-yyyy
                r'(20\d{2})[-/_\s](jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*',  # yyyy-month
                r'(january|february|march|april|may|june|july|august|september|october|november|december)[a-z]*[-/_\s](20\d{2})',  # fullmonth-yyyy
                r'(20\d{2})[-/_\s](january|february|march|april|may|june|july|august|september|october|november|december)[a-z]*'  # yyyy-fullmonth
            ]
            
            # Try to find date in text first
            text_lower = text.lower()
            for pattern in date_patterns:
                match = re.search(pattern, text_lower)
                if match:
                    return match.group(0)
            
            # Then try URL
            url_lower = url.lower()
            for pattern in date_patterns:
                match = re.search(pattern, url_lower)
                if match:
                    return match.group(0)
            
            # If no date found but contains 2024/2025, use that
            if '2024' in text_lower or '2024' in url_lower:
                return '2024'
            if '2025' in text_lower or '2025' in url_lower:
                return '2025'
            
            return ""
        
        def is_valid_url(url):
            """Check if URL is valid and belongs to the same domain"""
            try:
                return bool(urlparse(url).netloc)
            except:
                return False
        
        def is_same_domain(url1, url2):
            """Check if two URLs belong to the same domain"""
            try:
                domain1 = urlparse(url1).netloc
                domain2 = urlparse(url2).netloc
                return domain1 == domain2
            except:
                return False
        
        def should_explore_link(text, url):
            """Determine if a link should be explored further"""
            text_lower = text.lower()
            url_lower = url.lower()
            
            # Keywords that suggest board-related content
            board_keywords = [
                "board", "meeting", "minutes", "papers", "agenda", "trust",
                "governing body", "committee", "icb", "integrated care",
                "publications", "corporate", "about us", "past meetings"
            ]
            
            # Check if any keyword is present in either text or URL
            return any(keyword in text_lower or keyword in url_lower for keyword in board_keywords)
        
        # Process each URL
        for url in urls:
            try:
                print(f"\nProcessing {url}...")
                papers_found = []
                
                # Extract domain name for organization name
                domain = url.split('//')[-1].split('/')[0]
                org_name = domain.replace('www.', '').replace('.org.uk', '').replace('.nhs.uk', '')
                org_name = org_name.upper()
                
                def process_page(page_url, depth=0, max_depth=3):
                    """Process a page recursively to find board papers and more links"""
                    if depth > max_depth or page_url in visited_urls:
                        return
                    
                    visited_urls.add(page_url)
                    print(f"{'  ' * depth}Checking {page_url}")
                    
                    try:
                        # Use verify=False to bypass SSL certificate verification
                        response = requests.get(page_url, timeout=15, verify=False)
                        if response.status_code == 200:
                            soup = BeautifulSoup(response.text, 'html.parser')
                            
                            # First, look for board papers in any tables
                            for table in soup.find_all('table'):
                                for row in table.find_all('tr'):
                                    cells = row.find_all(['td', 'th'])
                                    if cells:
                                        # Get text from all cells
                                        row_text = ' '.join(cell.get_text().strip() for cell in cells)
                                        # Find links in the row
                                        for link in row.find_all('a', href=True):
                                            href = link['href']
                                            if not href or href.startswith(('javascript:', '#', 'mailto:', 'tel:')):
                                                continue
                                            full_url = urljoin(page_url, href)
                                            if not is_valid_url(full_url) or not is_same_domain(url, full_url):
                                                continue
                                            if self.is_likely_board_paper(full_url, row_text):
                                                date = extract_date(row_text, full_url)
                                                print(f"{'  ' * depth}Found board paper in table: {row_text} ({full_url}) - Date: {date}")
                                                papers_found.append({
                                                    "title": row_text if row_text else "Unknown",
                                                    "url": full_url,
                                                    "date": date,
                                                    "organization": org_name,
                                                    "org_type": "Trust" if "nhs.uk" in url else "ICB",
                                                    "org_url": url,
                                                    "found_page": page_url
                                                })
                            
                            # Then look for links in the page content
                            links_to_explore = []
                            for link in soup.find_all('a', href=True):
                                href = link['href']
                                text = link.get_text().strip()
                                
                                # Skip empty, javascript or anchor links
                                if not href or href.startswith(('javascript:', '#', 'mailto:', 'tel:')):
                                    continue
                                    
                                # Normalize URL
                                full_url = urljoin(page_url, href)
                                
                                # Skip if not valid URL or different domain
                                if not is_valid_url(full_url) or not is_same_domain(url, full_url):
                                    continue
                                
                                # Check if it's a likely board paper
                                if self.is_likely_board_paper(full_url, text):
                                    date = extract_date(text, full_url)
                                    print(f"{'  ' * depth}Found board paper: {text} ({full_url}) - Date: {date}")
                                    papers_found.append({
                                        "title": text if text else "Unknown",
                                        "url": full_url,
                                        "date": date,
                                        "organization": org_name,
                                        "org_type": "Trust" if "nhs.uk" in url else "ICB",
                                        "org_url": url,
                                        "found_page": page_url
                                    })
                                # If not a paper but looks promising, add to exploration list
                                elif should_explore_link(text, full_url):
                                    links_to_explore.append((text, full_url))
                            
                            # Sort links by relevance (board-related keywords first)
                            links_to_explore.sort(key=lambda x: sum(1 for kw in ["board", "meeting", "minutes", "papers"] if kw in x[0].lower()), reverse=True)
                            
                            # Explore the most promising links first (up to 10 per page)
                            for _, link_url in links_to_explore[:10]:
                                if link_url not in visited_urls:
                                    process_page(link_url, depth + 1, max_depth)
                                    
                    except Exception as e:
                        print(f"{'  ' * depth}Error checking path {page_url}: {str(e)}")
                
                # Start with the main URL
                process_page(url)
                
                # Also check some common paths directly
                common_paths = [
                    "/about-us/board", 
                    "/about-us/our-board",
                    "/about-us/board-meetings",
                    "/about-us/trust-board",
                    "/about-us/icb",
                    "/about-us/integrated-care-board",
                    "/board-meetings",
                    "/board-papers",
                    "/meetings-and-papers",
                    "/about-us/our-nhs-integrated-care-board-icb"
                ]
                
                for path in common_paths:
                    path_url = urljoin(url, path)
                    if path_url not in visited_urls:
                        process_page(path_url)
                
                print(f"Found {len(papers_found)} papers for {org_name}")
                all_papers.extend(papers_found)
                
                # Add a delay between sites
                time.sleep(Config.CRAWL_DELAY)
                
            except Exception as e:
                print(f"Error processing {url}: {str(e)}")
                continue
        
        # Filter out duplicate papers
        filtered_papers = self.filter_duplicate_papers(all_papers)
        
        # Count papers by year
        papers_by_year = {}
        recent_papers = []
        current_year = datetime.now().year
        
        for paper in filtered_papers:
            date_str = paper.get('date', '')
            if date_str:
                # Extract year from date string
                year_match = re.search(r'20\d{2}', date_str)
                if year_match:
                    year = int(year_match.group(0))
                    papers_by_year[year] = papers_by_year.get(year, 0) + 1
                    if year >= current_year:
                        recent_papers.append(paper)
        
        print(f"\nPapers found by year:")
        for year in sorted(papers_by_year.keys()):
            print(f"  {year}: {papers_by_year[year]} papers")
        
        print(f"\nTotal board papers found: {len(filtered_papers)} (after removing duplicates)")
        print(f"Papers from {current_year} onwards: {len(recent_papers)}")
        
        return filtered_papers

def test_specific_urls():
    """Test function that finds board papers for specific test URLs using a simple direct approach."""
    import requests
    from bs4 import BeautifulSoup
    import re
    import time
    from urllib.parse import urljoin
    
    try:
        test_urls = [
            "https://www.hct.nhs.uk/",
            "https://bedfordshirelutonandmiltonkeynes.icb.nhs.uk/",
            "https://www.bedfordshirehospitals.nhs.uk/",
            "https://www.mkuh.nhs.uk/",
            "https://www.imperial.nhs.uk/",
            "https://www.guysandstthomas.nhs.uk/",
            "https://www.england.nhs.uk/"
        ]

        # Don't initialize the full crawler - that's causing issues
        print(f"Starting to find board papers for {len(test_urls)} specific URLs...")
        all_papers = []

        # Paper detection patterns
        paper_patterns = [
            r'board[-_\s]?papers?',
            r'board[-_\s]?meeting',
            r'trust[-_\s]?board',
            r'governing[-_\s]?body',
            r'minutes',
            r'agenda',
            r'icb[-_\s]?board',
            r'board[-_\s]?of[-_\s]?directors'
        ]
        
        # Common paths that might contain board papers
        common_paths = [
            "/about-us/board", 
            "/about-us/board-of-directors",
            "/about-us/board-meetings",
            "/about-us/publications",
            "/publications",
            "/board",
            "/board-meetings",
            "/board-papers",
            "/trust-board",
            "/about/board",
            "/about/board-meetings"
        ]
        
        # Process each URL
        for url in test_urls:
            try:
                print(f"Processing {url}...")
                papers_found = []
                
                # First check the main URL
                try:
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        
                        # Find all links
                        for link in soup.find_all('a', href=True):
                            href = link['href']
                            text = link.get_text().strip()
                            
                            # Skip empty, javascript or anchor links
                            if not href or href.startswith(('javascript:', '#', 'mailto:', 'tel:')):
                                continue
                                
                            # Normalize URL
                            full_url = urljoin(url, href)
                            
                            # Check if it's a PDF or document
                            is_document = any(full_url.lower().endswith(ext) for ext in 
                                             ['.pdf', '.doc', '.docx', '.xls', '.xlsx'])
                            
                            # Check if it matches our patterns
                            matches_pattern = any(re.search(pattern, text.lower()) or 
                                                 re.search(pattern, full_url.lower()) 
                                                 for pattern in paper_patterns)
                            
                            if is_document and matches_pattern:
                                print(f"Found board paper: {text} ({full_url})")
                                papers_found.append({
                                    "title": text if text else "Unknown",
                                    "url": full_url,
                                    "date": "",
                                    "organization": url.split('//')[-1].split('/')[0]
                                })
                except Exception as e:
                    print(f"Error processing main URL {url}: {str(e)}")
                
                # Then check common paths
                for path in common_paths:
                    try:
                        path_url = urljoin(url, path)
                        print(f"Checking {path_url}")
                        
                        response = requests.get(path_url, timeout=10)
                        if response.status_code == 200:
                            soup = BeautifulSoup(response.text, 'html.parser')
                            
                            # Find all links
                            for link in soup.find_all('a', href=True):
                                href = link['href']
                                text = link.get_text().strip()
                                
                                # Skip empty, javascript or anchor links
                                if not href or href.startswith(('javascript:', '#', 'mailto:', 'tel:')):
                                    continue
                                    
                                # Normalize URL
                                full_url = urljoin(path_url, href)
                                
                                # Check if it's a PDF or document
                                is_document = any(full_url.lower().endswith(ext) for ext in 
                                                 ['.pdf', '.doc', '.docx', '.xls', '.xlsx'])
                                
                                # Check if it matches our patterns
                                matches_pattern = any(re.search(pattern, text.lower()) or 
                                                     re.search(pattern, full_url.lower()) 
                                                     for pattern in paper_patterns)
                                
                                if is_document and matches_pattern:
                                    print(f"Found board paper: {text} ({full_url})")
                                    papers_found.append({
                                        "title": text if text else "Unknown",
                                        "url": full_url,
                                        "date": "",
                                        "organization": url.split('//')[-1].split('/')[0]
                                    })
                    except Exception as e:
                        print(f"Error checking path {path_url}: {str(e)}")
                
                print(f"Found {len(papers_found)} papers for {url}")
                all_papers.extend(papers_found)
                
                # Add a delay between sites to avoid overwhelming them
                time.sleep(2)
                
            except Exception as e:
                print(f"Error processing {url}: {str(e)}")
                continue

        # Simple deduplication by URL
        unique_urls = set()
        filtered_papers = []
        for paper in all_papers:
            if paper['url'] not in unique_urls:
                unique_urls.add(paper['url'])
                filtered_papers.append(paper)
                
        # Count papers by year
        years = {}
        for paper in filtered_papers:
            for year in range(2020, 2026):
                if str(year) in paper['url'] or str(year) in paper['title']:
                    years[year] = years.get(year, 0) + 1
                    break

        print(f"Total board papers found: {len(filtered_papers)} (after removing duplicates)")
        if years:
            print("Years found in board papers:")
            for year, count in sorted(years.items()):
                print(f"  {year}: {count} papers")
            
        print(f"Test completed. Found {len(filtered_papers)} unique papers total.")
        
        return filtered_papers
    except Exception as e:
        print(f"Error in test_specific_urls: {str(e)}")
        return []

if __name__ == "__main__":
    # Test the crawler
    test_specific_urls() 