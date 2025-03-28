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

    def should_follow_link(self, text: str, url: str) -> bool:
        """Determine if a link should be followed for deeper crawling"""
        text_lower = text.lower()
        url_lower = url.lower()
        
        # Skip if it's a document
        if any(url_lower.endswith(ext) for ext in ['.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx']):
            return False
            
        # Skip common non-board sections
        skip_sections = [
            "news", "contact", "search", "accessibility", "privacy", "cookies",
            "terms", "sitemap", "feedback", "complaints", "jobs", "careers",
            "vacancies", "events", "training", "login", "register", "social",
            "twitter", "facebook", "youtube", "linkedin", "instagram"
        ]
        if any(section in url_lower or section in text_lower for section in skip_sections):
            return False
            
        # Strong indicators - these sections are very likely to contain board papers
        strong_indicators = [
            "board-papers", "board-meetings", "trust-board", "governing-body",
            "public-board", "committee-papers", "meeting-papers", "board-minutes",
            "trust-minutes", "governing-body-papers", "icb-board", "integrated-care-board"
        ]
        if any(indicator in url_lower.replace("/", "-") for indicator in strong_indicators):
            return True
            
        # Check for promising combinations of terms
        term_combinations = [
            ["board", "papers"],
            ["board", "meetings"],
            ["trust", "board"],
            ["governing", "body"],
            ["public", "board"],
            ["committee", "papers"],
            ["meeting", "papers"],
            ["board", "minutes"],
            ["trust", "minutes"],
            ["icb", "board"],
            ["integrated", "care", "board"]
        ]
        for combo in term_combinations:
            if all(term in text_lower or term in url_lower for term in combo):
                return True
                
        # Check for common paths that might lead to board papers
        common_paths = [
            "/about-us/board",
            "/about-us/our-board",
            "/about-us/trust-board",
            "/about-us/governing-body",
            "/about/board",
            "/about/trust-board",
            "/board",
            "/trust-board",
            "/governing-body",
            "/meetings",
            "/papers",
            "/minutes",
            "/corporate/board",
            "/corporate/governing-body",
            "/corporate/meetings",
            "/corporate/papers"
        ]
        if any(path in url_lower for path in common_paths):
            return True
            
        # Check for promising text in link
        promising_terms = [
            "view all board papers",
            "view board papers",
            "view papers",
            "view minutes",
            "download papers",
            "download minutes",
            "past meetings",
            "previous meetings",
            "meeting archive",
            "papers archive",
            "minutes archive"
        ]
        if any(term in text_lower for term in promising_terms):
            return True
            
        # Check for dates in text that might indicate meeting listings
        date_indicators = [
            r'\b20(24|25)\b',  # Years 2024-2025
            r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* +20\d{2}\b',  # Month Year
            r'\b20\d{2} +(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b'  # Year Month
        ]
        if any(re.search(pattern, text_lower) for pattern in date_indicators):
            # Only follow date links if they also have board-related terms
            return any(term in text_lower or term in url_lower 
                      for term in ["board", "meeting", "papers", "minutes", "governing"])
            
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
        
        # First check if it's a document
        document_extensions = ['.pdf', '.doc', '.docx']  # Removed .ppt, .pptx, .xls, .xlsx as they're usually supporting documents
        is_document = any(url_lower.endswith(ext) for ext in document_extensions)
        
        # Also check for URLs that contain these extensions but might have additional parameters
        if not is_document:
            is_document = any(f"{ext}?" in url_lower or f"{ext}&" in url_lower for ext in document_extensions)
            
        if not is_document:
            return False
            
        # Exclude patterns - documents we don't want
        exclude_patterns = [
            r'register[-_\s]?of[-_\s]?interests?',
            r'declaration[-_\s]?of[-_\s]?interests?',
            r'annual[-_\s]?report',
            r'accounts',
            r'financial[-_\s]?report',
            r'agenda[-_\s]?only',
            r'action[-_\s]?log',
            r'attendance[-_\s]?record',
            r'supporting[-_\s]?papers?',
            r'appendix',
            r'annex'
        ]
        
        # If any exclude pattern matches, it's not a board paper
        if any(re.search(pattern, title_lower) or re.search(pattern, url_lower) for pattern in exclude_patterns):
            return False
        
        # Key terms that indicate actual board papers
        key_terms = [
            "board papers",
            "board meeting papers",
            "trust board papers",
            "public board papers",
            "board pack",
            "board minutes",
            "minutes of the board",
            "board meeting minutes",
            "trust board minutes",
            "public board minutes",
            "board of directors minutes",
            "governing body papers",
            "governing body minutes",
            "icb board papers",
            "icb board minutes",
            "integrated care board papers",
            "integrated care board minutes"
        ]
        
        # Check for exact key terms first (these are strong indicators)
        if any(term in title_lower or term in url_lower for term in key_terms):
            return True
            
        # Check for combinations that must appear together
        term_combinations = [
            ["board", "papers"],
            ["board", "minutes"],
            ["trust", "board", "papers"],
            ["trust", "board", "minutes"],
            ["governing", "body", "papers"],
            ["governing", "body", "minutes"],
            ["public", "board", "papers"],
            ["public", "board", "minutes"]
        ]
        
        for combo in term_combinations:
            if all(term in title_lower or term in url_lower for term in combo):
                return True
        
        # If we have a date and "board" in the title/url, it's likely a board paper
        has_date = bool(
            re.search(r'\d{1,2}[._-]\d{1,2}[._-](?:20)?\d{2}', url_lower) or
            re.search(r'\d{1,2}[._-]\d{1,2}[._-](?:20)?\d{2}', title_lower) or
            any(month in title_lower for month in [
                "january", "february", "march", "april", "may", "june", "july",
                "august", "september", "october", "november", "december",
                "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec"
            ])
        )
        
        if has_date and ("board" in title_lower or "board" in url_lower):
            # Additional check: must contain "papers" or "minutes" if it has a date
            return "papers" in title_lower or "papers" in url_lower or "minutes" in title_lower or "minutes" in url_lower
            
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

    async def crawl_for_documents(self, url: str, crawler: AsyncWebCrawler) -> Dict[str, Any]:
        """Directly crawl a page looking for document links"""
        result = {
            "board_papers": [],
            "navigation_links": []
        }
        
        try:
            # Get the page content
            response = await crawler.get(url)
            if not response or not response.text:
                return result
                
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # First check for frames/iframes that might contain board papers
            frames = soup.find_all(['frame', 'iframe'])
            for frame in frames:
                frame_url = frame.get('src', '')
                if frame_url:
                    frame_url = self.normalize_url(url, frame_url)
                    if frame_url and frame_url not in self.visited_urls:
                        self.visited_urls.add(frame_url)
                        try:
                            frame_response = await crawler.get(frame_url)
                            if frame_response and frame_response.text:
                                frame_soup = BeautifulSoup(frame_response.text, 'html.parser')
                                soup = BeautifulSoup(str(soup) + str(frame_soup), 'html.parser')
                        except Exception as e:
                            print(f"Error processing frame {frame_url}: {e}")
            
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
                
                # Check if it's a document (PDF, DOC, etc.)
                if any(normalized_url.lower().endswith(ext) for ext in ['.pdf', '.doc', '.docx']):
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
                # If not a document but worth navigating to
                elif self.should_follow_link(link_text, normalized_url):
                    result["navigation_links"].append({
                        "text": link_text,
                        "url": normalized_url
                    })
            
            # Special case for tables that often contain board papers
            for table in soup.find_all('table'):
                # Check if this looks like a table of board papers
                table_text = table.get_text().lower()
                if any(term in table_text for term in ["board", "minutes", "papers", "meeting"]):
                    for row in table.find_all('tr'):
                        cells = row.find_all(['td', 'th'])
                        if len(cells) >= 2:  # Likely a table with title and link
                            row_text = ' '.join(cell.get_text().strip() for cell in cells)
                            
                            # Look for links in any cell
                            links = row.find_all('a', href=True)
                            for link in links:
                                link_url = link['href']
                                link_text = link.get_text().strip() or row_text
                                
                                # Normalize the URL
                                normalized_url = self.normalize_url(url, link_url)
                                if not normalized_url:
                                    continue
                                
                                # Check if it's a document
                                if any(normalized_url.lower().endswith(ext) for ext in ['.pdf', '.doc', '.docx']):
                                    # Check if it's likely a board paper
                                    if self.is_likely_board_paper(normalized_url, link_text):
                                        # Extract date if available
                                        date = self.extract_date_from_title(link_text)
                                        
                                        # Add to result
                                        result["board_papers"].append({
                                            "title": link_text,
                                            "url": normalized_url,
                                            "date": date
                                        })
            
            # Look for links in list items that might contain board papers
            for list_item in soup.find_all(['li', 'div']):
                item_text = list_item.get_text().lower()
                if any(term in item_text for term in ["board", "minutes", "papers", "meeting"]):
                    for link in list_item.find_all('a', href=True):
                        link_url = link['href']
                        link_text = link.get_text().strip()
                        
                        # Normalize the URL
                        normalized_url = self.normalize_url(url, link_url)
                        if not normalized_url:
                            continue
                        
                        # Check if it's a document
                        if any(normalized_url.lower().endswith(ext) for ext in ['.pdf', '.doc', '.docx']):
                            # Check if it's likely a board paper
                            if self.is_likely_board_paper(normalized_url, link_text):
                                # Extract date if available
                                date = self.extract_date_from_title(link_text)
                                
                                # Add to result
                                result["board_papers"].append({
                                    "title": link_text,
                                    "url": normalized_url,
                                    "date": date
                                })
            
            # Sort navigation links by relevance
            result["navigation_links"].sort(
                key=lambda x: sum(1 for term in ["board", "meeting", "minutes", "papers"] 
                                if term in x["text"].lower() or term in x["url"].lower()),
                reverse=True
            )
            
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