import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import google.generativeai as genai
import requests
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from .prompt_helper import (
    get_analysis_prompt,
    get_date_extraction_prompt,
    get_metadata_prompt,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class PDFAnalyzer:
    """
    Class for analyzing PDFs for mentions of virtual wards
    using Google's Gemini AI model.
    """

    def __init__(self, api_key=None):
        """Initialize the PDFAnalyzer with optional API key."""
        # Use provided API key or get from environment
        _api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not _api_key:
            raise ValueError("Gemini API key is required")

        # Configure Gemini
        genai.configure(api_key=_api_key)
        logger.debug("PDFAnalyzer initialized")  # Downgrade to debug

        # Initialize models as class attributes
        self._model = genai.GenerativeModel("gemini-2.0-flash")
        self._date_model = genai.GenerativeModel(
            "gemini-2.5-pro-preview-03-25"
        )  # Used for date extraction

        # Cache for downloaded PDFs
        self._download_cache: Dict[str, Dict[str, str]] = {}

    def clean_text(self, text: str) -> str:
        """Clean and normalize text from PDF."""
        # Replace multiple newlines with a single one
        text = " ".join(text.split())
        # Remove any non-standard whitespace
        text = text.replace("\xa0", " ")
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        # Fix common OCR issues with medical terms
        text = text.replace("heartfailure", "heart failure")
        text = text.replace("HeartFailure", "Heart Failure")
        text = text.replace("HEARTFAILURE", "HEART FAILURE")
        text = text.replace("virtualward", "virtual ward")
        text = text.replace("VirtualWard", "Virtual Ward")
        return text

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text content from a PDF file."""
        logger.debug(f"Extracting text from: {pdf_path}")  # Downgrade to debug
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += self.clean_text(page_text) + " "
                else:
                    logger.debug(f"Empty page in {pdf_path}")  # Downgrade to debug

            if not text.strip():
                logger.warning(f"No text extracted from {pdf_path}")
                return "No text could be extracted from this PDF."

            logger.info(f"Successfully extracted {len(reader.pages)} pages")
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise

    def chunk_text(self, text: str, chunk_size: int = 15000) -> List[str]:
        """Split text into chunks of approximately chunk_size characters."""
        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            if start + chunk_size >= text_len:
                chunks.append(text[start:])
                break

            # Find the last whitespace within the chunk_size limit
            split_point = start + chunk_size
            while split_point > start and not text[split_point - 1].isspace():
                split_point -= 1

            # If no good break point found, force split at chunk_size
            if split_point == start:
                split_point = start + chunk_size

            chunks.append(text[start:split_point])
            start = split_point.lstrip()

        logger.debug(f"Split into {len(chunks)} chunks")
        return chunks

    def _parse_gemini_response(self, content: str) -> List[Dict]:
        """Parse the structured response from Gemini into a list of mentions."""
        if content == "NO_RELEVANT_MENTIONS_FOUND":
            logger.info("No relevant term mentions found in chunk")
            return []

        mentions = []
        sections = content.split("---")

        for section in sections:
            if not section.strip():
                continue

            mention = self._extract_mention_from_section(section)
            if mention:
                mentions.append(mention)

        logger.info(f"Found {len(mentions)} term mentions in chunk")
        return mentions

    def _extract_mention_from_section(self, section: str) -> Optional[Dict]:
        """Extract a structured mention from a response section."""
        mention = {}

        # Helper function to extract content between backticks
        def extract_content(marker: str, offset: int) -> Optional[str]:
            start = section.find(f"```{marker}\n") + offset
            if start <= offset - 1:  # Not found
                return None
            end = section.find("```", start)
            if end == -1:  # No closing backticks
                return None
            return section[start:end].strip()

        # Extract each component
        mention["term"] = extract_content("term", 8)
        quotes = extract_content("mentions", 12)
        summary = extract_content("summary", 11)

        if not all([mention["term"], quotes, summary]):
            return None

        # Clean the content
        mention["quotes"] = self.clean_repetitive_text(quotes)
        mention["summary"] = self._clean_summary(summary)

        # Only return if we have valid content after cleaning
        if mention["summary"].strip() and mention["quotes"].strip():
            return mention
        return None

    def _clean_summary(self, summary: str) -> str:
        """Clean a summary by removing 'no mentions found' messages and repetitive text."""
        if not summary:
            return ""

        # Remove "no mentions found" messages
        for pattern in [r"NO_RELEVANT_MENTIONS_FOUND.*$", r"NO MENTIONS FOUND.*$"]:
            summary = re.sub(pattern, "", summary, flags=re.IGNORECASE | re.DOTALL)

        # Clean repetitive text
        return self.clean_repetitive_text(summary.strip())

    def analyze_text_chunk(self, text: str) -> List[Dict]:
        """Analyze a chunk of text for healthcare terms."""
        try:
            prompt = get_analysis_prompt(text)
            content = self._call_gemini(prompt)
            return self._parse_gemini_response(content)
        except Exception as e:
            logger.error(f"Text chunk analysis failed: {e}")
            logger.debug(
                f"Raw response: {content if 'content' in locals() else 'No response'}"
            )
            raise

    def analyze_pdf_file(self, pdf_path: str) -> List[Dict]:
        """Analyze a PDF file for healthcare terms."""
        try:
            text = self.extract_text_from_pdf(pdf_path)
            chunks = self.chunk_text(text)

            all_results = []
            for chunk in chunks:
                results = self.analyze_text_chunk(chunk)
                all_results.extend(results)

            return all_results
        except Exception as e:
            logger.error(f"PDF analysis failed: {e}")
            raise

    def extract_metadata(self, text: str) -> Dict[str, str]:
        """Extract metadata from PDF text using Gemini."""
        logger.info("Extracting metadata using Gemini API")

        # Define metadata fields and their markers
        METADATA_FIELDS = {
            "date": "```date\n",
            "title": "```title\n",
            "organization": "```organization\n",
        }

        # Default metadata values
        metadata = {field: "Unknown" for field in METADATA_FIELDS}

        try:
            # Only use first 2000 characters for metadata extraction
            text_sample = text[:2000]
            model = genai.GenerativeModel("gemini-2.0-flash")
            response = self._model.generate_content(get_metadata_prompt(text_sample))
            content = response.text.strip()

            # Extract each metadata field
            for field, marker in METADATA_FIELDS.items():
                if marker in content:
                    # Find the content between the marker and the next triple backticks
                    start = content.find(marker) + len(marker)
                    end = content.find("```", start)
                    if end != -1:  # Only update if we found the closing backticks
                        metadata[field] = content[start:end].strip()

            return metadata

        except Exception as e:
            logger.error(f"Error extracting metadata: {str(e)}")
            return metadata

    def analyze_pdf_url(self, url: str) -> Dict[str, Any]:
        """Download and analyze a PDF from a URL."""
        try:
            if os.path.exists(url):
                logger.debug(f"Analyzing local file: {url}")
                text = self.extract_text_from_pdf(url)
                mentions = self.analyze_pdf_file(url)
            else:
                cached = self._download_cache.get(url)
                if cached and os.path.exists(cached.get("path", "")):
                    logger.debug("Using cached PDF")
                    temp_path = cached["path"]
                else:
                    logger.debug(f"Downloading: {url}")
                    temp_path = self._download_pdf(url)

                text = self.extract_text_from_pdf(temp_path)
                mentions = self.analyze_pdf_file(temp_path)

            metadata = self.extract_metadata(text)
            return self._build_analysis_result(mentions, metadata)

        except Exception as e:
            logger.error(f"PDF URL analysis failed: {e}")
            return self._build_empty_result()

    def _download_pdf(self, url: str) -> str:
        """Download a PDF and return its local path."""
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(url, headers=headers, stream=True, verify=False)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    tmp.write(chunk)
            temp_path = tmp.name

        self._download_cache[url] = {"path": temp_path}
        return temp_path

    def _build_analysis_result(
        self, mentions: List[Dict], metadata: Dict[str, str]
    ) -> Dict[str, Any]:
        """Build a standardized analysis result dictionary."""
        if not mentions:
            return {
                "success": True,
                "terms_found": [],
                "terms_count": 0,
                "has_relevant_terms": False,
                "terms_data": {},
                "detailed_mentions": [],
                **metadata,
            }

        terms_found = set()
        terms_data = {}
        for mention in mentions:
            term = mention.get("term")
            if term:
                terms_found.add(term)
                if term not in terms_data:
                    terms_data[term] = {
                        "quotes": [mention.get("quotes", "")],
                        "summaries": [mention.get("summary", "")],
                    }
                else:
                    terms_data[term]["quotes"].append(mention.get("quotes", ""))
                    terms_data[term]["summaries"].append(mention.get("summary", ""))

        return {
            "success": True,
            "terms_found": list(terms_found),
            "terms_count": len(terms_found),
            "has_relevant_terms": True,
            "terms_data": terms_data,
            "detailed_mentions": mentions,
            **metadata,
        }

    def _build_empty_result(self) -> Dict[str, Any]:
        """Return an empty result structure for error cases."""
        return {
            "success": False,
            "terms_found": [],
            "terms_count": 0,
            "has_relevant_terms": False,
            "terms_data": {},
            "detailed_mentions": [],
            "date": "Unknown",
            "title": "Unknown",
            "organization": "Unknown",
        }

    def analyze_pdf_directory(
        self, directory_path: str, verbose: bool = True
    ) -> Dict[str, List[Dict]]:
        """Analyze all PDFs in a directory for virtual ward mentions."""
        results = {}
        pdf_files = Path(directory_path).glob("**/*.pdf")

        for pdf_file in pdf_files:
            results[str(pdf_file)] = self.analyze_pdf_file(
                str(pdf_file), verbose=verbose
            )

        return results

    def analyze_pdfs(
        self, urls: List[str], verbose: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple PDFs from a list of URLs.

        Args:
            urls: List of URLs to analyze
            verbose: Whether to print progress messages

        Returns:
            List of analysis results for each URL
        """
        results = []

        for i, url in enumerate(urls, 1):
            if verbose:
                logger.info(f"Analyzing PDF {i} of {len(urls)}: {url}")

            try:
                # Analyze the PDF
                analysis = self.analyze_pdf_url(url, verbose=verbose)

                # Add the URL to the result
                analysis["url"] = url

                results.append(analysis)

            except Exception as e:
                logger.error(f"Error analyzing {url}: {str(e)}")

                results.append(
                    {
                        "url": url,
                        "success": False,
                        "terms_found": [],
                        "terms_count": 0,
                        "has_relevant_terms": False,
                        "terms_data": {},
                        "detailed_mentions": [],
                        "date": "Unknown",
                        "title": "Unknown",
                        "organization": "Unknown",
                        "priorities_summary": "Summary unavailable.",
                    }
                )

        return results

    def extract_date_only(
        self, pdf_path_or_url: str, *, keep_temp_file: bool = False
    ) -> str:
        """Extract only the date from a PDF.

        If *keep_temp_file* is ``True`` and *pdf_path_or_url* is a remote URL, the
        downloaded file is cached so that subsequent full analysis calls do not
        need to download it again.  When ``False`` (default) the temporary file
        is deleted immediately after the date has been extracted.
        """
        logger.info(f"Extracting date only from: {pdf_path_or_url}")

        # Return cached date if we already processed this URL earlier in the run.
        cached = self._download_cache.get(pdf_path_or_url)
        if cached and cached.get("date"):
            logger.info("Using cached date result")
            return cached["date"]

        try:
            # Determine where the PDF lives and fetch if necessary
            if os.path.exists(pdf_path_or_url):
                pdf_path = pdf_path_or_url
            else:
                # If we already downloaded it earlier, reuse it
                if cached and os.path.exists(cached.get("path", "")):
                    pdf_path = cached["path"]
                else:
                    headers = {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                    }
                    response = requests.get(
                        pdf_path_or_url, headers=headers, stream=True, verify=False
                    )
                    response.raise_for_status()

                    with tempfile.NamedTemporaryFile(
                        suffix=".pdf", delete=False
                    ) as tmp:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                tmp.write(chunk)
                        pdf_path = tmp.name

                    # Store in cache so we can reuse the file later if needed
                    self._download_cache[pdf_path_or_url] = {"path": pdf_path}

            # Extract first two pages for date detection
            reader = PdfReader(pdf_path)
            text = ""
            for i in range(min(2, len(reader.pages))):
                text += reader.pages[i].extract_text() + " "

            prompt = get_date_extraction_prompt(text[:3000])

            response = self._date_model.generate_content(prompt)
            date = response.text.strip()

            logger.info(f"Extracted date: {date}")

            # Update cache with date information so future calls can reuse it
            if pdf_path_or_url in self._download_cache:
                self._download_cache[pdf_path_or_url]["date"] = date

            # Clean up file immediately if caller doesn't need it later
            if (not keep_temp_file) and (not os.path.exists(pdf_path_or_url)):
                try:
                    os.unlink(pdf_path)
                    # Remove cached entry because path is now invalid
                    self._download_cache.pop(pdf_path_or_url, None)
                except Exception:
                    pass

            return date

        except Exception as e:
            logger.error(f"Error extracting date: {str(e)}")
            return "Unknown"

    def is_from_2024_or_later(self, date_str: str) -> bool:
        """Check if a paper's date is from 2024 or later"""
        if not date_str or date_str == "Unknown":
            return False

        try:
            # Extract all numbers from the string that could be years (4 digits)
            years = [int(match) for match in re.findall(r"\b\d{4}\b", date_str)]
            # Return True if any year is >= 2024, False otherwise
            return any(year >= 2024 for year in years)
        except:
            return False

    def clean_repetitive_text(self, text: str) -> str:
        """Remove repetitive phrases from text."""
        if not text or len(text) < 20:
            return text

        # Split into sentences
        sentences = re.split(r"(?<=[.!?])\s+", text)
        if len(sentences) <= 1:
            return text

        # Remove exact duplicate sentences that appear consecutively
        cleaned_sentences = []
        for i, sentence in enumerate(sentences):
            # If this sentence is not a duplicate of the previous one, keep it
            if i == 0 or sentence.strip() != sentences[i - 1].strip():
                cleaned_sentences.append(sentence)

        # Join the cleaned sentences
        cleaned_text = " ".join(cleaned_sentences)

        # Also handle repeated phrases (not just sentences)
        # This pattern finds repeated phrases of 10+ characters that appear at least twice
        repeated_phrase_pattern = r"(\b\w{5,}\b.{5,}\b\w{5,}\b)(\s+\1)+"
        cleaned_text = re.sub(repeated_phrase_pattern, r"\1", cleaned_text)

        return cleaned_text

    def _call_gemini(
        self,
        prompt: str,
        *,
        temperature: float = 0.1,
        max_tokens: int = 8_192,
    ) -> str:
        """Single place to call the Gemini model."""
        response = self._model.generate_content(
            prompt,
            generation_config={
                "temperature": temperature,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": max_tokens,
            },
            safety_settings=[
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_ONLY_HIGH",
                }
            ],
        )
        return response.text.strip()


# CLI for standalone use
if __name__ == "__main__":
    data_dir = "data/pdfs"
    analyzer = PDFAnalyzer()
    results = analyzer.analyze_pdf_directory(data_dir)

    # Print results in a readable format
    for filename, mentions in results.items():
        if not mentions:
            logger.info(f"\nNo virtual ward mentions found in: {filename}")
            continue

        logger.info(f"\nFound {len(mentions)} mentions in: {filename}")
        logger.info("-" * 50)

        for i, mention in enumerate(mentions, 1):
            logger.info(f"\nMention {i}:")
            logger.info(f"Quote: {mention['quotes']}")
            logger.info(f"Summary: {mention['summary']}")
            logger.info(f"Term: {mention['term']}")
            logger.info("-" * 30)
