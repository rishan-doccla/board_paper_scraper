import os
import json
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import tempfile
import logging
import re

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
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("Gemini API key is required")
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        logger.info("PDFAnalyzer initialized with API key")
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text from PDF."""
        # Replace multiple newlines with a single one
        text = ' '.join(text.split())
        # Remove any non-standard whitespace
        text = text.replace('\xa0', ' ')
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        # Fix common OCR issues with medical terms
        text = text.replace('heartfailure', 'heart failure')
        text = text.replace('HeartFailure', 'Heart Failure')
        text = text.replace('HEARTFAILURE', 'HEART FAILURE')
        text = text.replace('virtualward', 'virtual ward')
        text = text.replace('VirtualWard', 'Virtual Ward')
        return text
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text content from a PDF file with improved handling of formatting."""
        logger.info(f"Extracting text from PDF: {pdf_path}")
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    # Clean the text
                    page_text = self.clean_text(page_text)
                    text += page_text + " "
                else:
                    # If extract_text() returns empty (possibly due to images/tables)
                    logger.warning(f"Empty text extracted from page in {pdf_path}")
                    
            if not text.strip():
                logger.warning(f"No text could be extracted from {pdf_path}")
                return "No text could be extracted from this PDF."
                
            logger.info(f"Successfully extracted {len(reader.pages)} pages")
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise
    
    def chunk_text(self, text: str, chunk_size: int = 15000) -> List[str]:
        """Split text into chunks to avoid token limits."""
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            current_size += len(word) + 1  # +1 for space
            if current_size > chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_size = len(word)
            else:
                current_chunk.append(word)
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        # Ensure we have at least one chunk
        if not chunks and text:
            chunks = [text[:min(chunk_size, len(text))]]
        
        logger.info(f"Split text into {len(chunks)} chunks")
        return chunks
    
    def analyze_text_chunk(self, text: str) -> List[Dict]:
        """Analyze a chunk of text for multiple healthcare terms."""
        logger.info("Analyzing text chunk with Gemini API")
        prompt = """

        Objective: Analyze an NHS board paper to identify and summarize mentions of specific healthcare concepts relevant to Doccla's commercial interests.
Role: Act as a specialized analyst AI, meticulously scanning healthcare documents (specifically NHS board papers).
Core Task: Carefully read the provided text from an NHS board paper. Your primary goal is to identify all occurrences of the concepts listed below. Be extremely thorough – check main text, tables, figures, appendices, and footnotes.

Target Concepts & Keywords:
Scan for any mentions related to the following concepts. Include the specific keywords listed, as well as variations, abbreviations, plurals, and closely related phrasings:
Concept: COPD
Keywords/Variations: Chronic Obstructive Pulmonary Disease, COPD, chronic lung disease, lung conditions (in the context of chronic illness).
Concept: Heart Failure
Keywords/Variations: Heart Failure, cardiac failure, CHF, HF, heart conditions, cardiac conditions, heart disease (specifically referring to failure/chronic conditions).
Concept: Long-Term Conditions
Keywords/Variations: Long term conditions, LTCs, chronic conditions, ongoing health needs, long-term illness, chronic disease management.
Concept: Proactive Care
Keywords/Variations: Proactive, preventative, early intervention, upstream work, proactively, anticipatory care.
Concept: Prevention
Keywords/Variations: Prevention, preventative care, preventing illness, risk reduction, health promotion (in the context of preventing specific conditions or exacerbations).
Concept: Neighbourhood/Place-Based Care
Keywords/Variations: Neighbourhood, place-based care, community care, local care, locality teams, integrated neighbourhood teams, care closer to home (as a location/model).
Concept: Care Shift (Left Shift)
Keywords/Variations: Shift left, demand shift, upstream, moving care, care pathway redesign (in context of shifting from acute), reducing hospital admissions, community shift.
Concept: Identifying/Managing Patients at Risk
Keywords/Variations: Rising risk, patients at risk, high risk cohorts, risk stratification, identifying risk, vulnerable populations, at-risk patients.
Concept: Virtual Wards / Remote Monitoring
Keywords/Variations: Virtual wards, virtual care, virtual hospital, remote monitoring, telehealth (in context of ward-level care), hospital at home (tech-enabled), remote patient management.

Output Format:

For each Concept listed above where you find at least one mention in the text, respond using this EXACT format:

```term
CONCEPT_NAME (e.g., Heart Failure)
```
```mentions
"Exact quote 1... [include enough surrounding text for context]." [Reference if possible, e.g., "Page 5"]
"Exact quote 2... [if multiple distinct mentions, list relevant ones]." [Reference if possible, e.g., "Table 3, Page 10"]
[Add more quotes as necessary]
```
```summary
**Headline 1:** [A concise sentence summarizing the first key status, development, challenge, or strategic point identified from the mentions.]
**Summary 1:**
    *   **Detailed Context:** [Elaborate on **Headline 1** with 3-5 sentences, providing supporting details *specifically related to this headline* from the text. Include relevant data points, initiatives, challenges, strategic importance, or decisions mentioned.]
    *   **Commercial Angle (Doccla Opportunity):** [Based *only* on the information related to **Headline 1**, suggest a potential actionable insight or opportunity for Doccla's commercial team in 1-3 sentences. How could Doccla leverage the specific information related to this headline for relevant outreach or positioning?]

**Headline 2:** [A concise sentence summarizing the second distinct key status, development, challenge, or strategic point identified from the mentions, *if applicable and distinct from Headline 1*.]
**Summary 2:**
    *   **Detailed Context:** [Elaborate on **Headline 2** with 3-5 sentences, providing supporting details *specifically related to this headline* from the text.]
    *   **Commercial Angle (Doccla Opportunity):** [Based *only* on the information related to **Headline 2**, suggest a potential actionable insight or opportunity for Doccla's commercial team in 1-3 sentences.]

**Headline 3:** [A concise sentence summarizing the third distinct key status, development, challenge, or strategic point identified from the mentions, *if applicable and distinct from Headlines 1 & 2*.]
**Summary 3:**
    *   **Detailed Context:** [Elaborate on **Headline 3** with 3-5 sentences, providing supporting details *specifically related to this headline* from the text.]
    *   **Commercial Angle (Doccla Opportunity):** [Based *only* on the information related to **Headline 3**, suggest a potential actionable insight or opportunity for Doccla's commercial team in 1-3 sentences.]

[Only include Headline/Summary blocks for the distinct points identified, up to a maximum of three.]
```

BE EXTREMELY THOROUGH. Include terms even if they are only mentioned briefly or in passing. Don't miss any terms. Read tables carefully. If you see 'HF' or 'CHF', that means Heart Failure. If you see 'COPD', that refers to Chronic Obstructive Pulmonary Disease.
IMPORTANT:
Headline Findings: Extract up to three distinct main points for the first part of the summary if the text supports it.
Summary Structure: Strictly follow the 3-point structure (Headline Findings, Detailed Context, Commercial Angle) for every summary.
Summary Content: Summaries should synthesize the information from the mentions, supporting the headline findings with relevant details. Aim for a slightly more comprehensive summary than before, but still focused.
Commercial Angle: This MUST be directly linked to the specific information presented in the document for that concept (as highlighted in the headlines and context). Avoid generic statements.
Repetition: Avoid repeating the exact same information across different concept summaries.
Stick to the Text: Extract information only from the provided text. Do not infer external knowledge or make assumptions.
No Mentions: If none of the target concepts are found anywhere in the document, respond ONLY with: NO_RELEVANT_MENTIONS_FOUND
One Entry Per Concept: Only provide one output block (term, mentions, summary) per target concept, even if mentioned multiple times. Consolidate all findings for that concept into its single entry.

Text to analyze:
{text}"""
        
        try:
            # Use the original model
            model = genai.GenerativeModel('gemini-2.0-flash')
            
            # Set generation config with longer timeout for complex documents
            generation_config = {
                "temperature": 0.0,  # Lower temperature for more focused extraction
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192,  # Allow longer responses
            }
            
            # Make API call with timeout handling
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_ONLY_HIGH"
                }
            ]
            
            response = model.generate_content(
                prompt.format(text=text),
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            content = response.text.strip()
            
            # If no mentions found
            if content == "NO_RELEVANT_MENTIONS_FOUND":
                logger.info("No relevant term mentions found in chunk")
                return []
                
            # Parse the response format
            mentions = []
            sections = content.split('---')
            
            for section in sections:
                if not section.strip():
                    continue
                    
                mention = {}
                
                # Extract term
                term_start = section.find('```term\n') + 8
                term_end = section.find('```', term_start)
                if term_start > 7 and term_end != -1:
                    mention['term'] = section[term_start:term_end].strip()
                
                # Extract mentions/quotes
                mentions_start = section.find('```mentions\n') + 12
                mentions_end = section.find('```', mentions_start)
                if mentions_start > 11 and mentions_end != -1:
                    quotes = section[mentions_start:mentions_end].strip()
                    # Clean up any repetitive text in quotes
                    mention['quotes'] = self.clean_repetitive_text(quotes)
                
                # Extract summary
                summary_start = section.find('```summary\n') + 11
                summary_end = section.find('```', summary_start)
                if summary_start > 10 and summary_end != -1:
                    summary = section[summary_start:summary_end].strip()
                    # Clean up any repetitive text in summary
                    mention['summary'] = self.clean_repetitive_text(summary)
                
                # Check if this mentions NO_RELEVANT_MENTIONS_FOUND at the end
                if mention.get('summary') and ('NO_RELEVANT_MENTIONS_FOUND' in mention['summary'] or 
                                              'NO MENTIONS FOUND' in mention['summary'].upper()):
                    # Remove the "no mentions found" part from the summary
                    clean_summary = re.sub(r'NO_RELEVANT_MENTIONS_FOUND.*$', '', mention['summary'], 
                                           flags=re.IGNORECASE | re.DOTALL)
                    clean_summary = re.sub(r'NO MENTIONS FOUND.*$', '', clean_summary, 
                                           flags=re.IGNORECASE | re.DOTALL)
                    mention['summary'] = clean_summary.strip()
                
                if len(mention) == 3:  # Only add if we found all three parts
                    # Only include if we still have valid content after cleaning
                    if mention['summary'].strip() and mention['quotes'].strip():
                        mentions.append(mention)
            
            logger.info(f"Found {len(mentions)} term mentions in chunk")
            return mentions
            
        except Exception as e:
            logger.error(f"Error analyzing text chunk: {str(e)}")
            logger.error(f"Raw response:\n{content if 'content' in locals() else 'No response'}")
            raise
    
    def analyze_pdf_file(self, pdf_path: str, verbose: bool = True) -> List[Dict]:
        """Analyze a PDF file for virtual ward mentions."""
        if verbose:
            logger.info(f"\nAnalyzing: {pdf_path}")
        
        try:
            # Extract text from PDF
            text = self.extract_text_from_pdf(pdf_path)
            
            # Split into chunks to handle large documents
            chunks = self.chunk_text(text)
            
            # Analyze each chunk
            all_results = []
            for i, chunk in enumerate(chunks, 1):
                if verbose:
                    logger.info(f"Processing chunk {i} of {len(chunks)}...")
                results = self.analyze_text_chunk(chunk)
                all_results.extend(results)
            
            return all_results
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {str(e)}")
            raise
    
    def extract_metadata(self, text: str) -> Dict[str, str]:
        """Extract metadata from PDF text using Gemini."""
        logger.info("Extracting metadata using Gemini API")
        prompt = """Extract the following metadata from this document text:
        1. Document Date: Find the most likely meeting or document date (in YYYY-MM-DD format if possible)
        2. Document Title: Extract the main title of the document
        3. Organization Name: Identify the NHS organization or trust name
        
        Respond in this EXACT format (including the triple backticks):
        ```date
        YYYY-MM-DD or any found date format
        ```
        ```title
        DOCUMENT TITLE
        ```
        ```organization
        ORGANIZATION NAME
        ```
        
        If any field cannot be found, use "Unknown" as the value.
        
        Text to analyze (first 2000 characters):
        {text}"""
        
        try:
            # Only use first 2000 characters for metadata extraction
            text_sample = text[:2000]
            model = genai.GenerativeModel('gemini-2.0-flash')
            response = model.generate_content(prompt.format(text=text_sample))
            content = response.text.strip()
            
            metadata = {
                'date': 'Unknown',
                'title': 'Unknown',
                'organization': 'Unknown'
            }
            
            # Parse the response
            if '```date' in content:
                date_start = content.find('```date\n') + 8
                date_end = content.find('```', date_start)
                metadata['date'] = content[date_start:date_end].strip()
                
            if '```title' in content:
                title_start = content.find('```title\n') + 9
                title_end = content.find('```', title_start)
                metadata['title'] = content[title_start:title_end].strip()
                
            if '```organization' in content:
                org_start = content.find('```organization\n') + 15
                org_end = content.find('```', org_start)
                metadata['organization'] = content[org_start:org_end].strip()
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting metadata: {str(e)}")
            return {
                'date': 'Unknown',
                'title': 'Unknown',
                'organization': 'Unknown'
            }

    def analyze_pdf_url(self, url: str, verbose: bool = True) -> Dict[str, Any]:
        """
        Download and analyze a PDF from a URL.
        
        Args:
            url: URL of the PDF to analyze or path to local PDF file
            verbose: Whether to print progress messages
            
        Returns:
            Dictionary with analysis results and metadata
        """
        try:
            # Check if this is a local file
            if os.path.exists(url):
                logger.info(f"Analyzing local PDF file: {url}")
                text = self.extract_text_from_pdf(url)
                mentions = self.analyze_pdf_file(url, verbose=verbose)
            else:
                # Handle URL
                if verbose:
                    logger.info(f"Downloading PDF from {url}")
                
                # Download the PDF with proper headers
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                response = requests.get(url, headers=headers, stream=True, verify=False)
                response.raise_for_status()
                
                # Create a temporary file
                with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                    temp_path = temp_file.name
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            temp_file.write(chunk)
                            
                # Extract text and analyze the PDF
                text = self.extract_text_from_pdf(temp_path)
                mentions = self.analyze_pdf_file(temp_path, verbose=verbose)
                
                # Clean up the temporary file
                os.unlink(temp_path)
            
            # Extract metadata
            metadata = self.extract_metadata(text)
            
            # Process found terms
            if mentions:
                # Group mentions by term
                terms_found = set()
                terms_data = {}
                for mention in mentions:
                    term = mention.get('term')
                    if term:
                        terms_found.add(term)
                        if term not in terms_data:
                            terms_data[term] = {
                                'quotes': [mention.get('quotes', '')],
                                'summaries': [mention.get('summary', '')]
                            }
                        else:
                            terms_data[term]['quotes'].append(mention.get('quotes', ''))
                            terms_data[term]['summaries'].append(mention.get('summary', ''))
                
                result = {
                    "success": True,
                    "terms_found": list(terms_found),
                    "terms_count": len(terms_found),
                    "has_relevant_terms": len(terms_found) > 0,
                    "terms_data": terms_data,
                    "detailed_mentions": mentions,
                    "date": metadata['date'],
                    "title": metadata['title'],
                    "organization": metadata['organization']
                }
                logger.info(f"Found {len(terms_found)} relevant terms")
            else:
                result = {
                    "success": True,
                    "terms_found": [],
                    "terms_count": 0,
                    "has_relevant_terms": False,
                    "terms_data": {},
                    "detailed_mentions": [],
                    "date": metadata['date'],
                    "title": metadata['title'],
                    "organization": metadata['organization']
                }
                logger.info("No relevant terms found")
                
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing PDF: {str(e)}")
            return {
                "success": False,
                "terms_found": [],
                "terms_count": 0,
                "has_relevant_terms": False,
                "terms_data": {},
                "detailed_mentions": [],
                "date": "Unknown",
                "title": "Unknown",
                "organization": "Unknown"
            }
    
    def analyze_pdf_directory(self, directory_path: str, verbose: bool = True) -> Dict[str, List[Dict]]:
        """Analyze all PDFs in a directory for virtual ward mentions."""
        results = {}
        pdf_files = Path(directory_path).glob('**/*.pdf')
        
        for pdf_file in pdf_files:
            results[str(pdf_file)] = self.analyze_pdf_file(str(pdf_file), verbose=verbose)
        
        return results

    def analyze_pdfs(self, urls: List[str], verbose: bool = True) -> List[Dict[str, Any]]:
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
                    
                results.append({
                    "url": url,
                    "success": False,
                    "terms_found": [],
                    "terms_count": 0,
                    "has_relevant_terms": False,
                    "terms_data": {},
                    "detailed_mentions": [],
                    "date": "Unknown",
                    "title": "Unknown",
                    "organization": "Unknown"
                })
        
        return results

    def extract_date_only(self, pdf_path_or_url: str) -> str:
        """Extract only the date from a PDF to determine if it's from 2024 or later.
        
        Args:
            pdf_path_or_url: Path to PDF file or URL
            
        Returns:
            The extracted date as a string or "Unknown"
        """
        logger.info(f"Extracting date only from: {pdf_path_or_url}")
        try:
            # Handle URL vs local file
            if not os.path.exists(pdf_path_or_url):
                # Download the PDF with proper headers
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                response = requests.get(pdf_path_or_url, headers=headers, stream=True, verify=False)
                response.raise_for_status()
                
                # Create a temporary file
                with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                    temp_path = temp_file.name
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            temp_file.write(chunk)
                pdf_path = temp_path
            else:
                pdf_path = pdf_path_or_url
            
            # Extract only the first 2 pages
            reader = PdfReader(pdf_path)
            text = ""
            # Only read first 2 pages
            for i in range(min(2, len(reader.pages))):
                text += reader.pages[i].extract_text() + " "
                
            # Clean up if temp file
            if not os.path.exists(pdf_path_or_url):
                os.unlink(pdf_path)
                
            # Extract date with a focused prompt
            prompt = """Extract ONLY the document date from this text.
            Look for meeting dates, publication dates, or any dates that appear to be when the document was created.
            
            Return the date in YYYY-MM-DD format if possible, or any clear date format you find.
            If multiple dates are found, choose the one most likely to be the document date.
            
            Respond with ONLY the date and nothing else. If no date is found, respond with "Unknown".
            
            Text:
            {text}"""
            
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt.format(text=text[:3000]))  # First 3000 chars should be enough
            date = response.text.strip()
            
            logger.info(f"Extracted date: {date}")
            return date
            
        except Exception as e:
            logger.error(f"Error extracting date: {str(e)}")
            return "Unknown"
            
    def is_from_2024_or_later(self, date_str: str) -> bool:
        """Check if a paper's date is from 2024 or later"""
        if date_str == "Unknown" or not date_str:
            return False
        
        # Try to extract year from various date formats
        try:
            # Handle ISO format (YYYY-MM-DD)
            if '-' in date_str and len(date_str) >= 4:
                year = int(date_str.split('-')[0])
                return year >= 2024
            
            # Handle other formats that have year at the beginning
            elif len(date_str) >= 4 and date_str[:4].isdigit():
                year = int(date_str[:4])
                return year >= 2024
                
            # Handle formats with year at the end (e.g., "Jan 2024" or "January 2024")
            elif " " in date_str and date_str.split()[-1].isdigit():
                year = int(date_str.split()[-1])
                return year >= 2024
                
            # Default to false if we can't determine the date
            else:
                return False
        except:
            return False

    def clean_repetitive_text(self, text: str) -> str:
        """Remove repetitive phrases from text."""
        if not text or len(text) < 20:
            return text
            
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) <= 1:
            return text
            
        # Remove exact duplicate sentences that appear consecutively
        cleaned_sentences = []
        for i, sentence in enumerate(sentences):
            # If this sentence is not a duplicate of the previous one, keep it
            if i == 0 or sentence.strip() != sentences[i-1].strip():
                cleaned_sentences.append(sentence)
                
        # Join the cleaned sentences
        cleaned_text = ' '.join(cleaned_sentences)
        
        # Also handle repeated phrases (not just sentences)
        # This pattern finds repeated phrases of 10+ characters that appear at least twice
        repeated_phrase_pattern = r'(\b\w{5,}\b.{5,}\b\w{5,}\b)(\s+\1)+'
        cleaned_text = re.sub(repeated_phrase_pattern, r'\1', cleaned_text)
        
        return cleaned_text

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