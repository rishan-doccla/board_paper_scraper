import os
import json
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import tempfile

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
        api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("Gemini API key is required")
        
        # Configure Gemini
        genai.configure(api_key=api_key)
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text from PDF."""
        # Replace multiple newlines with a single one
        text = ' '.join(text.split())
        # Remove any non-standard whitespace
        text = text.replace('\xa0', ' ')
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        return text
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text content from a PDF file."""
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += self.clean_text(page.extract_text()) + " "
        return text.strip()
    
    def chunk_text(self, text: str, chunk_size: int = 10000) -> List[str]:
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
        
        return chunks
    
    def analyze_text_chunk(self, text: str) -> List[Dict]:
        """Analyze a chunk of text for virtual ward mentions."""
        prompt = """Analyze this text for mentions of 'virtual ward' or related concepts.
    
    For each mention found, respond in this EXACT format (including the triple backticks):
    ```mention
    EXACT QUOTE CONTAINING VIRTUAL WARD
    ```
    ```summary
    BRIEF SUMMARY OF WHAT IS BEING DISCUSSED
    ```
    ```context
    SURROUNDING CONTEXT
    ```
    ---
    
    If no mentions are found, respond with just: NO_MENTIONS_FOUND
    
    Text to analyze:
    {text}"""
        
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt.format(text=text))
        content = response.text.strip()
        
        # If no mentions found
        if content == "NO_MENTIONS_FOUND":
            return []
            
        try:
            # Parse the response format
            mentions = []
            sections = content.split('---')
            
            for section in sections:
                if not section.strip():
                    continue
                    
                mention = {}
                
                # Extract quote
                start = section.find('```mention\n') + 10
                end = section.find('```', start)
                if start > 9 and end != -1:
                    mention['mention'] = section[start:end].strip()
                
                # Extract summary
                start = section.find('```summary\n') + 10
                end = section.find('```', start)
                if start > 9 and end != -1:
                    mention['summary'] = section[start:end].strip()
                
                # Extract context
                start = section.find('```context\n') + 10
                end = section.find('```', start)
                if start > 9 and end != -1:
                    mention['page_context'] = section[start:end].strip()
                
                if len(mention) == 3:  # Only add if we found all three parts
                    mentions.append(mention)
            
            return mentions
        except Exception as e:
            print(f"Error parsing response: {str(e)}")
            print(f"Raw response:\n{content}")
            return []
    
    def analyze_pdf_file(self, pdf_path: str, verbose: bool = True) -> List[Dict]:
        """Analyze a PDF file for virtual ward mentions."""
        if verbose:
            print(f"\nAnalyzing: {pdf_path}")
            print("-" * 50)
        
        try:
            # Extract text from PDF
            text = self.extract_text_from_pdf(pdf_path)
            
            # Split into chunks to handle large documents
            chunks = self.chunk_text(text)
            
            # Analyze each chunk
            all_results = []
            for i, chunk in enumerate(chunks, 1):
                if verbose:
                    print(f"Processing chunk {i} of {len(chunks)}...")
                results = self.analyze_text_chunk(chunk)
                all_results.extend(results)
            
            return all_results
        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")
            return []
    
    def analyze_pdf_url(self, url: str, verbose: bool = True) -> Dict[str, Any]:
        """
        Download and analyze a PDF from a URL.
        
        Args:
            url: URL of the PDF to analyze
            verbose: Whether to print progress messages
            
        Returns:
            Dictionary with analysis results
        """
        try:
            if verbose:
                print(f"Downloading PDF from {url}")
                
            # Download the PDF
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Create a temporary file
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                temp_path = temp_file.name
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        temp_file.write(chunk)
                        
            # Analyze the PDF
            mentions = self.analyze_pdf_file(temp_path, verbose=verbose)
            
            # Create a summary if mentions were found
            if mentions:
                # Extract and join all summaries
                summaries = [m["summary"] for m in mentions]
                summary = " ".join(summaries)
                
                result = {
                    "success": True,
                    "virtual_ward_mentioned": True,
                    "mentions_count": len(mentions),
                    "summary": summary,
                    "detailed_mentions": mentions
                }
            else:
                result = {
                    "success": True,
                    "virtual_ward_mentioned": False,
                    "mentions_count": 0,
                    "summary": "No mentions of virtual wards found in this document.",
                    "detailed_mentions": []
                }
                
            # Clean up the temporary file
            os.unlink(temp_path)
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "virtual_ward_mentioned": False,
                "mentions_count": 0,
                "summary": f"Error analyzing PDF: {str(e)}",
                "detailed_mentions": []
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
                print(f"Analyzing PDF {i} of {len(urls)}: {url}")
                
            try:
                # Analyze the PDF
                analysis = self.analyze_pdf_url(url, verbose=verbose)
                
                # Add the URL to the result
                analysis["url"] = url
                
                results.append(analysis)
                
            except Exception as e:
                if verbose:
                    print(f"Error analyzing {url}: {str(e)}")
                    
                results.append({
                    "url": url,
                    "success": False,
                    "virtual_ward_mentioned": False,
                    "mentions_count": 0,
                    "summary": f"Error analyzing PDF: {str(e)}",
                    "detailed_mentions": []
                })
        
        return results

# CLI for standalone use
if __name__ == "__main__":
    data_dir = "data/pdfs"
    analyzer = PDFAnalyzer()
    results = analyzer.analyze_pdf_directory(data_dir)
    
    # Print results in a readable format
    for filename, mentions in results.items():
        if not mentions:
            print(f"\nNo virtual ward mentions found in: {filename}")
            continue
            
        print(f"\nFound {len(mentions)} mentions in: {filename}")
        print("-" * 50)
        
        for i, mention in enumerate(mentions, 1):
            print(f"\nMention {i}:")
            print(f"Quote: {mention['mention']}")
            print(f"Summary: {mention['summary']}")
            print(f"Context: {mention['page_context']}")
            print("-" * 30) 