from utils.pdf_analyzer import PDFAnalyzer
import json

def main():
    # Initialize the analyzer with the API key
    api_key = "AIzaSyCcPBK0IZdA2UdXPPF_DJ5ObtQqM30eqUo"
    analyzer = PDFAnalyzer(api_key)
    
    # Test URL
    test_url = "https://www.hct.nhs.uk/download/00-final-trust-board-public-papers-270325pdf.pdf?ver=15043&doc=docm93jijm4n11231.pdf"
    
    print(f"Analyzing PDF from URL: {test_url}")
    
    # Analyze the PDF
    result = analyzer.analyze_pdf_url(test_url, verbose=True)
    
    # Print the result
    print("\nAnalysis Result:")
    print(json.dumps(result, indent=2))
    
    # Extract and print metadata
    if "metadata" in result:
        print("\nMetadata:")
        print(f"Title: {result['metadata'].get('title')}")
        print(f"Organization: {result['metadata'].get('organization')}")
        print(f"Date: {result['metadata'].get('date')}")
    
    # Extract and print healthcare terms found
    if "analysis" in result:
        terms = result["analysis"]
        print(f"\nHealthcare Terms Found: {len(terms)}")
        
        for term in terms:
            print(f"\n--- {term['term']} ---")
            print(f"Summary: {term['summary']}")
            print(f"Mentions ({len(term['mentions'])}):")
            for mention in term['mentions'][:2]: # Print first 2 mentions only to avoid clutter
                print(f"  - \"{mention}\"")
            if len(term['mentions']) > 2:
                print(f"  ... and {len(term['mentions']) - 2} more mentions.")

if __name__ == "__main__":
    main() 