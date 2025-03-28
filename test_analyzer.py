from utils.pdf_analyzer import PDFAnalyzer
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Initialize the analyzer
    analyzer = PDFAnalyzer()
    
    # Test URL - NHS Long Term Plan document (publicly accessible)
    test_url = "https://www.longtermplan.nhs.uk/wp-content/uploads/2019/08/nhs-long-term-plan-version-1.2.pdf"
    
    logger.info("Starting PDF analysis...")
    logger.info(f"Analyzing PDF from: {test_url}")
    
    # Analyze the PDF
    result = analyzer.analyze_pdf(test_url)
    
    # Print results in a formatted way
    logger.info("\nAnalysis Results:")
    print("=" * 80)
    if result["success"]:
        print(json.dumps(result, indent=2))
    else:
        logger.error(f"Error: {result['analysis']}")

if __name__ == "__main__":
    main() 