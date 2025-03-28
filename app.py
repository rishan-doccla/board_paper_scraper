from flask import Flask, render_template, request, jsonify
from apscheduler.schedulers.background import BackgroundScheduler
import os
import datetime
import json
import asyncio
from crawler.nhs_crawler import NHSCrawler
from utils.config import Config
import re
from utils.pdf_analyzer import PDFAnalyzer

app = Flask(__name__)
app.config['SECRET_KEY'] = Config.SECRET_KEY

# Initialize the crawler
crawler = NHSCrawler()

# Create a scheduler for periodic scraping
scheduler = BackgroundScheduler()

# Store the latest scraping results
latest_results = {
    "last_run": None,
    "board_papers": []
}

# Data directory for storing results
DATA_DIR = Config.DATA_DIR
os.makedirs(DATA_DIR, exist_ok=True)
RESULTS_FILE = os.path.join(DATA_DIR, "board_papers.json")

# Add Gemini API key
GEMINI_API_KEY = "AIzaSyCcPBK0IZdA2UdXPPF_DJ5ObtQqM30eqUo"

# Initialize PDF analyzer
pdf_analyzer = PDFAnalyzer(GEMINI_API_KEY)

def load_existing_data():
    """Load existing board paper data if available"""
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            return json.load(f)
    return {"last_run": None, "board_papers": []}

def save_results(results):
    """Save scraping results to file"""
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)

def run_crawler():
    """Run the NHS website crawler to find board papers"""
    global latest_results
    
    # Load existing data
    existing_data = load_existing_data()
    existing_papers = {(paper["url"], paper["title"]) for paper in existing_data.get("board_papers", [])}
    
    # Define a more comprehensive list of NHS organization URLs
    test_urls = [
        "https://www.hct.nhs.uk/",
        "https://bedfordshirelutonandmiltonkeynes.icb.nhs.uk/",
        "https://www.bedfordshirehospitals.nhs.uk/",
        "https://www.mkuh.nhs.uk/",
        "https://www.hertsandwestessex.ics.nhs.uk/",
        "https://www.enherts-tr.nhs.uk/",
        "https://www.hpft.nhs.uk/",
        "https://www.pah.nhs.uk/",
        "https://www.westhertshospitals.nhs.uk/",
        "https://www.cpics.org.uk/",
        "https://www.nwangliaft.nhs.uk/",
        "https://www.royalpapworth.nhs.uk/",
        "https://www.cambscommunityservices.nhs.uk/",
        "https://www.midandsouthessex.ics.nhs.uk/",
        "https://eput.nhs.uk/",
        "https://www.mse.nhs.uk/",
        "https://improvinglivesnw.org.uk/",
        "https://www.norfolkcommunityhealthandcare.nhs.uk/",
        "https://www.nnuh.nhs.uk/"
    ]
    
    # Run the crawler
    print(f"Starting crawler at {datetime.datetime.now()}")
    print(f"Crawling {len(test_urls)} NHS websites")
    
    # Use the async method to run the crawler for the specific URLs
    papers = asyncio.run(crawler.find_board_papers_for_urls(test_urls))
    
    # Filter for board papers from 2024 onwards
    filtered_papers = []
    for paper in papers:
        # Check if it's a board paper by title or URL
        is_board_paper = any(term in paper.get("title", "").lower() or term in paper.get("url", "").lower() 
                            for term in ["board", "meeting", "minutes", "papers", "agenda"])
        
        # Check if it's from 2024 or newer
        has_valid_year = False
        paper_date = paper.get("date", "")
        paper_title = paper.get("title", "")
        paper_url = paper.get("url", "")
        
        # Check for year in date, title or URL
        if "2024" in paper_date or "2025" in paper_date:
            has_valid_year = True
        elif "2024" in paper_title or "2025" in paper_title:
            has_valid_year = True
        elif "2024" in paper_url or "2025" in paper_url:
            has_valid_year = True
            
        if is_board_paper and has_valid_year:
            # Extract month and year for sorting
            month_names = ["january", "february", "march", "april", "may", "june", 
                          "july", "august", "september", "october", "november", "december",
                          "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
            
            # Default sort value (end of list)
            paper["sort_date"] = "9999-99"
            
            # Try to extract year and month
            year = None
            month = None
            
            # Look for year first
            year_match = re.search(r'20(24|25)', paper_date + " " + paper_title + " " + paper_url)
            if year_match:
                year = "20" + year_match.group(1)
                
                # Then look for month
                for i, month_name in enumerate(month_names):
                    month_i = i % 12 + 1  # Convert to 1-12
                    if month_name in (paper_date + " " + paper_title).lower():
                        month = str(month_i).zfill(2)  # pad with zero
                        break
                
                # If we found both year and month
                if year and month:
                    paper["sort_date"] = f"{year}-{month}"
                elif year:
                    paper["sort_date"] = f"{year}-00"
            
            # Initialize virtual ward fields without analyzing yet
            paper["virtual_ward_mentioned"] = False
            paper["virtual_ward_summary"] = "Not analyzed"
            paper["virtual_ward_mentions_count"] = 0
            
            filtered_papers.append(paper)
    
    # Mark new papers
    for paper in filtered_papers:
        paper["is_new"] = (paper["url"], paper["title"]) not in existing_papers
        paper["found_date"] = datetime.datetime.now().isoformat()
    
    # Sort papers by date (most recent first but from 2024 onwards)
    filtered_papers.sort(key=lambda x: x.get("sort_date", "9999-99"))
    
    print(f"Scraping finished at {datetime.datetime.now()}, found {len(filtered_papers)} board papers from 2024 onwards")
    
    # STAGE 2: Analyze the PDFs for virtual ward mentions
    print(f"Starting PDF analysis for virtual ward mentions...")
    
    pdf_papers = [p for p in filtered_papers if p["url"].lower().endswith('.pdf') or "pdf" in p["url"].lower()]
    print(f"Found {len(pdf_papers)} PDF papers to analyze")
    
    # Initialize the analyzer
    analyzer = PDFAnalyzer(GEMINI_API_KEY)
    
    # Analyze each PDF paper
    for i, paper in enumerate(pdf_papers, 1):
        try:
            print(f"Analyzing PDF {i} of {len(pdf_papers)}: {paper['title']}")
            
            # Analyze the PDF
            results = analyzer.analyze_pdf_url(paper["url"], verbose=False)
            
            # Find the paper in filtered_papers and update it
            for p in filtered_papers:
                if p["url"] == paper["url"] and p["title"] == paper["title"]:
                    p["virtual_ward_mentioned"] = results["virtual_ward_mentioned"]
                    p["virtual_ward_summary"] = results["summary"]
                    p["virtual_ward_mentions_count"] = results["mentions_count"]
                    print(f"  - Virtual ward mentioned: {results['virtual_ward_mentioned']}")
                    break
                    
        except Exception as e:
            print(f"Error analyzing PDF {paper['url']}: {str(e)}")
            # Find the paper in filtered_papers and update it with the error
            for p in filtered_papers:
                if p["url"] == paper["url"] and p["title"] == paper["title"]:
                    p["virtual_ward_summary"] = f"Error during analysis: {str(e)}"
                    break
    
    print(f"PDF analysis completed at {datetime.datetime.now()}")
    
    # Update latest results
    latest_results = {
        "last_run": datetime.datetime.now().isoformat(),
        "board_papers": filtered_papers
    }
    
    # Save results
    save_results(latest_results)
    print(f"Crawler and analysis completed at {datetime.datetime.now()}")

async def run_crawler_for_specific_urls(urls):
    """Run the NHS website crawler for specific URLs"""
    # Load existing data
    existing_data = load_existing_data()
    existing_papers = {(paper["url"], paper["title"]) for paper in existing_data.get("board_papers", [])}
    
    # Log the URLs we're about to process
    print(f"Processing {len(urls)} URLs:")
    for url in urls:
        print(f"  - {url}")
    
    # Run the crawler for specific URLs
    print(f"Starting crawler for specific URLs at {datetime.datetime.now()}")
    papers = await crawler.find_board_papers_for_urls(urls)
    
    # Filter for board papers from 2024 onwards
    filtered_papers = []
    for paper in papers:
        # Check if it's a board paper by title or URL
        is_board_paper = any(term in paper.get("title", "").lower() or term in paper.get("url", "").lower() 
                            for term in ["board", "meeting", "minutes", "papers", "agenda"])
        
        # Check if it's from 2024 or newer
        has_valid_year = False
        paper_date = paper.get("date", "")
        paper_title = paper.get("title", "")
        paper_url = paper.get("url", "")
        
        # Check for year in date, title or URL
        if "2024" in paper_date or "2025" in paper_date:
            has_valid_year = True
        elif "2024" in paper_title or "2025" in paper_title:
            has_valid_year = True
        elif "2024" in paper_url or "2025" in paper_url:
            has_valid_year = True
            
        if is_board_paper and has_valid_year:
            # Extract month and year for sorting
            month_names = ["january", "february", "march", "april", "may", "june", 
                          "july", "august", "september", "october", "november", "december",
                          "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
            
            # Default sort value (end of list)
            paper["sort_date"] = "9999-99"
            
            # Try to extract year and month
            year = None
            month = None
            
            # Look for year first
            year_match = re.search(r'20(24|25)', paper_date + " " + paper_title + " " + paper_url)
            if year_match:
                year = "20" + year_match.group(1)
                
                # Then look for month
                for i, month_name in enumerate(month_names):
                    month_i = i % 12 + 1  # Convert to 1-12
                    if month_name in (paper_date + " " + paper_title).lower():
                        month = str(month_i).zfill(2)  # pad with zero
                        break
                
                # If we found both year and month
                if year and month:
                    paper["sort_date"] = f"{year}-{month}"
                elif year:
                    paper["sort_date"] = f"{year}-00"
            
            # Initialize virtual ward fields
            paper["virtual_ward_mentioned"] = False
            paper["virtual_ward_summary"] = "Not analyzed"
            paper["virtual_ward_mentions_count"] = 0
            
            filtered_papers.append(paper)
    
    # Mark new papers
    for paper in filtered_papers:
        paper["is_new"] = (paper["url"], paper["title"]) not in existing_papers
        paper["found_date"] = datetime.datetime.now().isoformat()
    
    # STAGE 2: Analyze the PDFs for virtual ward mentions
    print(f"Starting PDF analysis for virtual ward mentions...")
    
    pdf_papers = [p for p in filtered_papers if p["url"].lower().endswith('.pdf') or "pdf" in p["url"].lower()]
    print(f"Found {len(pdf_papers)} PDF papers to analyze")
    
    # Initialize the analyzer
    analyzer = PDFAnalyzer(GEMINI_API_KEY)
    
    # Analyze each PDF paper
    for i, paper in enumerate(pdf_papers, 1):
        try:
            print(f"Analyzing PDF {i} of {len(pdf_papers)}: {paper['title']}")
            
            # Analyze the PDF
            results = analyzer.analyze_pdf_url(paper["url"], verbose=False)
            
            # Find the paper in filtered_papers and update it
            for p in filtered_papers:
                if p["url"] == paper["url"] and p["title"] == paper["title"]:
                    p["virtual_ward_mentioned"] = results["virtual_ward_mentioned"]
                    p["virtual_ward_summary"] = results["summary"]
                    p["virtual_ward_mentions_count"] = results["mentions_count"]
                    print(f"  - Virtual ward mentioned: {results['virtual_ward_mentioned']}")
                    break
                    
        except Exception as e:
            print(f"Error analyzing PDF {paper['url']}: {str(e)}")
            # Find the paper in filtered_papers and update it with the error
            for p in filtered_papers:
                if p["url"] == paper["url"] and p["title"] == paper["title"]:
                    p["virtual_ward_summary"] = f"Error during analysis: {str(e)}"
                    break
    
    print(f"PDF analysis completed at {datetime.datetime.now()}")
    
    return filtered_papers

@app.route('/')
def index():
    """Homepage route"""
    return render_template('index.html', results=latest_results)

@app.route('/run-crawler', methods=['POST'])
def trigger_crawler():
    """Route to manually trigger the crawler"""
    run_crawler()
    return jsonify({"status": "success", "message": "Crawler completed", "results": latest_results})

@app.route('/test-specific-urls', methods=['POST'])
def test_specific_urls():
    """Route to test specific URLs"""
    data = request.get_json()
    urls = data.get('urls', [])
    
    if not urls:
        return jsonify({"status": "error", "message": "No URLs provided"})
    
    # Run the crawler for specific URLs
    papers = asyncio.run(run_crawler_for_specific_urls(urls))
    
    # Create results object
    results = {
        "last_run": datetime.datetime.now().isoformat(),
        "board_papers": papers
    }
    
    # Update latest results
    global latest_results
    latest_results = results
    
    # Save results
    save_results(results)
    
    return jsonify({
        "status": "success", 
        "message": f"Crawler completed, found {len(papers)} papers", 
        "results": results
    })

@app.route('/results')
def view_results():
    """View the latest results"""
    return jsonify(latest_results)

@app.route('/analyze-papers', methods=['POST'])
def analyze_papers():
    """Analyze board papers for specific topics using Gemini"""
    try:
        data = request.get_json()
        urls = data.get('urls', [])
        
        if not urls:
            return jsonify({
                "error": "No URLs provided",
                "success": False
            }), 400
        
        # Get the current board papers data
        global latest_results
        current_papers = {paper["url"]: paper for paper in latest_results.get("board_papers", [])}
        
        # Initialize the analyzer
        analyzer = PDFAnalyzer(GEMINI_API_KEY)
        
        # Prepare results array
        results = []
        
        # Analyze each URL
        for url in urls:
            # Check if we already have analysis for this URL
            if url in current_papers and current_papers[url].get("virtual_ward_summary") != "Not analyzed":
                # Use existing analysis
                paper = current_papers[url]
                results.append({
                    "url": url,
                    "title": paper.get("title", "Unknown"),
                    "virtual_ward_mentioned": paper.get("virtual_ward_mentioned", False),
                    "summary": paper.get("virtual_ward_summary", "No analysis available"),
                    "mentions_count": paper.get("virtual_ward_mentions_count", 0),
                    "analysis_source": "cached"
                })
                print(f"Using cached analysis for {url}")
            else:
                # Perform new analysis
                try:
                    print(f"Analyzing {url} for virtual ward mentions")
                    analysis = analyzer.analyze_pdf_url(url)
                    
                    result = {
                        "url": url,
                        "title": current_papers.get(url, {}).get("title", "Unknown"),
                        "virtual_ward_mentioned": analysis["virtual_ward_mentioned"],
                        "summary": analysis["summary"],
                        "mentions_count": analysis["mentions_count"],
                        "analysis_source": "new"
                    }
                    
                    results.append(result)
                    
                    # Update the paper in latest_results if it exists
                    if url in current_papers:
                        for paper in latest_results["board_papers"]:
                            if paper["url"] == url:
                                paper["virtual_ward_mentioned"] = analysis["virtual_ward_mentioned"]
                                paper["virtual_ward_summary"] = analysis["summary"]
                                paper["virtual_ward_mentions_count"] = analysis["mentions_count"]
                                break
                                
                    # Save the updated results
                    save_results(latest_results)
                    
                except Exception as e:
                    results.append({
                        "url": url,
                        "title": current_papers.get(url, {}).get("title", "Unknown"),
                        "virtual_ward_mentioned": False,
                        "summary": f"Error during analysis: {str(e)}",
                        "mentions_count": 0,
                        "analysis_source": "error"
                    })
        
        return jsonify({
            "results": results,
            "success": True
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "success": False
        }), 500

@app.route('/test-analysis')
def test_analysis():
    """Test route to analyze a known working PDF"""
    try:
        # Use the local PDF file we know works
        pdf_path = "data/pdfs/blmk_march_2024.pdf"
        
        if not os.path.exists(pdf_path):
            return jsonify({
                "error": f"Test PDF file not found at {pdf_path}",
                "success": False
            }), 404
        
        # Initialize analyzer with our API key
        analyzer = PDFAnalyzer(GEMINI_API_KEY)
        
        # Analyze the PDF
        results = analyzer.analyze_pdf_url(pdf_path)
        
        return jsonify({
            "results": results,
            "success": True
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "success": False
        }), 500

if __name__ == '__main__':
    # Load existing data
    latest_results = load_existing_data()
    
    # Print API key info for debugging (first 8 chars only)
    print(f"Using API key starting with: {GEMINI_API_KEY[:8]}...")
    
    # Schedule the crawler to run every two weeks
    scheduler.add_job(run_crawler, 'interval', weeks=2)
    scheduler.start()
    
    # Start the flask app
    app.run(debug=True, host='0.0.0.0', port=5002) 