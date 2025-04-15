from flask import Flask, render_template, request, jsonify
from apscheduler.schedulers.background import BackgroundScheduler
import os
import datetime
import json
import asyncio
from crawler.crawler import AdvancedCrawler
from utils.config import Config
import re
from utils.pdf_analyzer import PDFAnalyzer

app = Flask(__name__)
app.config['SECRET_KEY'] = Config.SECRET_KEY


crawler = AdvancedCrawler()

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
            data = json.load(f)
            
            # Filter papers to only include those from 2024 or later
            if "board_papers" in data:
                data["board_papers"] = [
                    paper for paper in data["board_papers"] 
                    if is_from_2024_or_later(paper.get("date", "Unknown"))
                ]
                print(f"Loaded {len(data['board_papers'])} papers from 2024 or later")
            return data
    return {"last_run": None, "board_papers": []}

def save_results(results):
    """Save scraping results to file"""
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)

def process_paper_analysis(paper, analyzer, scrape_only=False):
    """Analyze a single paper for virtual ward mentions"""
    if scrape_only:
        return {
            "virtual_ward_mentioned": False,
            "summary": "Not analyzed (scrape-only mode)",
            "virtual_ward_mentions_count": 0
        }
    
    try:
        results = analyzer.analyze_pdf_url(paper["url"], verbose=False)
        print(f"  - Virtual ward mentioned: {results['virtual_ward_mentioned']}")
        
        # Update paper with metadata if available
        if results.get('title') and results['title'] != 'Unknown':
            paper['title'] = results['title']
        if results.get('date') and results['date'] != 'Unknown':
            paper['date'] = results['date']
        if results.get('organization') and results['organization'] != 'Unknown':
            paper['organization'] = results['organization']
        
        return {
            "virtual_ward_mentioned": results["virtual_ward_mentioned"],
            "summary": results["summary"],
            "virtual_ward_mentions_count": results["mentions_count"]
        }
    except Exception as e:
        error_msg = f"Error analyzing PDF {paper['url']}: {str(e)}"
        print(error_msg)
        return {
            "virtual_ward_mentioned": False,
            "summary": f"Error during analysis: {str(e)}",
            "virtual_ward_mentions_count": 0
        }

def create_paper_dict(paper, org_url, existing_papers):
    """Create a standardized paper dictionary"""
    if type(paper) == list:
        print(paper)
        return {}
    
    # Get filename from the URL
    filename = paper.get("title", "Unknown") if paper.get("title") != "Unknown" else paper["url"].split("/")[-1]
    
    return {
        "url": paper["url"],
        "filename": filename,
        "title": paper.get("title", "Unknown"),
        "date": paper.get("date", "Unknown"),
        "trust": paper.get("trust", org_url),
        "organization": paper.get("organization", "Unknown"),
        "virtual_ward_mentioned": False,
        "summary": "Not analyzed",
        "is_new": (paper["url"], paper.get("title", "Unknown")) not in existing_papers,
        "found_date": datetime.datetime.now().isoformat(),
        "sort_date": paper.get("date", "9999-99") if paper.get("date") != "Unknown" else "9999-99"
    }

def update_and_save_results(papers):
    """Update latest results and save to file"""
    global latest_results
    latest_results = {
        "last_run": datetime.datetime.now().isoformat(),
        "board_papers": papers
    }
    save_results(latest_results)

def is_from_2024_or_later(date_str):
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
            
        # Default to keeping the paper if we can't determine the date
        else:
            return False
    except:
        return False

async def process_organization(url, existing_papers, scrape_only=False):
    """Process a single organization's papers"""
    print(f"\nProcessing organization: {url}")
    papers = await crawler.deep_crawl(url)
    org_papers = []
    
    for paper in papers:
        paper_dict = create_paper_dict(paper, url, existing_papers)
        
        # Get metadata for date filtering
        if not scrape_only:
            analysis_results = process_paper_analysis(paper_dict, pdf_analyzer, scrape_only)
            paper_dict.update({
                "virtual_ward_mentioned": analysis_results["virtual_ward_mentioned"],
                "summary": analysis_results["summary"],
                "virtual_ward_mentions_count": analysis_results["virtual_ward_mentions_count"]
            })
            
        # Only include papers from 2024 onwards
        if is_from_2024_or_later(paper_dict.get("date", "Unknown")):
            org_papers.append(paper_dict)
        else:
            print(f"Skipping paper with date {paper_dict.get('date', 'Unknown')}: {paper_dict.get('title', 'Unknown')}")
    
    print(f"Found {len(org_papers)} papers from 2024 onwards for {url}")
    return org_papers

def run_crawler():
    """Run the NHS website crawler to find board papers"""
    existing_data = load_existing_data()
    existing_papers = {(paper["url"], paper["title"]) for paper in existing_data.get("board_papers", [])}
    
    # Define NHS organization URLs
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
    
    print(f"Starting crawler at {datetime.datetime.now()}")
    print(f"Crawling {len(test_urls)} NHS websites")
    
    all_papers = []
    for test_url in test_urls:
        org_papers = asyncio.run(process_organization(test_url, existing_papers))
        all_papers.extend(org_papers)
        update_and_save_results(all_papers)
        print(f"Saved results after processing {test_url}")
    
    print(f"\nCrawler and analysis completed at {datetime.datetime.now()}")
    print(f"Total papers found: {len(all_papers)}")

async def run_crawler_for_specific_urls(urls, scrape_only=False):
    """Run the website crawler for specific URLs"""
    existing_data = load_existing_data()
    existing_papers = {(paper["url"], paper["title"]) for paper in existing_data.get("board_papers", [])}
    
    all_papers = []
    for url in urls:
        org_papers = await process_organization(url, existing_papers, scrape_only)
        all_papers.extend(org_papers)
        update_and_save_results(all_papers)
        print(f"Saved results after processing {url}")
    
    print(f"\nCrawler and analysis completed at {datetime.datetime.now()}")
    print(f"Total papers found: {len(all_papers)}")
    return all_papers

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
    scrape_only = data.get('scrape_only', False)  # Get scrape_only parameter from request
    
    if not urls:
        return jsonify({"status": "error", "message": "No URLs provided"})
    
    # Run the crawler for specific URLs with scrape_only parameter
    papers = asyncio.run(run_crawler_for_specific_urls(urls, scrape_only=scrape_only))
    
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
            return jsonify({"error": "No URLs provided", "success": False}), 400
        
        current_papers = {paper["url"]: paper for paper in latest_results.get("board_papers", [])}
        results = []
        
        for url in urls:
            if url in current_papers and current_papers[url].get("summary") != "Not analyzed":
                paper = current_papers[url]
                
                # Skip if paper is before 2024
                if not is_from_2024_or_later(paper.get("date", "Unknown")):
                    print(f"Skipping pre-2024 paper: {paper.get('title', 'Unknown')}")
                    continue
                    
                results.append({
                    "url": url,
                    "title": paper.get("title", "Unknown"),
                    "virtual_ward_mentioned": paper.get("virtual_ward_mentioned", False),
                    "summary": paper.get("summary", "No analysis available"),
                    "mentions_count": paper.get("virtual_ward_mentions_count", 0),
                    "date": paper.get("date", "Unknown"),
                    "organization": paper.get("organization", "Unknown"),
                    "analysis_source": "cached"
                })
                print(f"Using cached analysis for {url}")
            else:
                try:
                    print(f"Analyzing {url} for virtual ward mentions")
                    analysis = pdf_analyzer.analyze_pdf_url(url)
                    
                    # Skip if paper is before 2024
                    if not is_from_2024_or_later(analysis.get("date", "Unknown")):
                        print(f"Skipping pre-2024 paper: {analysis.get('title', 'Unknown')}")
                        continue
                    
                    result = {
                        "url": url,
                        "title": analysis.get("title", current_papers.get(url, {}).get("title", "Unknown")),
                        "virtual_ward_mentioned": analysis["virtual_ward_mentioned"],
                        "summary": analysis["summary"],
                        "mentions_count": analysis["mentions_count"],
                        "date": analysis.get("date", "Unknown"),
                        "organization": analysis.get("organization", "Unknown"),
                        "analysis_source": "new"
                    }
                    results.append(result)
                    
                    if url in current_papers:
                        update_paper_analysis(url, analysis)
                        save_results(latest_results)
                        
                except Exception as e:
                    results.append({
                        "url": url,
                        "title": current_papers.get(url, {}).get("title", "Unknown"),
                        "virtual_ward_mentioned": False,
                        "summary": f"Error during analysis: {str(e)}",
                        "mentions_count": 0,
                        "date": "Unknown",
                        "organization": current_papers.get(url, {}).get("organization", "Unknown"),
                        "analysis_source": "error"
                    })
        
        return jsonify({"results": results, "success": True})
        
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500

def update_paper_analysis(url, analysis):
    """Update paper analysis in latest_results"""
    for paper in latest_results["board_papers"]:
        if paper["url"] == url:
            paper["virtual_ward_mentioned"] = analysis["virtual_ward_mentioned"]
            paper["summary"] = analysis["summary"]
            paper["virtual_ward_mentions_count"] = analysis["mentions_count"]
            
            # Update paper metadata if available
            if analysis.get("date") and analysis["date"] != "Unknown":
                paper["date"] = analysis["date"]
                # Update sort_date based on new date
                paper["sort_date"] = analysis["date"] if is_from_2024_or_later(analysis["date"]) else "9999-99"
                
            if analysis.get("title") and analysis["title"] != "Unknown":
                paper["title"] = analysis["title"]
                
            if analysis.get("organization") and analysis["organization"] != "Unknown":
                paper["organization"] = analysis["organization"]
            break

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
    

@app.route('/reset-existing-papers', methods=['POST'])
def reset_existing_papers():
    """
    Danger: This route clears the stored board_papers.json file and resets it
    so that existing_papers = {} for future crawler runs.
    """
    empty_results = {
        "last_run": None,
        "board_papers": []
    }
    save_results(empty_results)  # Same save_results() used elsewhere
    
    global latest_results
    latest_results = empty_results
    
    return jsonify({
        "status": "success",
        "message": "Existing papers have been reset",
        "results": latest_results
    })



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