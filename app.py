import asyncio
import datetime
import os
from typing import Any

from flask import Flask, Response, jsonify, render_template, request

from crawler.crawler import AdvancedCrawler
from utils.config import Config
from utils.pdf_analyzer import PDFAnalyzer
from utils.results_store import ResultsStore

app = Flask(__name__)
app.config["SECRET_KEY"] = Config.SECRET_KEY

crawler = AdvancedCrawler()

DATA_DIR = Config.DATA_DIR
os.makedirs(DATA_DIR, exist_ok=True)

# Persistent result cache
RESULTS_FILE = os.path.join(DATA_DIR, "board_papers.json")
results_store = ResultsStore(RESULTS_FILE)

GEMINI_API_KEY: str = Config.GEMINI_API_KEY
pdf_analyzer = PDFAnalyzer(GEMINI_API_KEY)


def _apply_metadata(
    paper: dict, results: dict, keys=("title", "date", "organization")
) -> None:
    for k in keys:
        v = results.get(k)
        if v and v != "Unknown":
            paper[k] = v


def _extract_valid_term_data(
    raw_term_data: dict[str, dict[str, Any]],
    analyzer: PDFAnalyzer,
) -> dict[str, dict[str, list[str]]]:
    """
    Given raw terms_data from Gemini, return only those terms
    which have non-empty, cleaned quotes and summaries.
    """
    cleaned = {}

    for term, entry in raw_term_data.items():
        # 1) normalize to lists
        quotes = entry.get("quotes", [])
        if isinstance(quotes, str):
            quotes = [quotes]
        summaries = entry.get("summaries", [])
        if isinstance(summaries, str):
            summaries = [summaries]

        # 2) clean & filter by length
        cleaned_quotes = [
            analyzer.clean_repetitive_text(q)
            for q in quotes
            if q and len(q.strip()) > 10
        ]
        cleaned_summaries = [
            analyzer.clean_repetitive_text(s)
            for s in summaries
            if s and len(s.strip()) > 10
        ]

        # 3) only keep if both exist
        if cleaned_quotes and cleaned_summaries:
            cleaned[term] = {
                "quotes": cleaned_quotes,
                "summaries": cleaned_summaries,
            }

    return cleaned


def analyze_full_paper(
    paper: dict,
    analyzer: PDFAnalyzer,
) -> dict:
    """Do a full Gemini extraction on one paper and return cleaned results."""
    results = analyzer.analyze_pdf_url(paper["url"], verbose=False)
    print(f"  - Found terms: {', '.join(results.get('terms_found', []))}")

    # add title, date and organisation if found in result
    _apply_metadata(paper, results)

    # Extract & clean term data
    raw_terms_data = results.get("terms_data", {})
    clean_terms_data = _extract_valid_term_data(raw_terms_data, analyzer)
    valid_terms = list(clean_terms_data)

    return {
        "has_relevant_terms": len(valid_terms) > 0,
        "terms_found": valid_terms,
        "terms_count": len(valid_terms),
        "terms_data": clean_terms_data,
    }


# Helper for formatting analysis responses
def _build_analysis_response(paper: dict, source: str) -> dict:
    return {
        "url": paper.get("url"),
        "title": paper.get("title", "Unknown"),
        "has_relevant_terms": paper.get("has_relevant_terms", False),
        "terms_found": paper.get("terms_found", []),
        "terms_count": paper.get("terms_count", 0),
        "terms_data": paper.get("terms_data", {}),
        "date": paper.get("date", "Unknown"),
        "organization": paper.get("organization", "Unknown"),
        "analysis_source": source,
    }


def update_paper_analysis(url: str, analysis: dict) -> None:
    """Update a stored paper's analysis using the cleaning helper."""
    for paper in results_store.board_papers:
        if paper["url"] != url:
            continue

        # Clean + filter the raw terms_data
        terms_data = _extract_valid_term_data(
            analysis.get("terms_data", {}), pdf_analyzer
        )

        # Bulk-update the paper record
        paper.update(
            {
                "has_relevant_terms": bool(terms_data),
                "terms_found": list(terms_data),
                "terms_count": len(terms_data),
                "terms_data": terms_data,
            }
        )

        # Refresh metadata and sort key
        _apply_metadata(paper, analysis)
        paper["sort_date"] = paper.get("date", "")
        break


def create_paper_dict(
    paper: dict[str, Any],
    org_url: str,
    existing_papers: set[tuple[str, str]],
) -> dict[str, Any]:
    """Create a standardized paper dictionary."""
    # Reject non-dicts or entries without a URL
    if not isinstance(paper, dict) or "url" not in paper:
        print(paper)
        return {}

    # Normalize fields
    title = paper.get("title", "Unknown")
    filename = title if title != "Unknown" else paper["url"].rsplit("/", 1)[-1]
    date = paper.get("date", "Unknown")
    sort_date = date if date != "Unknown" else "9999-99"
    trust = paper.get("trust", org_url)
    organization = paper.get("organization", "Unknown")
    is_new = (paper["url"], title) not in existing_papers

    return {
        "url": paper["url"],
        "filename": filename,
        "title": title,
        "date": date,
        "trust": trust,
        "organization": organization,
        "has_relevant_terms": False,
        "terms_found": [],
        "terms_count": 0,
        "terms_data": {},
        "is_new": is_new,
        "found_date": datetime.datetime.now().isoformat(),
        "sort_date": sort_date,
    }


async def process_organization(
    url: str,
    existing_papers: set[tuple[str, str]],
    scrape_only: bool = False,
) -> list[dict]:
    """Crawl one NHS site and (optionally) analyse its PDFs."""
    print(f"\nProcessing organization: {url}")
    papers = await crawler.deep_crawl(url)
    org_papers = []
    for raw in papers:
        paper = create_paper_dict(raw, url, existing_papers)

        # Always fetch a date first – we need it for every mode
        date = pdf_analyzer.extract_date_only(paper["url"], keep_temp_file=True)
        paper["date"] = date
        if not pdf_analyzer.is_from_2024_or_later(date):
            print(f"Skipping pre‑2024 paper: {date}")
            continue

        if not scrape_only:
            analysis = analyze_full_paper(paper, pdf_analyzer)
            _apply_metadata(paper, analysis)
            paper.update(
                {
                    "has_relevant_terms": analysis["has_relevant_terms"],
                    "terms_found": analysis["terms_found"],
                    "terms_count": analysis["terms_count"],
                    "terms_data": analysis["terms_data"],
                }
            )
        org_papers.append(paper)

    print(f"Found {len(org_papers)} ≥ 2024 papers for {url}")
    return org_papers


async def _crawl_urls(
    urls: list[str],
    scrape_only: bool = False,
) -> list[dict]:
    """Common crawl loop for a list of URLs."""
    existing_papers = {(p["url"], p["title"]) for p in results_store.board_papers}
    all_papers: list[dict] = []
    for url in urls:
        org_papers = await process_organization(url, existing_papers, scrape_only)
        all_papers.extend(org_papers)
        results_store.update(all_papers)
        print(f"Saved results after processing {url}")
    print(f"\nCrawler and analysis completed at {datetime.datetime.now()}")
    print(f"Total papers found: {len(all_papers)}")
    return all_papers


def run_crawler():
    """Run the NHS website crawler to find board papers."""
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
        "https://www.nnuh.nhs.uk/",
    ]
    print(f"Starting crawler at {datetime.datetime.now()}")
    print(f"Crawling {len(test_urls)} NHS websites")
    # reuse our helper in a sync context
    asyncio.run(_crawl_urls(test_urls, scrape_only=False))


async def run_crawler_for_specific_urls(
    urls: list[str],
    scrape_only: bool = False,
) -> list[dict]:
    """Run the website crawler for specific URLs."""
    # directly reuse the same helper
    return await _crawl_urls(urls, scrape_only)


@app.route("/")
def index():
    """Homepage route"""
    return render_template("index.html", results=results_store.board_papers)


@app.route("/results")
def view_results():
    """View the latest results"""
    return jsonify(results_store.board_papers)


@app.route("/analyze-papers", methods=["POST"])
def analyze_papers() -> Response:
    data = request.get_json() or {}
    urls = data.get("urls", [])
    if not urls:
        return jsonify({"error": "No URLs provided", "success": False}), 400

    current_papers = {p["url"]: p for p in results_store.board_papers}
    results = []

    for url in urls:
        paper = current_papers.get(url)
        try:
            # 1) Cached analysis
            if paper and paper.get("terms_count", 0) > 0:
                results.append(_build_analysis_response(paper, "cached"))
                print(f"Using cached analysis for {url}")
                continue

            # 2) Fresh analysis: first check the date
            print(f"Checking date for: {url}")
            date = pdf_analyzer.extract_date_only(url, keep_temp_file=True)
            if not pdf_analyzer.is_from_2024_or_later(date):
                print(f"Skipping pre-2024 paper: {date}")
                continue

            # 3) Do the full Gemini extraction
            print(f"Analyzing {url} for healthcare terms")
            # pass minimal paper dict (url plus any existing fields)
            analysis = analyze_full_paper({"url": url, **(paper or {})}, pdf_analyzer)

            # 4) Update the store
            if paper:
                update_paper_analysis(url, analysis)
            else:
                new_paper = {"url": url, **analysis}
                _apply_metadata(new_paper, analysis)
                new_paper["sort_date"] = new_paper.get("date", "")
                results_store.board_papers.append(new_paper)

            results_store.update(results_store.board_papers)

            # 5) Collect the response
            updated = current_papers.get(url, new_paper)
            results.append(_build_analysis_response(updated, "new"))

        except Exception:
            err = paper or {}
            results.append(
                {
                    "url": url,
                    "title": err.get("title", "Unknown"),
                    "has_relevant_terms": False,
                    "terms_found": [],
                    "terms_count": 0,
                    "terms_data": {},
                    "date": err.get("date", "Unknown"),
                    "organization": err.get("organization", "Unknown"),
                    "analysis_source": "error",
                }
            )

    return jsonify({"results": results, "success": True})


@app.route("/run-crawler", methods=["POST"])
def run_crawler_route():
    asyncio.run(run_crawler())
    return jsonify(success=True)


@app.route("/test-specific-urls", methods=["POST"])
def test_specific_urls():
    data = request.get_json()
    urls = data.get("urls", [])
    scrape_only = data.get("scrape_only", False)
    papers = asyncio.run(run_crawler_for_specific_urls(urls, scrape_only=scrape_only))
    return jsonify(results={"board_papers": papers}, status="success")


if __name__ == "__main__":
    # Start Flask in production (no debug=True)
    print(f"Using API key starting with: {GEMINI_API_KEY[:8]}...")
    app.run(debug=True, host="0.0.0.0", port=5002)
