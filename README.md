# NHS Board Paper Scraper

A Flask application that scrapes NHS Trust and ICB websites for board papers, analyzes them using LLM technology, and provides insights on specific healthcare topics.

## Features

- Automatically scrapes NHS Trust and ICB websites for board papers using Crawl4AI
- Periodic scraping (every two weeks) to find new board papers
- Highlights newly discovered board papers
- (Future) Uses Google Gemini to analyze papers for information on:
  - Virtual wards
  - Remote patient monitoring
  - Hospital at home
  - Proactive care

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the application:
   ```
   python app.py
   ```

## Project Structure

- `app.py`: Main Flask application
- `crawler/`: Web crawler implementation
- `models/`: Database models
- `static/`: Static files (CSS, JavaScript)
- `templates/`: HTML templates
- `utils/`: Utility functions 