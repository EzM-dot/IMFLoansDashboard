# IMF Loans Dashboard

A web application that scrapes IMF loan data and presents it in an interactive Tableau-style dashboard.

## Features
- Web scraping of IMF loan data by country
- Data export to CSV
- Interactive dashboard with visualizations
- Filter by country
- Responsive design

## Prerequisites
- Python 3.8+
- pip

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd imf_loans_dashboard
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the scraper to fetch the latest data:
   ```bash
   python src/scraper.py
   ```

2. Start the dashboard:
   ```bash
   streamlit run src/dashboard.py
   ```

3. Open your browser and navigate to `http://localhost:8501`

## Project Structure

```
imf_loans_dashboard/
├── data/                   # Directory for storing CSV data
├── src/
│   ├── scraper.py         # Web scraping script
│   └── dashboard.py       # Streamlit dashboard
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Notes
- The current implementation uses sample data. You'll need to update the `scraper.py` file with the actual IMF website scraping logic.
- The dashboard will automatically update when new data is available.

## License
This project is licensed under the MIT License.
