
# Netflix Content Strategy Analyzer 

## Overview

The **Netflix Content Strategy Analyzer** is a Streamlit-based interactive dashboard designed to analyze and visualize content performance data. It leverages both synthetic and real-world-style data to generate business insights, time series analysis, NLP-driven title exploration, clustering, and regional performance analytics.

---

## Project Structure

```
.
├── netflixAnalysisApp.py        # Streamlit UI and visual dashboard
├── netflixContentStrat.py       # Core analysis engine with analytics functions
└── README.md                    # This documentation file
```

---

##  Requirements

Ensure the following Python libraries are installed:

```bash
pip install streamlit pandas numpy matplotlib seaborn plotly scikit-learn nltk wordcloud statsmodels
```

---

##  How to Run the App

```bash
streamlit run netflixAnalysisApp.py
```

---

##  Module Descriptions

### `netflixAnalysisApp.py`

- **Purpose**: Provides a fully functional Streamlit dashboard interface.
- **Key Features**:
  - Generates a synthetic Netflix-style dataset
  - Supports content analysis, time-series forecasting, genre trends, sentiment-based recommendations, and regional breakdowns
  - Custom CSS for stylized dashboards

### `netflixContentStrat.py`

- **Purpose**: Defines the `NetflixAnalyzer` class responsible for data cleaning, statistical analysis, clustering, NLP processing, and more.

## `__init__(self, file_path)`
Initializes the analyzer object.
- `file_path`: path to the dataset (can be a dummy path if dataframe is provided manually)
- Initializes empty attributes: `df`, `clean_df`, `results`

---

## `load_data(self)`
Loads data from a CSV file into a DataFrame.
- Parses the dataset at the provided path.
- Prints a summary of rows and columns.
- Returns `True` on success.

---

## `preprocess_data(self)`
Cleans and transforms the dataset.
- Converts hours to numeric.
- Parses and extracts date components.
- Adds new features (season, days since release, etc.)
- Calls `add_derived_features()` to engineer additional columns.
- Drops missing values in release dates.

---

## `add_derived_features(self)`
Adds advanced engineered features to the data:
- Infers genre using keywords in the title and description.
- Maps language indicators to regions.
- Assigns performance tiers based on viewership percentiles.
- Extracts features from titles (length, word count).
- Adds holiday release period indicators.

---

## `infer_genre(self, title, description=None)`
Attempts to determine genre based on keywords in the title or description.
- Uses a dictionary of keyword lists for each genre.
- Returns the first genre matched or "Other".

---

## `map_language_to_region(self, language)`
Maps a language string to a geographical region.
- Covers many international languages.
- Defaults to "Other" if no match is found.

---

## `generate_basic_stats(self)`
Computes summary statistics:
- Total content count, total and average hours viewed.
- Distribution of content types, top genres and languages.
- Saves a basic bar chart of content types.

---

## `run_time_series_analysis(self)`
Analyzes trends over time:
- Aggregates viewership monthly.
- Performs seasonal decomposition.
- Forecasts viewership using ARIMA.
- Determines the best day of the week for releasing content.
- Saves multiple time series visualizations.

---

## `analyze_content_performance(self)`
Breaks down performance drivers:
- Distribution of performance tiers.
- Average viewership by content type and content age.
- Uses title length as a proxy for runtime.
- Detects decay in viewership over time.
- Highlights top-performing genres.

---

## `perform_cluster_analysis(self)`
Segments content into clusters using KMeans:
- Standardizes features like Hours Viewed and Title Length.
- Uses PCA for dimensionality reduction and visualization.
- Saves elbow curve and cluster profile plots.
- Describes each cluster with its characteristics.

---

## `analyze_titles_with_nlp(self)`
Analyzes title text using NLP:
- Creates word clouds and frequency plots.
- Performs sentiment analysis using VADER.
- Groups titles by sentiment and compares performance.
- Saves visualizations for common words and sentiment.

---

## `analyze_geographical_performance(self)`
Analyzes how content performs in different regions:
- Average hours viewed by region.
- Heatmaps of content types by region.
- Identifies the most popular genre in each region.

---

## `generate_business_recommendations(self)`
Synthesizes strategic recommendations based on all analyses:
- Uses stored `results` from previous steps.
- Suggests genre investment, content format, timing, etc.
- Ideal for business decision support.

---



##  Dashboard Components

The app is divided into the following dashboards:

| Component               | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| **Performance Overview** | Summary metrics, content type distribution, top genres                    |
| **Time Series Analysis** | Monthly trends, best release day/quarter, content aging impact            |
| **Content Analysis**     | Title length effects, genre effectiveness, word clouds, clustering         |
| **Geographical View**    | Regional preferences and best-performing content                           |
| **Recommendations**      | Data-driven content strategy insights for Netflix                          |

---

##  Sample Dataset Features

The synthetic dataset is generated with:
- Randomly simulated titles, genres, content types
- Viewership patterns that mimic real-world seasonality and genre popularity
- Derived features like performance tiers, title length buckets, release periods, etc.

---

##  Recommendations Engine

Key factors in recommendations include:
- Optimal content type by region
- Top-performing genres
- Release timing strategies
- Content longevity analysis
- Ideal runtime and title sentiment

---


