"""
Netflix Content Strategy - Comprehensive Analysis Framework
==========================================================
This script provides a comprehensive analysis of Netflix content data,
including advanced analytics, visualizations, time series analysis,
NLP insights, and business strategy recommendations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import os
import re
import json
from collections import Counter

# Advanced analytics libraries
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm

# NLP libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud

# Interactive visualizations (comment these if not installed)
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx

# Suppress warnings
warnings.filterwarnings('ignore')

# Create output directory
os.makedirs('netflix_analysis_output', exist_ok=True)

# Download necessary NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
except:
    print("Could not download NLTK resources. NLP functions may be limited.")

# Configure visualization settings
plt.style.use('fivethirtyeight')
sns.set_theme(style="whitegrid")

class NetflixAnalyzer:
    """Comprehensive Netflix content analysis toolkit"""
    
    def __init__(self, file_path):
        """Initialize with the path to the Netflix dataset"""
        self.file_path = file_path
        self.df = None
        self.clean_df = None
        self.results = {}  # Store analysis results
        
        # Configure plot styling
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
        
        # Define color schemes
        self.color_palette = sns.color_palette("viridis", 10)
        self.color_map = "viridis"
        
    def load_data(self):
        """Load and perform initial processing of Netflix data"""
        print("Loading Netflix dataset...")
        self.df = pd.read_csv(self.file_path)
        print(f"Successfully loaded {self.df.shape[0]} rows and {self.df.shape[1]} columns")
        return True
                        
    def preprocess_data(self):
        """Clean and preprocess the dataset"""
        if self.df is None:
            print("No data loaded. Call load_data() first.")
            return False
            
        print("Preprocessing data...")
        # Make a copy to avoid modifying the original dataframe
        self.clean_df = self.df.copy()
        
        # Clean Hours Viewed
        try:
            self.clean_df['Hours Viewed'] = pd.to_numeric(
                self.clean_df['Hours Viewed'].str.replace(",", ""), 
                errors='coerce'
            )
        except:
            print("Warning: Issues with 'Hours Viewed' column. Check format.")
        
        # Handle dates
        try:
            self.clean_df['Release Date'] = pd.to_datetime(self.clean_df['Release Date'], errors='coerce')
            
            # Create date-related features
            self.clean_df['Year'] = self.clean_df['Release Date'].dt.year
            self.clean_df['Month'] = self.clean_df['Release Date'].dt.month
            self.clean_df['Quarter'] = 'Q' + self.clean_df['Release Date'].dt.quarter.astype(str)
            self.clean_df['Day of Week'] = self.clean_df['Release Date'].dt.day_name()
            self.clean_df['Day'] = self.clean_df['Release Date'].dt.day
            self.clean_df['Week of Year'] = self.clean_df['Release Date'].dt.isocalendar().week
            
            # Create season
            self.clean_df['Season'] = self.clean_df['Release Date'].dt.month % 12 // 3 + 1
            season_map = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'}
            self.clean_df['Season'] = self.clean_df['Season'].map(season_map)
            
            # Calculate days since release (as of today)
            today = pd.Timestamp.now().normalize()
            self.clean_df['Days Since Release'] = (today - self.clean_df['Release Date']).dt.days
            
        except Exception as e:
            print(f"Warning with date processing: {str(e)}")
        
        # Drop rows with missing release dates
        self.clean_df = self.clean_df.dropna(subset=['Release Date'])
        
        # Add enhanced features
        self.add_derived_features()
        
        print(f"Preprocessing complete. Dataset now has {self.clean_df.shape[0]} rows.")
        return True
    
    def add_derived_features(self):
        """Add derived and calculated features to the dataset"""
        # Add inferred genre
        self.clean_df['Inferred Genre'] = self.clean_df['Title'].apply(self.infer_genre)
        
        # Map languages to regions
        self.clean_df['Region'] = self.clean_df['Language Indicator'].apply(self.map_language_to_region)
        
        # Add performance tiers
        hours_viewed_percentiles = self.clean_df['Hours Viewed'].quantile([0.25, 0.5, 0.75, 0.9])
        
        def assign_performance_tier(hours):
            if hours <= hours_viewed_percentiles[0.25]:
                return 'Low Performer'
            elif hours <= hours_viewed_percentiles[0.5]:
                return 'Below Average'
            elif hours <= hours_viewed_percentiles[0.75]:
                return 'Above Average'
            elif hours <= hours_viewed_percentiles[0.9]:
                return 'High Performer'
            else:
                return 'Top Performer'
                
        self.clean_df['Performance Tier'] = self.clean_df['Hours Viewed'].apply(assign_performance_tier)
        
        # Extract title features
        self.clean_df['Title Length'] = self.clean_df['Title'].apply(lambda x: len(str(x)) if pd.notnull(x) else 0)
        self.clean_df['Title Word Count'] = self.clean_df['Title'].apply(
            lambda x: len(str(x).split()) if pd.notnull(x) else 0
        )
        
        # Add holiday/special period indicator if appropriate column exists
        try:
            # Example approach - would need to be adjusted to actual data format
            def is_holiday_period(date):
                month_day = (date.month, date.day)
                
                # Major US/global holidays and periods
                holiday_ranges = [
                    # Christmas/New Year
                    {(12, day): 'Christmas/New Year' for day in range(15, 32)},
                    {(1, day): 'Christmas/New Year' for day in range(1, 8)},
                    
                    # Thanksgiving
                    {(11, day): 'Thanksgiving' for day in range(20, 30)},
                    
                    # Summer Break
                    {(6, day): 'Summer Break' for day in range(1, 31)},
                    {(7, day): 'Summer Break' for day in range(1, 32)},
                    {(8, day): 'Summer Break' for day in range(1, 32)},
                ]
                
                for holiday_dict in holiday_ranges:
                    if month_day in holiday_dict:
                        return holiday_dict[month_day]
                return 'Regular Period'
                
            self.clean_df['Release Period'] = self.clean_df['Release Date'].apply(is_holiday_period)
        except:
            pass
            
        print("Added derived features to the dataset")
    
    def infer_genre(self, title, description=None):
        """Infer genre from title and description with improved pattern matching"""
        if not isinstance(title, str):
            return "Unknown"
            
        text = (str(title) + ' ' + str(description or '')).lower()
        
        # Define genre patterns with comprehensive keywords
        genres = {
            'Comedy': ['comedy', 'funny', 'laugh', 'humor', 'sitcom', 'jokes', 'standup'],
            'Horror/Thriller': ['horror', 'thriller', 'scary', 'fear', 'terror', 'haunt', 'dark', 'creepy', 'ghost', 'apocalypse', 'zombie'],
            'Romance': ['romance', 'love', 'relationship', 'dating', 'romantic', 'passion', 'affair', 'marriage'],
            'Action/Adventure': ['action', 'adventure', 'mission', 'hunt', 'chase', 'war', 'battle', 'fight', 'epic', 'quest', 'journey', 'explosion'],
            'Documentary': ['documentary', 'real', 'true story', 'actual', 'history', 'historical', 'facts', 'interview', 'investigative'],
            'Drama': ['drama', 'story', 'emotional', 'life', 'struggle', 'family', 'relationship', 'character', 'conflict'],
            'Sci-Fi': ['sci-fi', 'future', 'space', 'alien', 'robot', 'tech', 'planet', 'science fiction', 'futuristic', 'dystopia', 'time travel'],
            'Fantasy': ['fantasy', 'magic', 'dragon', 'wizard', 'elf', 'myth', 'mythical', 'imaginary', 'fairy', 'supernatural'],
            'Crime': ['crime', 'detective', 'murder', 'mystery', 'police', 'heist', 'criminal', 'investigation', 'forensic', 'case', 'solve'],
            'Animation': ['animation', 'cartoon', 'anime', 'animated', 'cgi', 'pixar', 'disney', 'drawn'],
            'Family': ['family', 'kids', 'children', 'parents', 'child', 'friendly', 'childhood'],
            'Sports': ['sports', 'football', 'basketball', 'baseball', 'soccer', 'athlete', 'game', 'tournament', 'match', 'olympic', 'champion'],
            'Musical': ['musical', 'music', 'song', 'dance', 'singing', 'band', 'concert', 'rhythm', 'melody'],
            'Biography': ['biography', 'biopic', 'life story', 'based on', 'true story', 'memoir'],
            'History': ['history', 'historical', 'period', 'century', 'ancient', 'medieval', 'era', 'dynasty', 'kingdom']
        }
        
        # Check for each genre pattern
        matched_genres = []
        for genre, keywords in genres.items():
            if any(keyword in text for keyword in keywords):
                matched_genres.append(genre)
        
        # Return the first matched genre, or multiple genres if found
        if len(matched_genres) == 1:
            return matched_genres[0]
        elif len(matched_genres) > 1:
            return matched_genres[0]  # Just return the first match for simplicity
        else:
            return "Other"
    
    def map_language_to_region(self, language):
        """Map language to geographic region with comprehensive mapping"""
        if not isinstance(language, str):
            return "Unknown"
            
        language = str(language).lower()
        
        # Comprehensive language to region mapping
        language_map = {
            'english': 'North America/UK',
            'korean': 'South Korea',
            'spanish': 'Latin America/Spain',
            'hindi': 'India',
            'japanese': 'Japan',
            'french': 'France/Canada',
            'german': 'Germany/Austria',
            'portuguese': 'Brazil/Portugal',
            'turkish': 'Turkey',
            'italian': 'Italy',
            'thai': 'Thailand',
            'mandarin': 'China/Taiwan',
            'cantonese': 'Hong Kong/China',
            'arabic': 'Middle East',
            'russian': 'Russia',
            'swedish': 'Scandinavia',
            'danish': 'Scandinavia',
            'norwegian': 'Scandinavia',
            'dutch': 'Netherlands',
            'polish': 'Poland',
            'indonesian': 'Indonesia',
            'vietnamese': 'Vietnam',
            'filipino': 'Philippines',
            'greek': 'Greece',
            'hebrew': 'Israel',
            'bengali': 'India/Bangladesh',
            'tamil': 'India/Sri Lanka',
            'telugu': 'India',
            'malayalam': 'India',
            'marathi': 'India',
            'ukrainian': 'Ukraine',
            'czech': 'Czech Republic',
            'hungarian': 'Hungary',
            'finnish': 'Finland',
            'romanian': 'Romania',
            'bulgarian': 'Bulgaria',
            'serbian': 'Serbia/Balkans',
            'croatian': 'Croatia/Balkans',
            'malay': 'Malaysia',
            'icelandic': 'Iceland'
        }
        
        for lang, region in language_map.items():
            if lang in language:
                return region
        
        return "Other"
    
    def save_fig(self, plt, filename, dpi=300):
        """Save figure with proper error handling"""
        filepath = os.path.join('netflix_analysis_output', filename)
        try:
            plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
            print(f"Saved visualization: {filename}")
        except Exception as e:
            print(f"Error saving {filename}: {str(e)}")
    
    def generate_basic_stats(self):
        """Generate basic statistics and summary metrics"""
        if self.clean_df is None:
            print("No processed data available")
            return
            
        print("\n===== GENERATING BASIC STATISTICS =====")
        
        # Overall metrics
        total_content = len(self.clean_df)
        total_hours = self.clean_df['Hours Viewed'].sum()
        avg_hours = self.clean_df['Hours Viewed'].mean()
        
        # Content type distribution
        content_type_dist = self.clean_df['Content Type'].value_counts(normalize=True)
        
        # Top genres
        top_genres = self.clean_df['Inferred Genre'].value_counts().head(5)
        
        # Top languages
        top_languages = self.clean_df['Language Indicator'].value_counts().head(5)
        
        # Store results
        self.results['basic_stats'] = {
            'total_content': total_content,
            'total_hours_viewed': float(total_hours),
            'avg_hours_per_content': float(avg_hours),
            'content_type_distribution': content_type_dist.to_dict(),
            'top_genres': top_genres.to_dict(),
            'top_languages': top_languages.to_dict()
        }
        
        # Output results
        print(f"Total Content Items: {total_content}")
        print(f"Total Hours Viewed: {total_hours:,.2f} million")
        print(f"Average Hours per Content: {avg_hours:,.2f} million")
        print("\nContent Type Distribution:")
        print(content_type_dist)
        print("\nTop 5 Genres:")
        print(top_genres)
        print("\nTop 5 Languages:")
        print(top_languages)
        
        # Create basic visualizations
        plt.figure(figsize=(12, 6))
        ax = content_type_dist.plot(kind='bar', color=self.color_palette)
        plt.title('Content Type Distribution', fontsize=16)
        plt.ylabel('Proportion', fontsize=14)
        plt.xlabel('Content Type', fontsize=14)
        plt.xticks(rotation=45)
        
        # Add percentages on top of bars
        for i, v in enumerate(content_type_dist):
            ax.text(i, v + 0.01, f'{v:.1%}', ha='center', fontsize=10)
            
        plt.tight_layout()
        self.save_fig(plt, 'content_type_distribution.png')
        plt.close()
        
        return self.results['basic_stats']
        
    # SECTION 1: TIME SERIES ANALYSIS & FORECASTING
    def run_time_series_analysis(self):
        """Perform comprehensive time series analysis"""
        if self.clean_df is None:
            print("No processed data available")
            return
            
        print("\n===== RUNNING TIME SERIES ANALYSIS =====")
        
        # Create time series by aggregating by month
        try:
            # Create a date column with first day of month
            self.clean_df['YearMonth'] = self.clean_df['Release Date'].dt.to_period('M').dt.to_timestamp()
            
            # Monthly viewership
            monthly_views = self.clean_df.groupby('YearMonth')['Hours Viewed'].sum().reset_index()
            monthly_views = monthly_views.set_index('YearMonth')
            
            # 1. Plot the time series
            plt.figure(figsize=(14, 7))
            monthly_views['Hours Viewed'].plot()
            plt.title('Monthly Viewership Trend', fontsize=16)
            plt.ylabel('Total Hours Viewed (Millions)', fontsize=14)
            plt.xlabel('Date', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            self.save_fig(plt, 'monthly_viewership_trend.png')
            plt.close()
            
            # 2. Seasonal Decomposition
            # Check if we have enough data points (at least 2 full periods)
            if len(monthly_views) >= 24:  
                try:
                    result = seasonal_decompose(monthly_views['Hours Viewed'], model='additive', period=12)
                    
                    plt.figure(figsize=(14, 10))
                    plt.subplot(411)
                    plt.plot(result.observed)
                    plt.title('Observed', fontsize=14)
                    plt.subplot(412)
                    plt.plot(result.trend)
                    plt.title('Trend', fontsize=14)
                    plt.subplot(413)
                    plt.plot(result.seasonal)
                    plt.title('Seasonal', fontsize=14)
                    plt.subplot(414)
                    plt.plot(result.resid)
                    plt.title('Residual', fontsize=14)
                    plt.tight_layout()
                    self.save_fig(plt, 'seasonal_decomposition.png')
                    plt.close()
                    
                    # Store seasonal components
                    seasonal_factors = result.seasonal[-12:].tolist()
                    self.results['time_series'] = {
                        'seasonal_factors': {f'month_{i+1}': factor for i, factor in enumerate(seasonal_factors)}
                    }
                    
                    # Identify strongest seasonal months
                    strongest_month = np.argmax(seasonal_factors) + 1
                    weakest_month = np.argmin(seasonal_factors) + 1
                    
                    print(f"Strongest seasonal month: {strongest_month}")
                    print(f"Weakest seasonal month: {weakest_month}")
                    
                except Exception as e:
                    print(f"Could not perform seasonal decomposition: {str(e)}")
            else:
                print("Not enough data for seasonal decomposition (need at least 24 months)")
            
            # 3. ARIMA Forecasting
            if len(monthly_views) >= 12:  # Need at least 12 data points
                try:
                    # Determine order through ACF and PACF plots
                    plt.figure(figsize=(12, 6))
                    plt.subplot(121)
                    plot_acf(monthly_views['Hours Viewed'], ax=plt.gca(), lags=20)
                    plt.subplot(122)
                    plot_pacf(monthly_views['Hours Viewed'], ax=plt.gca(), lags=20)
                    plt.tight_layout()
                    self.save_fig(plt, 'acf_pacf_plots.png')
                    plt.close()
                    
                    # Fit ARIMA model - starting with a simple (1,1,1) model
                    # In practice, you'd use auto_arima or grid search for optimal parameters
                    model = ARIMA(monthly_views['Hours Viewed'], order=(1, 1, 1))
                    model_fit = model.fit()
                    
                    # Forecast next 6 months
                    forecast_steps = 6
                    forecast = model_fit.forecast(steps=forecast_steps)
                    
                    # Create forecast index
                    last_date = monthly_views.index[-1]
                    forecast_index = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=forecast_steps, freq='MS')
                    forecast_series = pd.Series(forecast, index=forecast_index)
                    
                    # Plot the forecast
                    plt.figure(figsize=(14, 7))
                    plt.plot(monthly_views.index, monthly_views['Hours Viewed'], label='Historical')
                    plt.plot(forecast_index, forecast, color='red', label='Forecast')
                    plt.fill_between(forecast_index, 
                                    [x - x*0.1 for x in forecast], 
                                    [x + x*0.1 for x in forecast], 
                                    color='red', alpha=0.2)
                    plt.title('Viewership Forecast (Next 6 Months)', fontsize=16)
                    plt.ylabel('Hours Viewed (Millions)', fontsize=14)
                    plt.xlabel('Date', fontsize=14)
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    self.save_fig(plt, 'viewership_forecast.png')
                    plt.close()
                    
                    # Store forecast
                    self.results['time_series']['forecast'] = {
                        str(date.date()): float(value) for date, value in zip(forecast_index, forecast)
                    }
                    
                    print(f"Forecast for next {forecast_steps} months generated")
                except Exception as e:
                    print(f"Could not perform ARIMA forecasting: {str(e)}")
            
            # 4. Release Date Optimization
            # Group by day of week and calculate average viewership
            day_performance = self.clean_df.groupby('Day of Week')['Hours Viewed'].mean().reindex([
                'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
            ])
            
            plt.figure(figsize=(12, 6))
            ax = day_performance.plot(kind='bar', color=self.color_palette)
            plt.title('Average Viewership by Release Day', fontsize=16)
            plt.ylabel('Average Hours Viewed (Millions)', fontsize=14)
            plt.xlabel('Day of Week', fontsize=14)
            
            # Add data labels
            for i, v in enumerate(day_performance):
                ax.text(i, v + 0.05, f'{v:.2f}M', ha='center', fontsize=10)
                
            plt.tight_layout()
            self.save_fig(plt, 'day_of_week_optimization.png')
            plt.close()
            
            # Best day to release
            best_day = day_performance.idxmax()
            self.results['time_series']['optimal_release_day'] = best_day
            print(f"Optimal day for content release: {best_day}")
            
            # 5. Monthly performance heat map
            month_content_views = pd.pivot_table(
                self.clean_df, 
                values='Hours Viewed',
                index='Month', 
                columns='Content Type',
                aggfunc='mean'
            ).fillna(0)
            
            plt.figure(figsize=(14, 8))
            sns.heatmap(month_content_views, annot=True, fmt='.2f', cmap='viridis')
            plt.title('Average Hours Viewed by Month and Content Type', fontsize=16)
            plt.tight_layout()
            self.save_fig(plt, 'month_content_heatmap.png')
            plt.close()
            
            print("Time series analysis complete")
            
        except Exception as e:
            print(f"Error in time series analysis: {str(e)}")
    
    # SECTION 2: CONTENT PERFORMANCE DEEP DIVES
    def analyze_content_performance(self):
        """Analyze detailed content performance metrics"""
        if self.clean_df is None:
            print("No processed data available")
            return
            
        print("\n===== ANALYZING CONTENT PERFORMANCE =====")
        
        # 1. Performance tier distribution
        try:
            tier_dist = self.clean_df['Performance Tier'].value_counts(normalize=True)
            
            plt.figure(figsize=(12, 6))
            ax = tier_dist.plot(kind='bar', color=self.color_palette)
            
            # Add percentages
            for i, v in enumerate(tier_dist):
                ax.text(i, v + 0.01, f'{v:.1%}', ha='center', fontsize=10)
                
            plt.title('Content Performance Tier Distribution', fontsize=16)
            plt.ylabel('Proportion', fontsize=14)
            plt.xlabel('Performance Tier', fontsize=14)  
            plt.tight_layout()
            self.save_fig(plt, 'performance_tier_distribution.png')
            plt.close()
            
            # 2. Performance by content type
            type_performance = self.clean_df.groupby('Content Type')['Hours Viewed'].agg(['mean', 'median', 'std']).reset_index()
            
            plt.figure(figsize=(12, 6))
            ax = sns.barplot(x='Content Type', y='mean', data=type_performance, palette=self.color_palette)
            
            # Add confidence intervals
            for i, row in type_performance.iterrows():
                ax.errorbar(i, row['mean'], yerr=row['std']/2, fmt='none', color='black', capsize=5)
                
            plt.title('Average Viewership by Content Type (with Std Dev)', fontsize=16)
            plt.ylabel('Average Hours Viewed (Millions)', fontsize=14)
            plt.xlabel('Content Type', fontsize=14)
            plt.tight_layout()
            self.save_fig(plt, 'content_type_performance.png')
            plt.close()
            
            # 3. Content Longevity Analysis
            # Group content into age buckets based on days since release
            self.clean_df['Age Bucket'] = pd.cut(
                self.clean_df['Days Since Release'],
                bins=[0, 30, 90, 180, 365, float('inf')],
                labels=['1 Month', '1-3 Months', '3-6 Months', '6-12 Months', '1+ Years']
            )
            
            age_performance = self.clean_df.groupby(['Age Bucket', 'Content Type'])['Hours Viewed'].mean().unstack()
            
            plt.figure(figsize=(14, 8))
            ax = age_performance.plot(kind='bar', figsize=(14, 8))
            plt.title('Content Performance by Age', fontsize=16)
            plt.ylabel('Average Hours Viewed (Millions)', fontsize=14)
            plt.xlabel('Content Age', fontsize=14)
            plt.legend(title='Content Type')
            plt.tight_layout()
            self.save_fig(plt, 'content_longevity.png')
            plt.close()
            
            # Calculate decay rate (how much viewership drops over time)
            try:
                decay_rates = {}
                for content_type in age_performance.columns:
                    if (content_type in age_performance.columns and 
                        '1 Month' in age_performance.index and 
                        '6-12 Months' in age_performance.index):
                        
                        initial = age_performance.loc['1 Month', content_type]
                        later = age_performance.loc['6-12 Months', content_type]
                        
                        if initial > 0:  # Avoid division by zero
                            decay = (initial - later) / initial
                            decay_rates[content_type] = decay
                
                self.results['content_performance'] = {
                    'decay_rates': decay_rates
                }
                
                print("Content decay rates (% viewership lost from 1 month to 6-12 months):")
                for content_type, rate in decay_rates.items():
                    print(f"  {content_type}: {rate:.1%}")
            except Exception as e:
                print(f"Could not calculate decay rates: {str(e)}")
            
            # 4. Runtime Impact Analysis
            # This requires a column with content runtime/episode count
            # Simulating with title length as a proxy (in real data, use actual runtime)
            if 'Runtime' in self.clean_df.columns:
                runtime_col = 'Runtime'
            else:
                # Use title length as a proxy
                runtime_col = 'Title Length'
                print("Note: Using title length as proxy for runtime (ideally use actual runtime)")
            
            # Create runtime buckets
            self.clean_df['Runtime Bucket'] = pd.qcut(
                self.clean_df[runtime_col], 
                q=5, 
                labels=['Very Short', 'Short', 'Medium', 'Long', 'Very Long']
            )
            
            runtime_performance = self.clean_df.groupby('Runtime Bucket')['Hours Viewed'].mean()
            
            plt.figure(figsize=(12, 6))
            ax = runtime_performance.plot(kind='bar', color=self.color_palette)
            
            # Add data labels
            for i, v in enumerate(runtime_performance):
                ax.text(i, v + 0.05, f'{v:.2f}M', ha='center', fontsize=10)
                
            plt.title(f'Impact of {runtime_col} on Viewership', fontsize=16)
            plt.ylabel('Average Hours Viewed (Millions)', fontsize=14)
            plt.xlabel('Runtime Category', fontsize=14)
            plt.tight_layout()
            self.save_fig(plt, 'runtime_impact.png')
            plt.close()
            
            # Find optimal runtime
            optimal_runtime = runtime_performance.idxmax()
            self.results['content_performance']['optimal_runtime'] = str(optimal_runtime)
            print(f"Optimal content {runtime_col} category: {optimal_runtime}")
            
            # 5. Genre Performance Analysis
            genre_performance = self.clean_df.groupby('Inferred Genre')['Hours Viewed'].agg(['count', 'mean'])
            genre_performance = genre_performance.sort_values('mean', ascending=False)
            
            # Plot only top 10 genres for readability
            top_genres = genre_performance.head(10).reset_index()
            
            plt.figure(figsize=(14, 7))
            ax = sns.barplot(x='Inferred Genre', y='mean', data=top_genres, palette=self.color_palette)
            
            # Add count annotations
            for i, row in top_genres.iterrows():
                ax.text(i, row['mean'] + 0.05, f'n={int(row["count"])}', ha='center', fontsize=9)
            
            plt.title('Performance by Genre (Top 10)', fontsize=16)
            plt.ylabel('Average Hours Viewed (Millions)', fontsize=14)
            plt.xlabel('Genre', fontsize=14)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            self.save_fig(plt, 'genre_performance.png')
            plt.close()
            
            # Store top performing genres
            top_genres_dict = genre_performance.head(5)['mean'].to_dict()
            self.results['content_performance']['top_genres'] = {
                k: float(v) for k, v in top_genres_dict.items()
            }
            
            print("Content performance analysis complete")
            
        except Exception as e:
            print(f"Error in content performance analysis: {str(e)}")
            
    # SECTION 3: CLUSTERING & SEGMENTATION
    def perform_cluster_analysis(self):
        """Perform cluster analysis to identify content segments"""
        if self.clean_df is None:
            print("No processed data available")
            return
            
        print("\n===== PERFORMING CLUSTER ANALYSIS =====")
        
        try:
            # Select numerical features for clustering
            cluster_features = [
                'Hours Viewed', 
                'Title Length', 
                'Days Since Release'
            ]
            
            # Add year if available
            if 'Year' in self.clean_df.columns:
                cluster_features.append('Year')
            
            # Check if we have at least 100 rows (for meaningful clustering)
            if len(self.clean_df) < 100:
                print("Not enough data for clustering analysis")
                return
            
            # Create dataset for clustering (dropping missing values)
            cluster_data = self.clean_df[cluster_features].dropna()
            
            # Standardize the data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(cluster_data)
            
            # Find optimal number of clusters using elbow method
            inertia = []
            k_range = range(2, 11)
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(scaled_data)
                inertia.append(kmeans.inertia_)
            
            # Plot elbow curve
            plt.figure(figsize=(10, 6))
            plt.plot(k_range, inertia, 'o-')
            plt.title('Elbow Method for Optimal k', fontsize=16)
            plt.xlabel('Number of clusters', fontsize=14)
            plt.ylabel('Inertia', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            self.save_fig(plt, 'elbow_method.png')
            plt.close()
            
            # Choose optimal k (would be automated in production)
            # For now, use k=4 as a reasonable default
            optimal_k = 4
            
            # Apply KMeans with optimal k
            kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(scaled_data)
            
            # Add cluster labels back to original data
            cluster_df = cluster_data.copy()
            cluster_df['Cluster'] = clusters
            
            # PCA for visualization
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(scaled_data)
            
            # Create PCA dataframe
            pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
            pca_df['Cluster'] = clusters
            
            # Visualize clusters
            plt.figure(figsize=(12, 8))
            sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_df, palette='viridis', s=50)
            plt.title('Content Clusters Visualization', fontsize=16)
            plt.xlabel('Principal Component 1', fontsize=14)
            plt.ylabel('Principal Component 2', fontsize=14)
            plt.tight_layout()
            self.save_fig(plt, 'content_clusters.png')
            plt.close()
            
            # Analyze cluster characteristics
            cluster_profiles = cluster_df.groupby('Cluster').mean()
            
            # Calculate z-scores for better interpretation
            cluster_profiles_z = (cluster_profiles - cluster_profiles.mean()) / cluster_profiles.std()
            
            # Visualize cluster profiles
            plt.figure(figsize=(14, 8))
            sns.heatmap(cluster_profiles_z, cmap='RdBu_r', annot=True, fmt='.2f')
            plt.title('Cluster Profiles (Z-scores)', fontsize=16)
            plt.tight_layout()
            self.save_fig(plt, 'cluster_profiles.png')
            plt.close()
            
            # Get cluster sizes
            cluster_sizes = cluster_df['Cluster'].value_counts().sort_index()
            
            plt.figure(figsize=(10, 6))
            ax = cluster_sizes.plot(kind='bar', color=self.color_palette)
            
            # Add percentages
            for i, v in enumerate(cluster_sizes):
                ax.text(i, v + 3, f'{v/sum(cluster_sizes):.1%}', ha='center', fontsize=10)
                
            plt.title('Cluster Sizes', fontsize=16)
            plt.ylabel('Count', fontsize=14)
            plt.xlabel('Cluster', fontsize=14)
            plt.tight_layout()
            self.save_fig(plt, 'cluster_sizes.png')
            plt.close()
            
            # Detailed analysis of each cluster
            cluster_descriptions = {}
            for cluster in range(optimal_k):
                profile = cluster_profiles.loc[cluster]
                profile_z = cluster_profiles_z.loc[cluster]
                
                # Identify distinguishing features
                distinguishing = profile_z.sort_values(ascending=False)
                
                # Create cluster description
                if profile['Hours Viewed'] > cluster_profiles['Hours Viewed'].mean():
                    performance = "High performing"
                else:
                    performance = "Underperforming"
                    
                # Age description
                if 'Days Since Release' in profile:
                    if profile['Days Since Release'] > cluster_profiles['Days Since Release'].mean():
                        age = "older"
                    else:
                        age = "newer"
                else:
                    age = "unknown age"
                    
                description = f"{performance} {age} content"
                
                cluster_descriptions[f"Cluster {cluster}"] = {
                    "description": description,
                    "size": int(cluster_sizes[cluster]),
                    "percent": float(cluster_sizes[cluster]/sum(cluster_sizes)),
                    "avg_hours_viewed": float(profile['Hours Viewed']),
                    "key_features": distinguishing.to_dict()
                }
            
            # Store cluster results
            self.results['clustering'] = {
                'optimal_clusters': optimal_k,
                'cluster_profiles': cluster_descriptions
            }
            
            # Print cluster insights
            print(f"Identified {optimal_k} distinct content clusters:")
            for cluster, details in cluster_descriptions.items():
                print(f"  {cluster}: {details['description']} ({details['percent']:.1%} of content)")
                
            print("Cluster analysis complete")
            
        except Exception as e:
            print(f"Error in cluster analysis: {str(e)}")
    
    # SECTION 4: NATURAL LANGUAGE PROCESSING ON TITLES
    def analyze_titles_with_nlp(self):
        """Analyze content titles using natural language processing"""
        if self.clean_df is None:
            print("No processed data available")
            return
            
        print("\n===== ANALYZING TITLES WITH NLP =====")
        
        try:
            # Process titles using NLP
            titles = self.clean_df['Title'].dropna().astype(str).tolist()
            
            if len(titles) < 10:
                print("Not enough title data for NLP analysis")
                return
                
            # 1. Word frequency analysis
            all_words = ' '.join(titles).lower()
            
            # Create stop words
            stop_words = set(stopwords.words('english'))
            additional_stops = {'season', 'part', 'episode', 'vol', 'volume', 'new', 'series'}
            stop_words.update(additional_stops)
            
            # Tokenize and filter
            word_tokens = word_tokenize(all_words)
            filtered_words = [word for word in word_tokens if word.isalpha() and word not in stop_words]
            
            # Get word frequencies
            word_freq = Counter(filtered_words)
            common_words = word_freq.most_common(20)
            
            # Create dataframe for plotting
            word_df = pd.DataFrame(common_words, columns=['Word', 'Frequency'])
            
            plt.figure(figsize=(12, 6))
            ax = sns.barplot(x='Frequency', y='Word', data=word_df, palette=self.color_palette)
            plt.title('Most Common Words in Titles', fontsize=16)
            plt.xlabel('Frequency', fontsize=14)
            plt.ylabel('Word', fontsize=14)
            plt.tight_layout()
            self.save_fig(plt, 'common_title_words.png')
            plt.close()
            
            # 2. Word Cloud
            try:
                wordcloud = WordCloud(
                    width=800, height=400,
                    background_color='white',
                    max_words=100,
                    colormap='viridis',
                    contour_width=1
                ).generate(' '.join(filtered_words))
                
                plt.figure(figsize=(12, 6))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.tight_layout()
                self.save_fig(plt, 'title_wordcloud.png')
                plt.close()
            except Exception as e:
                print(f"Could not generate word cloud: {str(e)}")
            
            # 3. Sentiment Analysis
            try:
                sia = SentimentIntensityAnalyzer()
                
                # Calculate sentiment for each title
                self.clean_df['Title Sentiment'] = self.clean_df['Title'].apply(
                    lambda x: sia.polarity_scores(str(x))['compound'] if pd.notnull(x) else 0
                )
                
                # Group sentiment scores into categories
                self.clean_df['Sentiment Category'] = pd.cut(
                    self.clean_df['Title Sentiment'],
                    bins=[-1, -0.25, 0.25, 1],
                    labels=['Negative', 'Neutral', 'Positive']
                )
                
                # Analyze sentiment distribution
                sentiment_dist = self.clean_df['Sentiment Category'].value_counts(normalize=True)
                
                plt.figure(figsize=(10, 6))
                ax = sentiment_dist.plot.pie(autopct='%1.1f%%', colors=self.color_palette, startangle=90)
                plt.title('Title Sentiment Distribution', fontsize=16)
                plt.ylabel('')  # Hide ylabel
                plt.tight_layout()
                self.save_fig(plt, 'title_sentiment_distribution.png')
                plt.close()
                
                # Compare performance by sentiment
                sentiment_perf = self.clean_df.groupby('Sentiment Category')['Hours Viewed'].mean()
                
                plt.figure(figsize=(10, 6))
                ax = sentiment_perf.plot(kind='bar', color=self.color_palette)
                
                # Add data labels
                for i, v in enumerate(sentiment_perf):
                    ax.text(i, v + 0.05, f'{v:.2f}M', ha='center')
                    
                plt.title('Performance by Title Sentiment', fontsize=16)
                plt.xlabel('Sentiment Category', fontsize=14)
                plt.ylabel('Average Hours Viewed (Millions)', fontsize=14)
                plt.tight_layout()
                self.save_fig(plt, 'sentiment_performance.png')
                plt.close()
                
                # Store sentiment results
                self.results['nlp_analysis'] = {
                    'sentiment_distribution': sentiment_dist.to_dict(),
                    'sentiment_performance': sentiment_perf.to_dict(),
                    'top_words': dict(common_words)
                }
                
                # Calculate best-performing sentiment
                best_sentiment = sentiment_perf.idxmax()
                print(f"Best performing title sentiment: {best_sentiment}")
                
            except Exception as e:
                print(f"Could not perform sentiment analysis: {str(e)}")
                
            print("Title NLP analysis complete")
            
        except Exception as e:
            print(f"Error in NLP analysis: {str(e)}")
    
    # SECTION 5: GEOGRAPHICAL MARKET ANALYSIS
    def analyze_geographical_performance(self):
        """Analyze content performance across different geographical markets"""
        if self.clean_df is None:
            print("No processed data available")
            return
            
        print("\n===== ANALYZING GEOGRAPHICAL PERFORMANCE =====")
        
        try:
            # Check if we have region data
            if 'Region' not in self.clean_df.columns or self.clean_df['Region'].isnull().sum() > len(self.clean_df) * 0.5:
                print("Not enough region data for geographical analysis")
                return
                
            # 1. Regional Performance Overview
            region_perf = self.clean_df.groupby('Region')['Hours Viewed'].agg(['mean', 'count']).reset_index()
            region_perf = region_perf.sort_values('mean', ascending=False)
            
            # Filter to top 10 regions
            top_regions = region_perf.head(10)
            
            plt.figure(figsize=(14, 7))
            ax = sns.barplot(x='Region', y='mean', data=top_regions, palette=self.color_palette)
            
            # Add count annotations
            for i, row in top_regions.iterrows():
                ax.text(i, row['mean'] + 0.05, f'n={int(row["count"])}', ha='center', fontsize=9)
                
            plt.title('Average Viewership by Region (Top 10)', fontsize=16)
            plt.ylabel('Average Hours Viewed (Millions)', fontsize=14)
            plt.xlabel('Region', fontsize=14)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            self.save_fig(plt, 'region_performance.png')
            plt.close()
            
            # 2. Content Type by Region
            type_region_data = pd.pivot_table(
                self.clean_df,
                values='Hours Viewed',
                index='Region',
                columns='Content Type',
                aggfunc='mean'
            ).fillna(0)
            
            # Select top 8 regions for readability
            type_region_data = type_region_data.loc[region_perf['Region'][:8]]
            
            plt.figure(figsize=(14, 8))
            sns.heatmap(type_region_data, annot=True, fmt='.2f', cmap='viridis')
            plt.title('Content Type Performance by Region', fontsize=16)
            plt.tight_layout()
            self.save_fig(plt, 'content_type_by_region.png')
            plt.close()
            
            # 3. Genre Preferences by Region
            # Get top genre for each region
            region_genres = {}
            for region in region_perf['Region'][:8]:  # Top 8 regions
                region_data = self.clean_df[self.clean_df['Region'] == region]
                genre_perf = region_data.groupby('Inferred Genre')['Hours Viewed'].mean()
                
                if not genre_perf.empty:
                    top_genre = genre_perf.idxmax()
                    region_genres[region] = {
                        'top_genre': top_genre,
                        'avg_hours': float(genre_perf.max())
                    }
            
            # Store regional data
            self.results['geographical'] = {
                'top_markets': region_perf.head(5)[['Region', 'mean']].set_index('Region')['mean'].to_dict(),
                'regional_preferences': region_genres
            }
            
            # Print regional insights
            print("Top performing regions:")
            for i, row in region_perf.head(5).iterrows():
                print(f"  {row['Region']}: {row['mean']:.2f}M hours (n={int(row['count'])})")
                
            print("\nRegional genre preferences:")
            for region, data in region_genres.items():
                print(f"  {region}: {data['top_genre']} ({data['avg_hours']:.2f}M hours)")
                
            print("Geographical analysis complete")
            
        except Exception as e:
            print(f"Error in geographical analysis: {str(e)}")
    
    # SECTION 6: BUSINESS STRATEGY & RECOMMENDATIONS
    def generate_business_recommendations(self):
        """Generate strategic business recommendations based on all analyses"""
        if self.clean_df is None or not self.results:
            print("No analyzed data available for recommendations")
            return
            
        print("\n===== GENERATING BUSINESS RECOMMENDATIONS =====")
        
        try:
            recommendations = []
            
            # 1. Content Investment Recommendations
            print("Developing content investment recommendations...")
            
            # Top performing genres
            if 'content_performance' in self.results and 'top_genres' in self.results['content_performance']:
                top_genres = list(self.results['content_performance']['top_genres'].keys())
                if top_genres:
                    recommendations.append({
                        'category': 'Content Investment',
                        'title': 'Invest in high-performing genres',
                        'description': f"Increase investment in {', '.join(top_genres[:3])} content which show consistently higher viewership.",
                        'expected_impact': 'High',
                        'implementation_complexity': 'Medium'
                    })
            
            # Content length optimization
            if 'content_performance' in self.results and 'optimal_runtime' in self.results['content_performance']:
                optimal_runtime = self.results['content_performance']['optimal_runtime']
                recommendations.append({
                    'category': 'Content Strategy',
                    'title': 'Optimize content length',
                    'description': f"Target {optimal_runtime} content length which shows optimal engagement metrics.",
                    'expected_impact': 'Medium',
                    'implementation_complexity': 'Low'
                })
            
            # 2. Release Strategy Recommendations
            print("Developing release strategy recommendations...")
            
            # Optimal release day
            if 'time_series' in self.results and 'optimal_release_day' in self.results['time_series']:
                best_day = self.results['time_series']['optimal_release_day']
                recommendations.append({
                    'category': 'Release Strategy',
                    'title': f'Prioritize {best_day} releases',
                    'description': f"Content released on {best_day} consistently achieves higher viewership.",
                    'expected_impact': 'Medium',
                    'implementation_complexity': 'Low'
                })
            
            # Seasonal strategy
            if 'time_series' in self.results and 'seasonal_factors' in self.results['time_series']:
                recommendations.append({
                    'category': 'Release Strategy',
                    'title': 'Implement seasonal content strategy',
                    'description': "Align major releases with identified seasonal peaks in viewership patterns.",
                    'expected_impact': 'Medium',
                    'implementation_complexity': 'Medium'
                })
            
            # 3. Market-specific Recommendations
            print("Developing market-specific recommendations...")
            
            if 'geographical' in self.results and 'regional_preferences' in self.results['geographical']:
                region_prefs = self.results['geographical']['regional_preferences']
                
                # Get top 3 markets by potential
                if 'top_markets' in self.results['geographical']:
                    top_markets = list(self.results['geographical']['top_markets'].keys())[:3]
                    
                    if top_markets:
                        recommendations.append({
                            'category': 'Market Strategy',
                            'title': f'Focus on high-potential markets',
                            'description': f"Prioritize content development for {', '.join(top_markets)} markets which show highest engagement.",
                            'expected_impact': 'High',
                            'implementation_complexity': 'High'
                        })
                
                # Market-specific genre recommendations
                market_recs = []
                for region, data in region_prefs.items():
                    market_recs.append(f"{region}: {data['top_genre']}")
                
                if market_recs:
                    recommendations.append({
                        'category': 'Market Strategy',
                        'title': 'Tailor content to regional preferences',
                        'description': "Develop market-specific content based on regional genre preferences: " + "; ".join(market_recs[:3]) + ".",
                        'expected_impact': 'High',
                        'implementation_complexity': 'High'
                    })
            
            # 4. Content Longevity Recommendations
            print("Developing content longevity recommendations...")
            
            if 'content_performance' in self.results and 'decay_rates' in self.results['content_performance']:
                decay_rates = self.results['content_performance']['decay_rates']
                
                # Find content types with lowest decay
                sorted_decay = sorted(decay_rates.items(), key=lambda x: x[1])
                low_decay = [item[0] for item in sorted_decay[:2]]
                
                if low_decay:
                    recommendations.append({
                        'category': 'Content Strategy',
                        'title': 'Invest in evergreen content',
                        'description': f"Increase production of {', '.join(low_decay)} content which shows stronger long-term engagement.",
                        'expected_impact': 'Medium',
                        'implementation_complexity': 'Medium'
                    })
            
            # 5. Title Optimization Recommendations
            print("Developing title optimization recommendations...")
            
            if 'nlp_analysis' in self.results:
                # Sentiment recommendations
                if 'sentiment_performance' in self.results['nlp_analysis']:
                    sentiment_perf = self.results['nlp_analysis']['sentiment_performance']
                    best_sentiment = max(sentiment_perf.items(), key=lambda x: x[1])[0]
                    
                    recommendations.append({
                        'category': 'Marketing',
                        'title': 'Optimize title sentiment',
                        'description': f"Favor {best_sentiment.lower()} title sentiment which correlates with higher viewership.",
                        'expected_impact': 'Low',
                        'implementation_complexity': 'Low'
                    })
                
                # Word usage recommendations
                if 'top_words' in self.results['nlp_analysis']:
                    top_words = list(self.results['nlp_analysis']['top_words'].keys())[:5]
                    
                    if top_words:
                        recommendations.append({
                            'category': 'Marketing',
                            'title': 'Use high-impact keywords in titles',
                            'description': f"Incorporate popular keywords like {', '.join(top_words)} in content titles.",
                            'expected_impact': 'Low',
                            'implementation_complexity': 'Low'
                        })
            
            # 6. Algorithm-based Recommendations
            print("Developing algorithm-based recommendations...")
            
            if 'clustering' in self.results and 'cluster_profiles' in self.results['clustering']:
                cluster_profiles = self.results['clustering']['cluster_profiles']
                
                # Find the highest performing cluster
                best_cluster = None
                best_hours = 0
                
                for cluster, details in cluster_profiles.items():
                    if details['avg_hours_viewed'] > best_hours:
                        best_hours = details['avg_hours_viewed']
                        best_cluster = cluster
                
                if best_cluster:
                    recommendations.append({
                        'category': 'Content Strategy',
                        'title': 'Replicate successful content patterns',
                        'description': f"Create more content similar to {cluster_profiles[best_cluster]['description']} cluster which shows highest engagement.",
                        'expected_impact': 'High',
                        'implementation_complexity': 'Medium'
                    })
            
            # Store recommendations
            self.results['recommendations'] = recommendations
            
            # Print recommendations summary
            print(f"\nGenerated {len(recommendations)} strategic recommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec['category']} - {rec['title']}")
                print(f"     Impact: {rec['expected_impact']}, Complexity: {rec['implementation_complexity']}")
                print(f"     {rec['description']}")
                print()
                
            # Generate summary report
            self.generate_summary_report()
            
            print("Business recommendations generated")
            
        except Exception as e:
            print(f"Error generating recommendations: {str(e)}")
    
    # SECTION 7: PERFORMANCE OPTIMIZATIONS
    def optimize_performance(self):
        """Optimize performance for large datasets"""
        print("\n===== OPTIMIZING PERFORMANCE =====")
        
        try:
            # 1. Use efficient data structures
            # Convert string columns to categorical for memory efficiency
            string_columns = self.clean_df.select_dtypes(include=['object']).columns
            for col in string_columns:
                self.clean_df[col] = self.clean_df[col].astype('category')
            
            # 2. Implement parallel processing for heavy computations
            from multiprocessing import Pool, cpu_count
            
            # Example: Parallel sentiment analysis
            def process_chunk(chunk):
                sia = SentimentIntensityAnalyzer()
                chunk['Title Sentiment'] = chunk['Title'].apply(
                    lambda x: sia.polarity_scores(str(x))['compound'] if pd.notnull(x) else 0
                )
                return chunk
            
            # Split dataframe into chunks for parallel processing
            def parallelize_dataframe(df, func, n_cores=None):
                n_cores = n_cores or max(1, cpu_count() - 1)
                df_split = np.array_split(df, n_cores)
                pool = Pool(n_cores)
                df = pd.concat(pool.map(func, df_split))
                pool.close()
                pool.join()
                return df
            
            # Example usage:
            # self.clean_df = parallelize_dataframe(self.clean_df, process_chunk)
            
            # 3. Implement lazy loading for large datasets
            def lazy_load_chunks(file_path, chunksize=100000):
                for chunk in pd.read_csv(file_path, chunksize=chunksize):
                    yield chunk
            
            # 4. Optimize memory usage
            import gc
            gc.collect()  # Force garbage collection
            
            print("Performance optimizations applied")
            
        except Exception as e:
            print(f"Error in performance optimization: {str(e)}")
    
    # SECTION 8: ADVANCED MACHINE LEARNING
    def implement_advanced_ml(self):
        """Implement advanced machine learning models"""
        if self.clean_df is None:
            print("No processed data available")
            return
            
        print("\n===== IMPLEMENTING ADVANCED ML MODELS =====")
        
        try:
            # 1. Content recommendation engine
            from sklearn.metrics.pairwise import cosine_similarity
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            # Create content features
            content_features = self.clean_df['Title'] + ' ' + self.clean_df['Inferred Genre'] + ' ' + self.clean_df['Content Type']
            
            # Create TF-IDF matrix
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(content_features)
            
            # Calculate cosine similarity
            cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
            
            # Function to get content recommendations
            def get_recommendations(title_idx, cosine_sim=cosine_sim):
                # Get similarity scores
                sim_scores = list(enumerate(cosine_sim[title_idx]))
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
                sim_scores = sim_scores[1:11]  # Top 10 similar items
                content_indices = [i[0] for i in sim_scores]
                return self.clean_df.iloc[content_indices]
            
            # Example: Get recommendations for the top performing content
            try:
                top_idx = self.clean_df['Hours Viewed'].argmax()
                top_title = self.clean_df.iloc[top_idx]['Title']
                similar_content = get_recommendations(top_idx)
                
                # Store recommendations
                self.results['recommendation_engine'] = {
                    'reference_title': top_title,
                    'similar_titles': similar_content['Title'].tolist()
                }
                
                print(f"Content recommendation engine built. Example for '{top_title}':")
                print(similar_content['Title'].tolist()[:3])
            except:
                print("Could not generate sample recommendations")
            
            # 2. Predictive modeling for content performance
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import OneHotEncoder, StandardScaler
            
            print("\nBuilding performance prediction model...")
            
            # Prepare features
            try:
                # Create feature set
                features = ['Content Type', 'Year', 'Month', 'Season', 'Inferred Genre']
                numeric_features = ['Title Length', 'Days Since Release']
                
                # Filter for required columns and drop missing values
                model_df = self.clean_df[features + numeric_features + ['Hours Viewed']].dropna()
                
                if len(model_df) > 100:  # Ensure sufficient data
                    # One-hot encode categorical features
                    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
                    encoded_cats = encoder.fit_transform(model_df[features])
                    
                    # Scale numeric features
                    scaler = StandardScaler()
                    scaled_nums = scaler.fit_transform(model_df[numeric_features])
                    
                    # Combine features
                    X = np.hstack([encoded_cats, scaled_nums])
                    y = model_df['Hours Viewed']
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )
                    
                    # Train model
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                    model.fit(X_train, y_train)
                    
                    # Evaluate model
                    test_score = model.score(X_test, y_test)
                    
                    # Store model performance
                    self.results['ml_models'] = {
                        'performance_prediction': {
                            'algorithm': 'Random Forest',
                            'r2_score': float(test_score),
                            'feature_importance': dict(
                                zip(encoder.get_feature_names_out().tolist() + numeric_features,
                                    model.feature_importances_)
                            )
                        }
                    }
                    
                    print(f"Performance prediction model built (R² score: {test_score:.2f})")
                    
                    # Plot feature importance
                    feature_names = encoder.get_feature_names_out().tolist() + numeric_features
                    importance = model.feature_importances_
                    
                    # Sort features by importance
                    indices = np.argsort(importance)[-10:]  # Top 10 features
                    
                    plt.figure(figsize=(12, 8))
                    plt.barh(range(len(indices)), importance[indices], align='center')
                    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
                    plt.xlabel('Feature Importance')
                    plt.title('Top Features for Predicting Content Performance')
                    plt.tight_layout()
                    self.save_fig(plt, 'feature_importance.png')
                    plt.close()
                    
                else:
                    print("Not enough data for predictive modeling")
                    
            except Exception as e:
                print(f"Could not build prediction model: {str(e)}")
            
            print("Advanced ML models implementation complete")
            
        except Exception as e:
            print(f"Error in advanced ML implementation: {str(e)}")
    
    # SECTION 9: INTERACTIVE DASHBOARDS
    def create_interactive_dashboard(self):
        """Create interactive dashboards for the analysis results"""
        if not self.results:
            print("No analysis results available for dashboard")
            return
            
        print("\n===== CREATING INTERACTIVE DASHBOARD =====")
        
        try:
            import plotly.express as px
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            import dash
            from dash import dcc, html
            from flask import Flask
            
            print("Setting up dashboard components...")
            
            # 1. Content overview
            if self.clean_df is not None:
                # Content distribution
                fig1 = px.pie(
                    self.clean_df, 
                    names='Content Type', 
                    title='Content Type Distribution',
                    hole=0.4
                )
                fig1.update_layout(margin=dict(t=50, b=50, l=50, r=50))
                fig1.write_html('netflix_analysis_output/content_distribution.html')
                
                # Performance tiers
                fig2 = px.bar(
                    self.clean_df['Performance Tier'].value_counts().reset_index(),
                    x='index',
                    y='Performance Tier',
                    title='Content Performance Tiers',
                    labels={'index': 'Performance Tier', 'Performance Tier': 'Count'}
                )
                fig2.write_html('netflix_analysis_output/performance_tiers.html')
                
                # Genre distribution
                fig3 = px.bar(
                    self.clean_df['Inferred Genre'].value_counts().head(10).reset_index(),
                    x='index', 
                    y='Inferred Genre',
                    title='Top Genres',
                    labels={'index': 'Genre', 'Inferred Genre': 'Count'}
                )
                fig3.update_layout(xaxis={'categoryorder':'total descending'})
                fig3.write_html('netflix_analysis_output/genre_distribution.html')
                
                print("Created content overview visualizations")
            
            # 2. Time series dashboard
            if 'time_series' in self.results:
                # Monthly viewership with forecast
                try:
                    self.clean_df['YearMonth'] = pd.to_datetime(
                        self.clean_df['Release Date']
                    ).dt.to_period('M').dt.to_timestamp()
                    monthly_views = self.clean_df.groupby('YearMonth')['Hours Viewed'].sum().reset_index()
                    
                    fig = go.Figure()
                    
                    # Historical data
                    fig.add_trace(go.Scatter(
                        x=monthly_views['YearMonth'], 
                        y=monthly_views['Hours Viewed'],
                        mode='lines+markers',
                        name='Historical'
                    ))
                    
                    # Add forecast if available
                    if 'forecast' in self.results['time_series']:
                        forecast_dates = list(self.results['time_series']['forecast'].keys())
                        forecast_values = list(self.results['time_series']['forecast'].values())
                        
                        fig.add_trace(go.Scatter(
                            x=forecast_dates,
                            y=forecast_values,
                            mode='lines+markers',
                            name='Forecast',
                            line=dict(color='red', dash='dash')
                        ))
                    
                    fig.update_layout(
                        title='Monthly Viewership with Forecast',
                        xaxis_title='Date',
                        yaxis_title='Hours Viewed (Millions)',
                        hovermode='x unified'
                    )
                    
                    fig.write_html('netflix_analysis_output/viewership_forecast_interactive.html')
                    print("Created time series dashboard")
                    
                except Exception as e:
                    print(f"Could not create time series visualization: {str(e)}")
            
            # 3. Regional performance map
            if self.clean_df is not None and 'Region' in self.clean_df.columns:
                try:
                    # Map regions to countries for visualization
                    region_map = {
                        'North America/UK': ['United States', 'Canada', 'United Kingdom'],
                        'Latin America/Spain': ['Mexico', 'Brazil', 'Spain', 'Argentina'],
                        'South Korea': ['South Korea'],
                        'Japan': ['Japan'],
                        'India': ['India'],
                        'France/Canada': ['France'],
                        'Germany/Austria': ['Germany'],
                        'Middle East': ['Saudi Arabia', 'United Arab Emirates'],
                        'China/Taiwan': ['China', 'Taiwan'],
                        'Russia': ['Russia']
                    }
                    
                    # Create country performance data
                    country_data = []
                    
                    for region, countries in region_map.items():
                        region_avg = self.clean_df[self.clean_df['Region'] == region]['Hours Viewed'].mean()
                        if not pd.isna(region_avg):
                            for country in countries:
                                country_data.append({
                                    'Country': country,
                                    'Hours Viewed': region_avg
                                })
                    
                    # Create choropleth map
                    if country_data:
                        country_df = pd.DataFrame(country_data)
                        fig = px.choropleth(
                            country_df,
                            locations='Country',
                            locationmode='country names',
                            color='Hours Viewed',
                            hover_name='Country',
                            color_continuous_scale=px.colors.sequential.Viridis,
                            title='Global Content Performance'
                        )
                        
                        fig.update_layout(
                            geo=dict(
                                showframe=False,
                                showcoastlines=False
                            )
                        )
                        
                        fig.write_html('netflix_analysis_output/global_performance_map.html')
                        print("Created global performance map")
                        
                except Exception as e:
                    print(f"Could not create regional map: {str(e)}")
            
            # 4. Generate index.html that links to all dashboards
            dashboard_files = [
                f for f in os.listdir('netflix_analysis_output') 
                if f.endswith('.html')
            ]
            
            if dashboard_files:
                with open('netflix_analysis_output/index.html', 'w') as f:
                    f.write("""
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <title>Netflix Content Analysis Dashboard</title>
                        <style>
                            body { font-family: Arial, sans-serif; margin: 20px; }
                            h1 { color: #E50914; }
                            .dashboard-link { 
                                display: block; 
                                margin: 10px 0; 
                                padding: 15px;
                                background-color: #f4f4f4;
                                text-decoration: none;
                                color: #333;
                                border-radius: 5px;
                            }
                            .dashboard-link:hover {
                                background-color: #e0e0e0;
                            }
                        </style>
                    </head>
                    <body>
                        <h1>Netflix Content Analysis Dashboard</h1>
                        <h2>Interactive Visualizations</h2>
                        <div class="dashboard-links">
                    """)
                    
                    for file in dashboard_files:
                        # Skip index.html itself
                        if file == 'index.html':
                            continue
                        
                        # Create readable name
                        name = file.replace('.html', '').replace('_', ' ').title()
                        f.write(f'<a class="dashboard-link" href="{file}">{name}</a>\n')
                    
                    f.write("""
                        </div>
                    </body>
                    </html>
                    """)
                    
                print(f"Created dashboard index at netflix_analysis_output/index.html")
                print(f"Open this file in a web browser to access all {len(dashboard_files)-1} interactive visualizations")
            
        except ImportError:
            print("Could not create interactive dashboards. Required packages (plotly, dash) not available.")
        except Exception as e:
            print(f"Error creating interactive dashboard: {str(e)}")

    # SECTION 10: ENHANCED DOCUMENTATION
    def generate_documentation(self):
        """Generate comprehensive documentation for the analysis"""
        if not self.results:
            print("No analysis results available for documentation")
            return
            
        print("\n===== GENERATING DOCUMENTATION =====")
        
        try:
            # Create markdown documentation
            doc_path = 'netflix_analysis_output/documentation.md'
            
            with open(doc_path, 'w') as f:
                # Header
                f.write("# Netflix Content Analysis Documentation\n\n")
                
                # Table of Contents
                f.write("## Table of Contents\n")
                f.write("1. [Introduction](#introduction)\n")
                f.write("2. [Data Overview](#data-overview)\n")
                f.write("3. [Methodology](#methodology)\n")
                f.write("4. [Analysis Results](#analysis-results)\n")
                f.write("5. [Business Recommendations](#business-recommendations)\n")
                f.write("6. [Technical Reference](#technical-reference)\n\n")
                
                # Introduction
                f.write("## Introduction\n\n")
                f.write("This document provides comprehensive documentation for the Netflix content analysis framework. ")
                f.write("The analysis examines viewership patterns, content performance metrics, and provides ")
                f.write("data-driven recommendations for content strategy.\n\n")
                
                # Data Overview
                f.write("## Data Overview\n\n")
                if self.clean_df is not None:
                    f.write(f"The analysis is based on {len(self.clean_df)} content items from the Netflix catalog. ")
                    f.write("The dataset includes metrics such as hours viewed, release dates, content types, ")
                    f.write("and derived features such as genres and regional indicators.\n\n")
                    
                    # Data description
                    f.write("### Dataset Structure\n\n")
                    f.write("```\n")
                    buffer = io.StringIO()
                    self.clean_df.info(buf=buffer)
                    f.write(buffer.getvalue())
                    f.write("\n```\n\n")
                    
                    # Sample data
                    f.write("### Sample Data\n\n")
                    f.write("```\n")
                    f.write(self.clean_df.head(5).to_string())
                    f.write("\n```\n\n")
                
                # Methodology
                f.write("## Methodology\n\n")
                f.write("The analysis employs multiple methodologies across several domains:\n\n")
                
                # Time Series
                f.write("### Time Series Analysis\n\n")
                f.write("- Trend decomposition to identify seasonal patterns\n")
                f.write("- ARIMA forecasting for future viewership prediction\n")
                f.write("- Release date optimization analysis\n\n")
                
                # Content Performance
                f.write("### Content Performance Analysis\n\n")
                f.write("- Performance tier classification\n")
                f.write("- Content decay analysis to measure viewership retention\n")
                f.write("- Genre performance comparison\n\n")
                
                # Clustering
                f.write("### Segmentation & Clustering\n\n")
                f.write("- K-means clustering to identify content segments\n")
                f.write("- Principal Component Analysis for dimension reduction\n")
                f.write("- Cluster profiling to identify segment characteristics\n\n")
                
                # NLP
                f.write("### Natural Language Processing\n\n")
                f.write("- Word frequency analysis of titles\n")
                f.write("- Sentiment analysis to measure title sentiment impact\n")
                f.write("- Word cloud visualization of common terms\n\n")
                
                # Geographic Analysis
                f.write("### Geographic Market Analysis\n\n")
                f.write("- Regional performance comparison\n")
                f.write("- Market-specific genre preferences\n")
                f.write("- Content type distribution by region\n\n")
                
                # Results 
                f.write("## Analysis Results\n\n")
                
                # Basic Stats
                if 'basic_stats' in self.results:
                    stats = self.results['basic_stats']
                    f.write("### Overall Performance Metrics\n\n")
                    f.write(f"- Total content items: {stats.get('total_content', 'N/A')}\n")
                    f.write(f"- Total hours viewed: {stats.get('total_hours_viewed', 'N/A'):,.2f} million\n")
                    f.write(f"- Average hours per content: {stats.get('avg_hours_per_content', 'N/A'):,.2f} million\n\n")
                    
                    # Content type distribution
                    if 'content_type_distribution' in stats:
                        f.write("### Content Type Distribution\n\n")
                        f.write("| Content Type | Proportion |\n")
                        f.write("|-------------|------------|\n")
                        
                        for content_type, prop in stats['content_type_distribution'].items():
                            f.write(f"| {content_type} | {prop:.1%} |\n")
                        f.write("\n")
                
                # Time Series
                if 'time_series' in self.results:
                    ts_data = self.results['time_series']
                    f.write("### Time Series Insights\n\n")
                    
                    if 'optimal_release_day' in ts_data:
                        f.write(f"- Optimal release day: **{ts_data['optimal_release_day']}**\n")
                    
                    if 'forecast' in ts_data:
                        f.write("- Viewership forecast for upcoming months:\n")
                        f.write("| Month | Projected Hours (millions) |\n")
                        f.write("|-------|---------------------------|\n")
                        
                        for date, value in ts_data['forecast'].items():
                            f.write(f"| {date} | {value:.2f} |\n")
                        f.write("\n")
                
                # Content Performance
                if 'content_performance' in self.results:
                    perf_data = self.results['content_performance']
                    f.write("### Content Performance Insights\n\n")
                    
                    if 'top_genres' in perf_data:
                        f.write("#### Top Performing Genres\n\n")
                        f.write("| Genre | Average Hours (millions) |\n")
                        f.write("|-------|--------------------------|\n")
                        
                        for genre, hours in perf_data['top_genres'].items():
                            f.write(f"| {genre} | {hours:.2f} |\n")
                        f.write("\n")
                    
                    if 'decay_rates' in perf_data:
                        f.write("#### Content Decay Rates\n\n")
                        f.write("| Content Type | Decay Rate |\n")
                        f.write("|--------------|------------|\n")
                        
                        for content_type, rate in perf_data['decay_rates'].items():
                            f.write(f"| {content_type} | {rate:.1%} |\n")
                        f.write("\n")
                
                # Geographical
                if 'geographical' in self.results:
                    geo_data = self.results['geographical']
                    f.write("### Geographical Insights\n\n")
                    
                    if 'top_markets' in geo_data:
                        f.write("#### Top Markets\n\n")
                        f.write("| Region | Average Hours (millions) |\n")
                        f.write("|--------|--------------------------|\n")
                        
                        for region, hours in geo_data['top_markets'].items():
                            f.write(f"| {region} | {hours:.2f} |\n")
                        f.write("\n")
                    
                    if 'regional_preferences' in geo_data:
                        f.write("#### Regional Genre Preferences\n\n")
                        f.write("| Region | Top Genre | Average Hours (millions) |\n")
                        f.write("|--------|----------|---------------------------|\n")
                        
                        for region, data in geo_data['regional_preferences'].items():
                            f.write(f"| {region} | {data['top_genre']} | {data['avg_hours']:.2f} |\n")
                        f.write("\n")
                
                # Business Recommendations
                if 'recommendations' in self.results:
                    recs = self.results['recommendations']
                    f.write("## Business Recommendations\n\n")
                    
                    # Group recommendations by category
                    categories = {}
                    for rec in recs:
                        cat = rec.get('category', 'Other')
                        if cat not in categories:
                            categories[cat] = []
                        categories[cat].append(rec)
                    
                    for cat, cat_recs in categories.items():
                        f.write(f"### {cat}\n\n")
                        
                        for i, rec in enumerate(cat_recs, 1):
                            f.write(f"**{i}. {rec['title']}**\n\n")
                            f.write(f"{rec['description']}\n\n")
                            f.write(f"- Expected impact: {rec['expected_impact']}\n")
                            f.write(f"- Implementation complexity: {rec['implementation_complexity']}\n\n")
                
                # Technical Reference
                f.write("## Technical Reference\n\n")
                f.write("### Analysis Pipeline\n\n")
                f.write("1. **Data Loading & Preprocessing**\n")
                f.write("   - Missing value handling\n")
                f.write("   - Feature engineering\n")
                f.write("   - Data cleaning\n\n")
                
                f.write("2. **Basic Statistics Generation**\n")
                f.write("   - Overall metrics calculation\n")
                f.write("   - Distribution analysis\n\n")
                
                f.write("3. **Time Series Analysis**\n")
                f.write("   - Trend decomposition\n")
                f.write("   - Forecasting\n")
                f.write("   - Seasonal pattern detection\n\n")
                
                f.write("4. **Content Performance Analysis**\n")
                f.write("   - Performance tier classification\n")
                f.write("   - Runtime impact analysis\n")
                f.write("   - Genre performance comparison\n\n")
                
                f.write("5. **Clustering & Segmentation**\n")
                f.write("   - K-means clustering\n")
                f.write("   - Cluster profiling\n")
                f.write("   - PCA visualization\n\n")
                
                f.write("6. **NLP Title Analysis**\n")
                f.write("   - Word frequency analysis\n")
                f.write("   - Sentiment analysis\n")
                f.write("   - Word cloud generation\n\n")
                
                f.write("7. **Geographical Market Analysis**\n")
                f.write("   - Regional performance comparison\n")
                f.write("   - Market-specific preferences\n\n")
                
                f.write("8. **Business Recommendations Generation**\n")
                f.write("   - Insight synthesis\n")
                f.write("   - Strategy formulation\n")
                f.write("   - Impact assessment\n\n")
                
                # Footer
                f.write("---\n\n")
                f.write("*Documentation generated on " + datetime.now().strftime('%Y-%m-%d') + "*\n")
            
            print(f"Documentation generated at {doc_path}")
            
        except Exception as e:
            print(f"Error generating documentation: {str(e)}")

    # SECTION 11: EXECUTIVE SUMMARY REPORT
    def generate_summary_report(self):
        """Generate a comprehensive summary report of all analysis results"""
        if not self.results:
            print("No results available for summary report")
            return
            
        print("\n===== GENERATING SUMMARY REPORT =====")
        
        try:
            # Create a consolidated report
            report = {
                'title': 'Netflix Content Strategy - Executive Summary',
                'generated_date': datetime.now().strftime('%Y-%m-%d'),
                'dataset_size': len(self.clean_df) if self.clean_df is not None else 0,
                'key_metrics': {},
                'content_strategy': {},
                'market_insights': {},
                'recommendations': self.results.get('recommendations', [])
            }
            
            # Add key metrics
            if 'basic_stats' in self.results:
                report['key_metrics'] = {
                    'total_content': self.results['basic_stats'].get('total_content', 0),
                    'total_hours_viewed': self.results['basic_stats'].get('total_hours_viewed', 0),
                    'avg_hours_per_content': self.results['basic_stats'].get('avg_hours_per_content', 0)
                }
            
            # Add content strategy insights
            content_insights = {}
            
            if 'content_performance' in self.results:
                if 'top_genres' in self.results['content_performance']:
                    content_insights['top_genres'] = self.results['content_performance']['top_genres']
                    
                if 'decay_rates' in self.results['content_performance']:
                    content_insights['content_decay'] = self.results['content_performance']['decay_rates']
                    
                if 'optimal_runtime' in self.results['content_performance']:
                    content_insights['optimal_runtime'] = self.results['content_performance']['optimal_runtime']
            
            if 'time_series' in self.results:
                if 'optimal_release_day' in self.results['time_series']:
                    content_insights['optimal_release_day'] = self.results['time_series']['optimal_release_day']
                    
                if 'seasonal_factors' in self.results['time_series']:
                    content_insights['seasonality'] = self.results['time_series']['seasonal_factors']
            
            report['content_strategy'] = content_insights
            
            # Add market insights
            market_insights = {}
            
            if 'geographical' in self.results:
                if 'top_markets' in self.results['geographical']:
                    market_insights['top_markets'] = self.results['geographical']['top_markets']
                    
                if 'regional_preferences' in self.results['geographical']:
                    market_insights['regional_preferences'] = self.results['geographical']['regional_preferences']
            
            report['market_insights'] = market_insights
            
            # Save report to JSON
            report_path = os.path.join('netflix_analysis_output', 'executive_summary.json')
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=4)
                
            print(f"Summary report saved to {report_path}")
            
        except Exception as e:
            print(f"Error generating summary report: {str(e)}")
    
    # Execute full analysis pipeline
    def run_full_analysis(self):
        """Run the complete analysis pipeline"""
        print("\n===== STARTING NETFLIX CONTENT ANALYSIS =====")
        
        # Step 1: Load and preprocess data
        if not self.load_data():
            print("Analysis halted due to data loading issues")
            return
            
        if not self.preprocess_data():
            print("Analysis halted due to preprocessing issues")
            return
        
        # Step 2: Basic statistics
        self.generate_basic_stats()
        
        # Step 3: Time series analysis
        self.run_time_series_analysis()
        
        # Step 4: Content performance analysis
        self.analyze_content_performance()
        
        # Step 5: Cluster analysis
        self.perform_cluster_analysis()
        
        # Step 6: NLP title analysis
        self.analyze_titles_with_nlp()
        
        # Step 7: Geographical analysis
        self.analyze_geographical_performance()
        
        # Step 8: Generate business recommendations
        self.generate_business_recommendations()
        
        print("\n===== NETFLIX CONTENT ANALYSIS COMPLETE =====")
        print(f"All results saved to the 'netflix_analysis_output' directory")
        
        return self.results
    
# Main entry point
if __name__ == "__main__":
    # Set input file path
    input_file = "netflix_content_2023.csv"
    
    # Create analyzer and run full pipeline
    analyzer = NetflixAnalyzer(input_file)
    results = analyzer.run_full_analysis()
    # Print final summary
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE - FINAL SUMMARY")
    print("="*60)
    
    if results:
        # Print key findings
        if 'basic_stats' in results:
            stats = results['basic_stats']
            print(f"✓ Total Content Analyzed: {stats.get('total_content', 'N/A')}")
            print(f"✓ Total Hours Viewed: {stats.get('total_hours_viewed', 'N/A')}M")
            print(f"✓ Average Performance: {stats.get('avg_hours_per_content', 'N/A')}M hours per title")
        
        # Print top recommendations
        if 'recommendations' in results:
            recs = results['recommendations']
            print(f"\n✓ Generated {len(recs)} strategic recommendations")
            
            # Show high-impact recommendations
            high_impact_recs = [r for r in recs if r.get('expected_impact') == 'High']
            if high_impact_recs:
                print("\nHIGH-IMPACT RECOMMENDATIONS:")
                for i, rec in enumerate(high_impact_recs[:3], 1):
                    print(f"  {i}. {rec['title']}")
                    print(f"     {rec['description']}")
        
        # Print output location
        print(f"\n✓ All visualizations saved to: netflix_analysis_output/")
        print("✓ Executive summary saved to: netflix_analysis_output/executive_summary.json")
        
        # Generate final performance report
        total_files = len([f for f in os.listdir('netflix_analysis_output') if f.endswith('.png')])
        print(f"✓ Generated {total_files} visualization charts")
        
    else:
        print("❌ Analysis failed - no results generated")
        print("Please check the input file and try again")
    
    print("\n" + "="*60)
    print("Thank you for using Netflix Content Analyzer!")
    print("="*60)
                