from flask import Flask, render_template
import pandas as pd
import plotly.express as px
import datetime

app = Flask(__name__)

# Step 1: Define the New Data
global_population = 8000000000  # Example value, replace with real data
births_per_day = 360000
deaths_per_day = 150000
population_growth_per_day = births_per_day - deaths_per_day
deaths_per_year = deaths_per_day * 365
population_growth_per_year = population_growth_per_day * 365

# Calculate future population trends
def calculate_population_trend(current_population, growth_per_year, years):
    future_population = []
    for year in range(years):
        current_population += growth_per_year
        future_population.append(current_population)
    return future_population

# Projecting the population for the next 10 years
years = 10
future_population = calculate_population_trend(global_population, population_growth_per_year, years)
future_years = list(range(datetime.datetime.now().year, datetime.datetime.now().year + years))

# Step 2: Gather Mortality Data
# Assuming datasets are in CSV format and stored locally
regions = ['Europe', 'Asia', 'America', 'China', 'Japan', 'Russia']
mortality_data = {}
for region in regions:
    mortality_data[region] = pd.read_csv(f'{region}_mortality.csv')

# Step 3: Data Preprocessing
def preprocess_data(df):
    df.dropna(inplace=True)
    df['Year'] = df['Year'].astype(int)
    return df

for region in regions:
    mortality_data[region] = preprocess_data(mortality_data[region])

# Combine all data into a single DataFrame
combined_data = pd.concat(mortality_data.values())

# Step 4: Statistical Analysis
# Example: Linear Regression to predict future trends
from sklearn.linear_model import LinearRegression
import numpy as np

X = combined_data[['Year']]
y = combined_data['Mortality_Rate']
model = LinearRegression()
model.fit(X, y)
future_mortality_years = np.array([[year] for year in range(2025, 2035)])
predictions = model.predict(future_mortality_years)

@app.route('/')
def index():
    # Step 5: Visualization using Plotly
    fig = px.line(combined_data, x='Year', y='Mortality_Rate', color='Region', title='Mortality Rate Over Time')
    fig.add_scatter(x=future_mortality_years.flatten(), y=predictions, mode='lines', name='Predicted Mortality Rate')
    
    # Population Trend Over Time
    fig_population = px.line(x=future_years, y=future_population, title='Projected Population Over Time')
    
    graph_html = fig.to_html(full_html=False)
    graph_population_html = fig_population.to_html(full_html=False)
    
    return render_template('index.html', graph_html=graph_html, graph_population_html=graph_population_html)

if __name__ == "__main__":
    app.run(debug=True)