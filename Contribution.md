# Dynamic Parking Pricing System

## Project Overview  
This project introduces a smart pricing system for urban parking lots that uses real-time data and advanced pricing techniques. It aims to increase revenue, improve space usage, and offer clear insights to parking operators. The system includes interactive dashboards and can be run easily in Jupyter Notebooks or cloud environments.

## Key Features  
- Three Pricing Models:
  - Linear Model: Adjusts pricing based on how full the parking lot is.  
  - Demand-Based Model: Uses factors like occupancy, queue length, traffic, vehicle type, and special events.  
  - Competitive Model: Considers nearby competitors’ prices within a 2km range using location data.  
- Real-Time Data Processing: 
  - Supports both live-streaming and batch data from multiple parking locations.  
  - Updates pricing every 30 minutes based on current conditions.  
- Detailed Analytics:  
  - Tracks how pricing changes affect revenue.  
  - Provides hourly and occupancy-based insights.  
  - Separates results by vehicle type and traffic conditions.  
- Interactive Visualization: 
  - Built using Bokeh and Panel for easy real-time monitoring.  
  - Includes charts for hourly pricing, occupancy vs price, traffic impact, and model comparisons.  
- Production-Ready: 
  - Modular and clean Python code.  
  - Includes error handling and CSV export options.  
  - Easy to plug into existing data systems.

## How It Works  
1. Data Ingestion:Loads historical or real-time data (like occupancy, vehicle type, traffic, etc.).  
2. Pricing Engine:Applies a chosen pricing model to calculate optimal prices for each location and time.  
3. Analytics & Reports:Generates reports on revenue, usage, and pricing performance.  
4. Visualization:Displays interactive dashboards with clear business and operational insights.

## Results  
- Revenue Growth:15–30% increase compared to static pricing.  
- Peak Hour Management:Adjusts prices automatically during busy times.  
- Traffic & Vehicle Awareness:Smart pricing based on vehicle type and road conditions.

## Usage  
1. Place your `dataset.csv` file in the project folder.  
2. Run the notebook or script provided.  
3. Open your browser to explore the dashboards and analytics.

## Technologies Used  
- Python (pandas, numpy, math)  
- Bokeh and Panel for dashboard creation  
- Jupyter Notebook or Google Colab for running the project
