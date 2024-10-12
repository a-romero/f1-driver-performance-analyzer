# F1 Driver Performance Analyzer

A comprehensive system for analyzing Formula 1 driver performance using Machine Learning. 

This system processes telemetry data to provide insights into driving technique, optimize racing lines, and suggest performance improvements.

## Features

### 1. Telemetry Analysis
- Real-time processing of car sensor data
- Corner entry/exit analysis
- Braking point optimization
- Racing line evaluation
- Energy management assessment
- Tire usage analysis
- Consistency metrics

### 2. Machine Learning Models
- LSTM networks for sequential pattern analysis
- Random Forest for corner classification
- Random Forest for brake point optimization
- Driver feedback analysis using NLP
- Ensemble predictions for comprehensive insights

### 3. Performance Metrics
- Corner execution quality
- Braking efficiency
- Racing line deviation
- Tire management
- Energy deployment efficiency
- Overall consistency

### 4. Visualization Capabilities
- Racing line overlays
- Corner analysis plots
- Tire usage heat maps
- Energy management graphs
- Consistency analysis charts
- Performance trend visualization

## Installation

```bash
# Clone the repository
git clone https://github.com/a-romero/f1-driver-performance-analyzer.git
cd f1-driver-performance-analyzer

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

## Required Dependencies

```python
# requirements.txt
pandas==2.0.0
numpy==1.24.0
scikit-learn==1.2.0
tensorflow==2.13.0
plotly==5.13.0
seaborn==0.12.0
matplotlib==3.7.0
transformers==4.30.0
scipy==1.10.0
```

## Usage

### Basic Analysis

```python
from driver_performance_analyzer import DriverPerformanceAnalyzer
from driver_performance_visualizer import DriverPerformanceVisualizer

# Initialize analyzers
analyzer = DriverPerformanceAnalyzer()
visualizer = DriverPerformanceVisualizer()

# Load data
telemetry_data = pd.read_csv('telemetry.csv')
track_map_data = pd.read_csv('track_map.csv')

# Perform analysis
analysis = analyzer.analyze_driver_performance(
    telemetry_data,
    track_map_data,
    driver_feedback="The car feels unstable in high-speed corners"
)

# Generate visualizations
report = visualizer.generate_performance_report(analysis)
visualizer.save_report(report)
```

### Training Corner Analysis Model

```python
# Prepare training data
corner_data = pd.DataFrame({
    'entry_speed': [...],
    'min_speed': [...],
    'exit_speed': [...],
    'brake_point_distance': [...],
    'brake_pressure_max': [...],
    'throttle_application_point': [...],
    'steering_smoothness': [...],
    'racing_line_deviation': [...],
    'entry_angle': [...],
    'exit_angle': [...]
})

corner_labels = ['optimal', 'good', 'suboptimal', 'poor']

# Train the model
analyzer.train_corner_analyzer(corner_data, corner_labels)
```

### Training Braking Optimizer

```python
# Prepare training data
braking_data = pd.DataFrame({
    'approach_speed': [...],
    'track_gradient': [...],
    'corner_type': [...],
    'tire_temp_fl': [...],
    'tire_temp_fr': [...],
    'track_temp': [...],
    'fuel_load': [...],
    'brake_temp': [...],
    'weather_condition': [...],
    'track_grip': [...]
})

optimal_brake_points = [...]  # Optimal braking points in meters

# Train the model
analyzer.train_braking_optimizer(braking_data, optimal_brake_points)
```

## Data Format

### Telemetry Data Format (telemetry.csv)
```csv
timestamp,distance,speed,throttle_position,brake_pressure,steering_angle,gear,engine_rpm,tire_temp_fl,tire_temp_fr,tire_temp_rl,tire_temp_rr,gps_latitude,gps_longitude,drs,ers_deployment,fuel_flow,g_lat,g_lon
2024-03-15 14:30:00.000,0,280.5,100,0,0,8,12000,95.2,94.8,93.5,93.8,45.5,7.5,1,80,85.5,0.1,0.2
...
```

### Track Map Data Format (track_map.csv)
```csv
distance,centerline_lat,centerline_lon,racing_line_lat,racing_line_lon,track_width,corner_number,corner_type,drs_zone,slope,surface_grip
0,45.5,7.5,45.501,7.501,15,0,straight,1,0,0.95
...
```

## Analysis Output

The analysis provides a comprehensive dictionary containing:

```python
{
    'corner_analysis': {
        'corner_1': {
            'execution_quality': 'good',
            'quality_confidence': 0.85,
            'minimum_speed': 125.5,
            'entry_speed': 180.2,
            'exit_speed': 165.8,
            'potential_gains': {
                'time_gain': 0.05,
                'speed_gain': 0.8
            },
            'recommendations': [
                'Fine-tune braking point',
                'Optimize throttle application'
            ]
        },
        # ... more corners
    },
    'braking_analysis': {
        'braking_zone_1': {
            'current_brake_point': 75.5,
            'optimal_brake_point': 78.2,
            'brake_point_delta': 2.7,
            'efficiency_metrics': {
                'brake_pressure_efficiency': 0.92,
                'deceleration_efficiency': 0.95,
                'brake_stability': 0.88
            },
            'recommendations': [
                'Adjust brake point by 2.7 meters',
                'Improve braking stability'
            ]
        },
        # ... more braking zones
    },
    # ... other analysis components
}
```

## Visualization Examples

The visualizer generates various plots and charts:

1. Racing Line Analysis:
   - Track map with actual vs. optimal racing line
   - Color-coded speed overlay
   - Corner numbers and DRS zones

2. Corner Analysis:
   - Speed profile through corner
   - Throttle/brake application
   - Steering angle
   - Racing line deviation

3. Tire Usage:
   - Temperature distribution
   - Wear patterns
   - Temperature variation

4. Energy Management:
   - ERS deployment
   - DRS usage
   - Energy recovery efficiency

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Authors

* **Alberto Romero** - [@a-romero](https://github.com/a-romero)

## Acknowledgments

* Formula 1 telemetry data standards
* Machine learning best practices in motorsport
* Advanced driver analysis techniques