import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from transformers import pipeline
import tensorflow as tf
from scipy.signal import savgol_filter
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle, Arc
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class DriverPerformanceAnalyzer:
    def __init__(self):
        self.corner_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.braking_optimizer = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.racing_line_analyzer = tf.keras.models.Sequential()
        try:
            self.sentiment_analyzer = pipeline("sentiment-analysis")
        except Exception as e:
            print(f"Warning: Sentiment analyzer not available: {str(e)}")
            self.sentiment_analyzer = None
            
        self.corner_scaler = StandardScaler()
        self.braking_scaler = StandardScaler()
        self.is_trained = False
        
    def analyze_driver_performance(self, session_telemetry, track_map, driver_feedback=None):
        """
        Comprehensive analysis of driver performance
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before performing analysis. Call train_models() first.")
            
        try:
            telemetry = self.prepare_telemetry_data(session_telemetry)
            
            # Segment telemetry into corners and straights
            segments = self._segment_track(telemetry, track_map)
            
            # If no corners or braking zones found, return limited analysis
            if not segments['corners'] and not segments['braking_zones']:
                return {
                    'warning': 'No corners or braking zones detected in telemetry',
                    'racing_line': self._analyze_racing_line(telemetry, track_map),
                    'energy_management': self._analyze_energy_usage(telemetry),
                    'tire_management': self._analyze_tire_usage(telemetry),
                    'consistency_metrics': self._calculate_consistency(telemetry)
                }
            
            performance_analysis = {
                'corner_analysis': self._analyze_corners(segments['corners']),
                'braking_analysis': self._analyze_braking(segments['braking_zones']),
                'racing_line': self._analyze_racing_line(telemetry, track_map),
                'energy_management': self._analyze_energy_usage(telemetry),
                'tire_management': self._analyze_tire_usage(telemetry),
                'consistency_metrics': self._calculate_consistency(telemetry)
            }
            
            if driver_feedback:
                performance_analysis['feedback_analysis'] = self._analyze_driver_feedback(driver_feedback)
                
            return performance_analysis
            
        except Exception as e:
            print(f"Error during analysis: {str(e)}")
            print("Telemetry data types:")
            print(telemetry.dtypes)
            raise e

    def prepare_telemetry_data(self, raw_telemetry):
        """
        Process raw telemetry data into analyzable format and ensure all required features exist
        """
        # Convert DataFrame and handle data types
        telemetry = pd.DataFrame(raw_telemetry)
        
        # Required features for analysis
        required_features = {
            'track_gradient': 0.0,
            'track_temp': 25.0,
            'track_grip': 1.0,
            'weather_condition': 0,
            'corner_type': 2,
            'brake_temp': 400.0,
            'fuel_load': 100.0
        }
        
        # Add any missing required features with default values
        for feature, default_value in required_features.items():
            if feature not in telemetry.columns:
                print(f"Warning: Adding missing feature '{feature}' with default value {default_value}")
                telemetry[feature] = default_value
        
        # Convert timestamp to datetime if it's not already
        if telemetry['timestamp'].dtype == 'object':
            telemetry['timestamp'] = pd.to_datetime(telemetry['timestamp'])
        
        # Convert numeric columns from strings if necessary
        numeric_columns = [
            'speed', 'throttle_position', 'brake_pressure', 'steering_angle',
            'gear', 'engine_rpm', 'tire_temp_fl', 'tire_temp_fr', 'tire_temp_rl',
            'tire_temp_rr', 'gps_latitude', 'gps_longitude', 'drs', 'ers_deployment',
            'fuel_flow', 'g_lat', 'g_lon', 'track_gradient', 'track_temp',
            'track_grip', 'brake_temp', 'fuel_load', 'weather_condition',
            'corner_type'
        ]
        
        for col in numeric_columns:
            if col in telemetry.columns and telemetry[col].dtype == 'object':
                telemetry[col] = pd.to_numeric(telemetry[col], errors='coerce')
        
        # Calculate time differences in seconds
        time_diff = telemetry['timestamp'].diff().dt.total_seconds()
        
        # Calculate derived metrics
        telemetry['acceleration'] = telemetry['speed'].diff() / time_diff
        telemetry['brake_intensity'] = savgol_filter(telemetry['brake_pressure'], 5, 2)
        telemetry['throttle_smoothness'] = self._calculate_input_smoothness(telemetry['throttle_position'])
        
        # Calculate g-forces
        telemetry['lateral_g'] = self._calculate_lateral_g(telemetry['speed'], telemetry['steering_angle'])
        telemetry['longitudinal_g'] = self._calculate_longitudinal_g(telemetry['acceleration'])
        
        # Fill NaN values that might have been created during calculations
        telemetry = telemetry.fillna(method='ffill').fillna(method='bfill')
        
        return telemetry

    def train_corner_analyzer(self, training_data, corner_labels):
        """
        Train model to classify corner execution quality
        """
        corner_features = [
            'entry_speed', 'min_speed', 'exit_speed',
            'brake_point_distance', 'brake_pressure_max',
            'throttle_application_point', 'steering_smoothness',
            'racing_line_deviation', 'entry_angle', 'exit_angle'
        ]
        
        X = self.corner_scaler.fit_transform(training_data[corner_features])
        self.corner_classifier.fit(X, corner_labels)
        
        # Calculate feature importance
        self.corner_feature_importance = pd.DataFrame({
            'feature': corner_features,
            'importance': self.corner_classifier.feature_importances_
        }).sort_values('importance', ascending=False)
    
    def train_braking_optimizer(self, training_data, optimal_brake_points):
        """
        Train model to optimize braking points
        """
        braking_features = [
            'approach_speed', 'track_gradient', 'corner_type',
            'tire_temp_fl', 'tire_temp_fr', 'track_temp',
            'fuel_load', 'brake_temp', 'weather_condition',
            'track_grip'
        ]
        
        X = self.braking_scaler.fit_transform(training_data[braking_features])
        self.braking_optimizer.fit(X, optimal_brake_points)
        
        # Calculate feature importance
        self.braking_feature_importance = pd.DataFrame({
            'feature': braking_features,
            'importance': self.braking_optimizer.feature_importances_
        }).sort_values('importance', ascending=False)
    
    def train_models(self, training_data, corner_labels, optimal_brake_points):
        """
        Train all models at once with provided training data
        """
        try:
            # Train corner analyzer
            self.train_corner_analyzer(training_data, corner_labels)
            print("Corner analyzer trained successfully")
            
            # Train braking optimizer
            self.train_braking_optimizer(training_data, optimal_brake_points)
            print("Braking optimizer trained successfully")
            
            self.is_trained = True
            print("All models successfully trained")
            
        except Exception as e:
            print(f"Error during model training: {str(e)}")
            self.is_trained = False
            raise e

    def _analyze_driver_feedback(self, feedback_text):
        """
        Analyze driver feedback using text analysis.
        If sentiment analyzer is not available, provides basic keyword analysis.
        
        Parameters:
        feedback_text (str): Driver's feedback text
        
        Returns:
        dict: Analysis results including key themes and sentiment
        """
        # Define key terms to look for in feedback
        handling_terms = ['understeer', 'oversteer', 'balance', 'stable', 'unstable']
        power_terms = ['power', 'straight', 'acceleration', 'speed', 'fast']
        brake_terms = ['brake', 'stopping', 'locked', 'braking']
        tire_terms = ['grip', 'tires', 'tyres', 'degradation', 'wear']
        
        feedback_lower = feedback_text.lower()
        
        # Basic keyword analysis
        analysis = {
            'handling_issues': [term for term in handling_terms if term in feedback_lower],
            'power_issues': [term for term in power_terms if term in feedback_lower],
            'brake_issues': [term for term in brake_terms if term in feedback_lower],
            'tire_issues': [term for term in tire_terms if term in feedback_lower],
        }
        
        # Add sentiment analysis if available
        if self.sentiment_analyzer is not None:
            try:
                sentiment = self.sentiment_analyzer(feedback_text)
                analysis['sentiment'] = sentiment
            except Exception as e:
                analysis['sentiment'] = {'error': str(e)}
        else:
            # Basic sentiment through keyword matching
            positive_terms = ['good', 'great', 'better', 'improved', 'confident']
            negative_terms = ['bad', 'worse', 'difficult', 'struggling', 'problem']
            
            positive_count = sum(term in feedback_lower for term in positive_terms)
            negative_count = sum(term in feedback_lower for term in negative_terms)
            
            if positive_count > negative_count:
                analysis['basic_sentiment'] = 'positive'
            elif negative_count > positive_count:
                analysis['basic_sentiment'] = 'negative'
            else:
                analysis['basic_sentiment'] = 'neutral'
        
        # Generate action items based on feedback
        action_items = []
        
        if analysis['handling_issues']:
            action_items.append("Review suspension settings and aero balance")
        if analysis['power_issues']:
            action_items.append("Check engine mapping and power delivery")
        if analysis['brake_issues']:
            action_items.append("Analyze brake temperature and pressure distribution")
        if analysis['tire_issues']:
            action_items.append("Review tire pressure and temperature management")
            
        analysis['suggested_actions'] = action_items
        
        return analysis

    # [Previous methods remain the same...]
    def _calculate_input_smoothness(self, input_series):
        return savgol_filter(np.abs(input_series.diff().fillna(0)), 7, 3)
    
    def _calculate_lateral_g(self, speed, steering_angle):
        return (speed ** 2 * np.sin(np.radians(steering_angle))) / 9.81
    
    def _calculate_longitudinal_g(self, acceleration):
        return acceleration / 9.81
    
    def _segment_track(self, telemetry, track_map):
        segments = {
            'corners': [],
            'straights': [],
            'braking_zones': []
        }
        
        corner_mask = abs(telemetry['steering_angle']) > 10
        brake_mask = telemetry['brake_pressure'] > 20
        
        corner_segments = self._create_segments(telemetry[corner_mask])
        brake_segments = self._create_segments(telemetry[brake_mask])
        
        segments['corners'] = corner_segments
        segments['braking_zones'] = brake_segments
        
        return segments
    
    def _create_segments(self, data):
        if len(data) == 0:
            return []
            
        segments = []
        segment = {'data': data.iloc[0:1]}
        
        for _, row in data.iloc[1:].iterrows():
            segment['data'] = pd.concat([segment['data'], pd.DataFrame([row])])
            segments.append(segment)
            segment = {'data': pd.DataFrame([row])}
            
        return segments
    
    def _analyze_corners(self, corner_segments):
        """Detailed analysis of corner performance using RandomForest"""
        if not corner_segments:
            return {'warning': 'No corner segments found'}
            
        corner_analysis = {}
        for i, segment in enumerate(corner_segments):
            # Extract corner features
            corner_features = self._extract_corner_features(segment['data'])
            
            # Scale features
            scaled_features = self.corner_scaler.transform(corner_features.reshape(1, -1))
            
            # Predict corner execution quality
            quality_pred = self.corner_classifier.predict(scaled_features)[0]
            quality_proba = self.corner_classifier.predict_proba(scaled_features)[0]
            
            # Calculate potential gains
            potential_gains = self._calculate_corner_potential(
                segment['data'],
                corner_features,
                quality_pred
            )
            
            analysis = {
                'corner_number': i + 1,
                'execution_quality': quality_pred,
                'quality_confidence': float(max(quality_proba)),
                'minimum_speed': float(segment['data']['speed'].min()),
                'entry_speed': float(segment['data']['speed'].iloc[0]),
                'exit_speed': float(segment['data']['speed'].iloc[-1]),
                'potential_gains': potential_gains,
                'recommendations': self._get_corner_recommendations(
                    quality_pred,
                    potential_gains
                )
            }
            
            corner_analysis[f'corner_{i+1}'] = analysis
            
        return corner_analysis
    
    def _analyze_braking(self, braking_segments):
        """Analyze braking performance using RandomForest"""
        if not braking_segments:
            return {'warning': 'No braking segments found'}
            
        braking_analysis = {}
        for i, segment in enumerate(braking_segments):
            # Extract braking features
            braking_features = self._extract_braking_features(segment['data'])
            
            # Scale features
            scaled_features = self.braking_scaler.transform(braking_features.reshape(1, -1))
            
            # Predict optimal brake point
            optimal_brake_point = self.braking_optimizer.predict(scaled_features)[0]
            
            # Calculate braking efficiency
            efficiency_metrics = self._calculate_braking_efficiency(
                segment['data'],
                optimal_brake_point
            )
            
            analysis = {
                'zone_number': i + 1,
                'current_brake_point': float(segment['data']['distance'].iloc[0]),
                'optimal_brake_point': float(optimal_brake_point),
                'brake_point_delta': float(optimal_brake_point - segment['data']['distance'].iloc[0]),
                'efficiency_metrics': efficiency_metrics,
                'recommendations': self._get_braking_recommendations(efficiency_metrics)
            }
            
            braking_analysis[f'braking_zone_{i+1}'] = analysis
            
        return braking_analysis
    
    def _extract_corner_features(self, corner_data):
        """Extract features for corner analysis"""
        features = np.zeros(10)  # 10 features as defined in train_corner_analyzer
        
        features[0] = corner_data['speed'].iloc[0]  # entry_speed
        features[1] = corner_data['speed'].min()    # min_speed
        features[2] = corner_data['speed'].iloc[-1] # exit_speed
        features[3] = self._calculate_brake_point_distance(corner_data)
        features[4] = corner_data['brake_pressure'].max()
        features[5] = self._find_throttle_application_point(corner_data)
        features[6] = self._calculate_steering_smoothness(corner_data)
        features[7] = self._calculate_racing_line_deviation(corner_data)
        features[8] = self._calculate_entry_angle(corner_data)
        features[9] = self._calculate_exit_angle(corner_data)
        
        return features
    
    def _extract_braking_features(self, braking_data):
        """Extract features for braking analysis with safety checks"""
        features = np.zeros(10)  # 10 features as defined in train_braking_optimizer
        
        # Define default values for missing features
        default_values = {
            'approach_speed': braking_data['speed'].iloc[0] if 'speed' in braking_data else 200.0,
            'track_gradient': braking_data['track_gradient'].mean() if 'track_gradient' in braking_data else 0.0,
            'corner_type': braking_data['corner_type'].iloc[0] if 'corner_type' in braking_data else 2,
            'tire_temp_fl': braking_data['tire_temp_fl'].mean() if 'tire_temp_fl' in braking_data else 80.0,
            'tire_temp_fr': braking_data['tire_temp_fr'].mean() if 'tire_temp_fr' in braking_data else 80.0,
            'track_temp': braking_data['track_temp'].mean() if 'track_temp' in braking_data else 25.0,
            'fuel_load': braking_data['fuel_load'].iloc[0] if 'fuel_load' in braking_data else 100.0,
            'brake_temp': braking_data['brake_temp'].mean() if 'brake_temp' in braking_data else 400.0,
            'weather_condition': braking_data['weather_condition'].iloc[0] if 'weather_condition' in braking_data else 0,
            'track_grip': braking_data['track_grip'].mean() if 'track_grip' in braking_data else 1.0
        }
        
        # Fill features array with actual or default values
        features[0] = default_values['approach_speed']
        features[1] = default_values['track_gradient']
        features[2] = default_values['corner_type']
        features[3] = default_values['tire_temp_fl']
        features[4] = default_values['tire_temp_fr']
        features[5] = default_values['track_temp']
        features[6] = default_values['fuel_load']
        features[7] = default_values['brake_temp']
        features[8] = default_values['weather_condition']
        features[9] = default_values['track_grip']
        
        return features
    
    def _calculate_brake_point_distance(self, data):
        """Calculate distance from corner entry to brake point"""
        brake_idx = data['brake_pressure'].idxmax()
        return float(data['distance'].iloc[0] - data['distance'].loc[brake_idx])
    
    def _find_throttle_application_point(self, data):
        """Find point of throttle application after corner apex"""
        apex_idx = data['speed'].idxmin()
        throttle_data = data.loc[apex_idx:]['throttle_position']
        application_idx = throttle_data[throttle_data > 20].index[0]
        return float(data['distance'].loc[application_idx] - data['distance'].loc[apex_idx])
    
    def _calculate_steering_smoothness(self, data):
        """Calculate smoothness of steering inputs"""
        return float(np.std(np.diff(data['steering_angle'])))
    
    def _calculate_racing_line_deviation(self, data):
        """Calculate deviation from optimal racing line"""
        # Simplified calculation - would need actual racing line data
        return float(np.mean(np.abs(data['lateral_g'])))
    
    def _calculate_entry_angle(self, data):
        """Calculate corner entry angle"""
        return float(data['steering_angle'].iloc[0])
    
    def _calculate_exit_angle(self, data):
        """Calculate corner exit angle"""
        return float(data['steering_angle'].iloc[-1])
    
    def _calculate_corner_potential(self, data, features, quality):
        """Calculate potential time gains in corner execution"""
        if quality == 'optimal':
            return {'time_gain': 0.0, 'speed_gain': 0.0}
            
        # Calculate potential gains based on corner classification
        speed_potential = {
            'good': 0.5,      # 0.5 km/h potential gain
            'suboptimal': 1.0, # 1.0 km/h potential gain
            'poor': 2.0       # 2.0 km/h potential gain
        }
        
        speed_gain = speed_potential.get(quality, 0.0)
        time_gain = speed_gain / data['speed'].mean() * len(data) * 0.01  # Rough time gain estimation
        
        return {'time_gain': time_gain, 'speed_gain': speed_gain}
    
    def _calculate_braking_efficiency(self, data, optimal_point):
        """Calculate braking efficiency metrics"""
        current_point = data['distance'].iloc[0]
        
        return {
            'brake_point_difference': float(optimal_point - current_point),
            'brake_pressure_efficiency': float(np.mean(data['brake_pressure']) / np.max(data['brake_pressure'])),
            'deceleration_efficiency': float(np.min(np.diff(data['speed'])) / -9.81),
            'brake_stability': float(1.0 - np.std(data['lateral_g']) / np.mean(np.abs(data['lateral_g'])))
        }
    
    def _get_corner_recommendations(self, quality, potential_gains):
        """Generate recommendations for corner improvement"""
        if quality == 'optimal':
            return ['Maintain current technique']
            
        recommendations = []
        if quality == 'poor':
            recommendations.extend([
                'Review corner entry speed and braking point',
                'Focus on smooth steering inputs',
                'Optimize racing line'
            ])
        elif quality == 'suboptimal':
            recommendations.extend([
                'Fine-tune braking point',
                'Optimize throttle application'
            ])
        elif quality == 'good':
            recommendations.append('Minor adjustments to racing line')
            
        return recommendations
    
    def _get_braking_recommendations(self, efficiency_metrics):
        """Generate recommendations for braking improvement"""
        recommendations = []
        
        if abs(efficiency_metrics['brake_point_difference']) > 10:
            recommendations.append(
                f"Adjust brake point by {efficiency_metrics['brake_point_difference']:.1f} meters"
            )
            
        if efficiency_metrics['brake_pressure_efficiency'] < 0.85:
            recommendations.append("Increase initial brake pressure")
            
        if efficiency_metrics['deceleration_efficiency'] < 0.95:
            recommendations.append("Optimize brake modulation")
            
        if efficiency_metrics['brake_stability'] < 0.9:
            recommendations.append("Improve braking stability")
            
        return recommendations if recommendations else ['Current braking technique is optimal']
    
    def _analyze_racing_line(self, telemetry, track_map):
        return {
            'average_speed': telemetry['speed'].mean(),
            'top_speed': telemetry['speed'].max(),
            'speed_consistency': telemetry['speed'].std()
        }
    
    def _analyze_energy_usage(self, telemetry):
        return {
            'average_ers_deployment': telemetry['ers_deployment'].mean(),
            'drs_usage': (telemetry['drs'] > 0).mean() * 100
        }
    
    def _analyze_tire_usage(self, telemetry):
        return {
            'average_temp_fl': telemetry['tire_temp_fl'].mean(),
            'average_temp_fr': telemetry['tire_temp_fr'].mean(),
            'average_temp_rl': telemetry['tire_temp_rl'].mean(),
            'average_temp_rr': telemetry['tire_temp_rr'].mean(),
            'temp_variation': telemetry[['tire_temp_fl', 'tire_temp_fr', 'tire_temp_rl', 'tire_temp_rr']].std().mean()
        }
    
    def _calculate_consistency(self, telemetry):
        return {
            'speed_consistency': telemetry['speed'].std(),
            'throttle_consistency': telemetry['throttle_position'].std(),
            'brake_consistency': telemetry['brake_pressure'].std()
        }

class DriverPerformanceVisualizer:
    def __init__(self):
        self.track_map = None
        self.telemetry_plots = None
        plt.style.use('seaborn-v0_8')
        
    def plot_racing_line(self, telemetry, track_map, analysis):
        """Generate racing line visualization with analysis overlay"""
        fig = plt.figure(figsize=(15, 10))
        
        # Plot track outline
        plt.plot(track_map['centerline_lon'], track_map['centerline_lat'], 
                'k--', alpha=0.5, label='Track Centerline')
        
        # Plot actual racing line colored by speed
        scatter = plt.scatter(telemetry['gps_longitude'], telemetry['gps_latitude'],
                            c=telemetry['speed'], cmap='viridis',
                            s=10, alpha=0.6)
        plt.colorbar(scatter, label='Speed (km/h)')
        
        # Add corner numbers
        for corner in track_map[track_map['corner_number'] > 0].iterrows():
            plt.annotate(f"T{int(corner[1]['corner_number'])}",
                        (corner[1]['centerline_lon'], corner[1]['centerline_lat']))
        
        plt.title('Racing Line Analysis')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.legend()
        
        return fig
        
    def plot_corner_analysis(self, corner_data, analysis):
        """Generate detailed corner analysis visualization"""
        fig = make_subplots(rows=2, cols=2,
                           subplot_titles=('Speed Profile', 'Racing Line',
                                         'Throttle/Brake', 'Steering Angle'))
        
        # Speed profile
        fig.add_trace(
            go.Scatter(y=corner_data['speed'], name='Speed'),
            row=1, col=1
        )
        
        # Racing line (position)
        fig.add_trace(
            go.Scatter(x=corner_data['gps_longitude'],
                      y=corner_data['gps_latitude'],
                      mode='lines+markers',
                      name='Racing Line'),
            row=1, col=2
        )
        
        # Throttle and brake
        fig.add_trace(
            go.Scatter(y=corner_data['throttle_position'],
                      name='Throttle'),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(y=corner_data['brake_pressure'],
                      name='Brake'),
            row=2, col=1
        )
        
        # Steering angle
        fig.add_trace(
            go.Scatter(y=corner_data['steering_angle'],
                      name='Steering'),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text="Corner Analysis")
        return fig
        
    def plot_tire_usage(self, tire_data, analysis):
        """Generate tire usage and degradation visualization"""
        fig = plt.figure(figsize=(15, 10))
        
        # Create subplots for different tire metrics
        gs = fig.add_gridspec(2, 2)
        
        # Temperature plot
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(tire_data['tire_temp_fl'], label='Front Left')
        ax1.plot(tire_data['tire_temp_fr'], label='Front Right')
        ax1.plot(tire_data['tire_temp_rl'], label='Rear Left')
        ax1.plot(tire_data['tire_temp_rr'], label='Rear Right')
        ax1.set_title('Tire Temperatures')
        ax1.set_ylabel('Temperature (°C)')
        ax1.legend()
        
        # Tire temperature distribution
        ax2 = fig.add_subplot(gs[1, 0])
        tire_temps = [
            tire_data['tire_temp_fl'].mean(),
            tire_data['tire_temp_fr'].mean(),
            tire_data['tire_temp_rl'].mean(),
            tire_data['tire_temp_rr'].mean()
        ]
        ax2.bar(['FL', 'FR', 'RL', 'RR'], tire_temps)
        ax2.set_title('Average Tire Temperatures')
        
        # Temperature variation
        ax3 = fig.add_subplot(gs[1, 1])
        temp_vars = [
            tire_data['tire_temp_fl'].std(),
            tire_data['tire_temp_fr'].std(),
            tire_data['tire_temp_rl'].std(),
            tire_data['tire_temp_rr'].std()
        ]
        ax3.bar(['FL', 'FR', 'RL', 'RR'], temp_vars)
        ax3.set_title('Temperature Variation')
        
        plt.tight_layout()
        return fig
        
    def plot_energy_management(self, telemetry):
        """Generate energy management visualization"""
        fig = make_subplots(rows=2, cols=1,
                           subplot_titles=('ERS Deployment', 'DRS Usage'))
        
        # ERS deployment
        fig.add_trace(
            go.Scatter(y=telemetry['ers_deployment'],
                      name='ERS Deployment'),
            row=1, col=1
        )
        
        # DRS usage
        fig.add_trace(
            go.Scatter(y=telemetry['drs'],
                      name='DRS Active'),
            row=2, col=1
        )
        
        fig.update_layout(height=600, title_text="Energy Management Analysis")
        return fig
        
    def plot_consistency_analysis(self, telemetry):
        """Generate consistency analysis visualization"""
        fig = plt.figure(figsize=(15, 10))
        
        # Create violin plots for key metrics
        metrics = {
            'Speed Consistency': telemetry['speed'],
            'Throttle Consistency': telemetry['throttle_position'],
            'Brake Consistency': telemetry['brake_pressure'],
            'Steering Consistency': telemetry['steering_angle']
        }
        
        plt.violinplot([metrics[key] for key in metrics.keys()])
        plt.xticks(range(1, len(metrics) + 1), metrics.keys(), rotation=45)
        plt.title('Driver Consistency Analysis')
        
        return fig
        
    def generate_performance_report(self, analysis_results):
        """Generate comprehensive performance report with visualizations"""
        report = {
            'racing_line': self.plot_racing_line(
                analysis_results['telemetry'],
                analysis_results['track_map'],
                analysis_results['racing_line']
            ),
            'corner_analysis': self.plot_corner_analysis(
                analysis_results['telemetry'],
                analysis_results['corner_analysis']
            ),
            'tire_usage': self.plot_tire_usage(
                analysis_results['telemetry'],
                analysis_results['tire_management']
            ),
            'energy_management': self.plot_energy_management(
                analysis_results['telemetry']
            ),
            'consistency': self.plot_consistency_analysis(
                analysis_results['telemetry']
            )
        }
        
        return report
        
    def save_report(self, report, output_dir='performance_report'):
        """Save all visualizations to files"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for name, fig in report.items():
            if isinstance(fig, go.Figure):
                fig.write_html(f"{output_dir}/{name}.html")
            else:
                fig.savefig(f"{output_dir}/{name}.png")
                plt.close(fig)


def generate_sample_training_data():
    """Generate sample training data for model training"""
    # Generate sample data for both corner analysis and braking optimization
    n_samples = 100
    
    training_data = pd.DataFrame({
        # Corner analysis features
        'entry_speed': np.random.uniform(150, 250, n_samples),
        'min_speed': np.random.uniform(80, 150, n_samples),
        'exit_speed': np.random.uniform(150, 250, n_samples),
        'brake_point_distance': np.random.uniform(50, 150, n_samples),
        'brake_pressure_max': np.random.uniform(0.6, 1.0, n_samples),
        'throttle_application_point': np.random.uniform(0.2, 0.8, n_samples),
        'steering_smoothness': np.random.uniform(0.5, 1.0, n_samples),
        'racing_line_deviation': np.random.uniform(0, 0.5, n_samples),
        'entry_angle': np.random.uniform(10, 30, n_samples),
        'exit_angle': np.random.uniform(5, 25, n_samples),
        
        # Braking optimization features
        'approach_speed': np.random.uniform(200, 300, n_samples),
        'track_gradient': np.random.uniform(-10, 10, n_samples),
        'corner_type': np.random.randint(1, 4, n_samples),  # 1=slow, 2=medium, 3=fast
        'tire_temp_fl': np.random.uniform(60, 100, n_samples),
        'tire_temp_fr': np.random.uniform(60, 100, n_samples),
        'track_temp': np.random.uniform(20, 40, n_samples),
        'fuel_load': np.random.uniform(50, 105, n_samples),
        'brake_temp': np.random.uniform(200, 800, n_samples),
        'weather_condition': np.random.randint(0, 3, n_samples),  # 0=dry, 1=damp, 2=wet
        'track_grip': np.random.uniform(0.7, 1.0, n_samples)
    })
    
    # Generate sample corner quality labels
    qualities = ['optimal', 'good', 'suboptimal', 'poor']
    corner_labels = np.random.choice(qualities, size=n_samples)
    
    # Generate sample optimal brake points
    optimal_brake_points = np.random.uniform(80, 120, n_samples)
    
    return training_data, corner_labels, optimal_brake_points

def generate_sample_telemetry(n_samples=1000):
    """Generate sample telemetry data with all required features"""
    return pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=n_samples, freq='S'),
        'distance': np.cumsum(np.random.uniform(0, 5, n_samples)),  # Cumulative distance
        'speed': np.random.uniform(0, 300, n_samples),
        'throttle_position': np.random.uniform(0, 100, n_samples),
        'brake_pressure': np.random.uniform(0, 100, n_samples),
        'steering_angle': np.random.uniform(-45, 45, n_samples),
        'gear': np.random.randint(1, 8, n_samples),
        'engine_rpm': np.random.uniform(5000, 15000, n_samples),
        'tire_temp_fl': np.random.uniform(60, 100, n_samples),
        'tire_temp_fr': np.random.uniform(60, 100, n_samples),
        'tire_temp_rl': np.random.uniform(60, 100, n_samples),
        'tire_temp_rr': np.random.uniform(60, 100, n_samples),
        'gps_latitude': np.random.uniform(-0.1, 0.1, n_samples),
        'gps_longitude': np.random.uniform(-0.1, 0.1, n_samples),
        'drs': np.random.choice([0, 1], n_samples),
        'ers_deployment': np.random.uniform(0, 100, n_samples),
        'fuel_flow': np.random.uniform(0, 100, n_samples),
        'g_lat': np.random.uniform(-4, 4, n_samples),
        'g_lon': np.random.uniform(-4, 4, n_samples),
        'track_gradient': np.random.uniform(-10, 10, n_samples),  # Added
        'track_temp': np.random.uniform(20, 40, n_samples),
        'track_grip': np.random.uniform(0.8, 1.0, n_samples),
        'fuel_load': np.random.uniform(50, 100, n_samples),
        'brake_temp': np.random.uniform(200, 800, n_samples),
        'weather_condition': np.random.randint(0, 3, n_samples),  # Added
        'corner_type': np.random.randint(1, 4, n_samples),  # Added
    })

def main():
    # Initialize analyzers
    analyzer = DriverPerformanceAnalyzer()
    visualizer = DriverPerformanceVisualizer()
    
    # Load telemetry and track data
    try:
        telemetry_data = pd.read_csv('data/telemetry.csv')
        track_map_data = pd.read_csv('data/track_map.csv')
        print("Loaded real telemetry and track data")
    except FileNotFoundError:
        print("Generating sample data for demonstration...")
        telemetry_data = generate_sample_telemetry()
        track_map_data = pd.DataFrame({
            'centerline_lat': np.random.uniform(-0.1, 0.1, 100),
            'centerline_lon': np.random.uniform(-0.1, 0.1, 100),
            'corner_number': np.concatenate([np.zeros(80), np.arange(1, 21)])
        })

    # Generate and load training data
    print("Generating training data...")
    training_data, corner_labels, optimal_brake_points = generate_sample_training_data()
    
    # Train the models
    print("Training models...")
    try:
        analyzer.train_models(training_data, corner_labels, optimal_brake_points)
    except Exception as e:
        print(f"Error during model training: {str(e)}")
        return

    # Analyze performance
    print("Analyzing driver performance...")
    try:
        analysis = analyzer.analyze_driver_performance(
            telemetry_data,
            track_map_data,
            driver_feedback="The car feels unstable in high-speed corners and I'm struggling with understeer in slow corners. Tire degradation seems high on the front left."
        )
    except Exception as e:
        print(f"Error during performance analysis: {str(e)}")
        return

    # Print analysis results
    print("\nAnalysis Results:")
    for key, value in analysis.items():
        print(f"\n{key.upper()}:")
        print(value)
    
    # Add telemetry and track data to analysis for visualization
    analysis['telemetry'] = telemetry_data
    analysis['track_map'] = track_map_data
    
    # Generate and save visualization report
    print("\nGenerating performance report...")
    try:
        report = visualizer.generate_performance_report(analysis)
        visualizer.save_report(report)
        print("Performance report saved successfully!")
    except Exception as e:
        print(f"Error generating performance report: {str(e)}")

if __name__ == "__main__":
    main()