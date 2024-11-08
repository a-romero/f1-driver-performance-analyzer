import pandas as pd
import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from transformers import pipeline
import tensorflow as tf
from scipy.signal import savgol_filter
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from driver_perf_visualizer import DriverPerformanceVisualizer

class DriverPerformanceAnalyzer:
    def __init__(self):
        self.corner_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.braking_optimizer = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            objective='reg:squarederror',
            early_stopping_rounds=10,
            eval_metric=['rmse', 'mae'],
            booster='gbtree',
            tree_method='hist',  # For faster training
            subsample=0.8,
            colsample_bytree=0.8,
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
        Train XGBoost model to optimize braking points
        
        Parameters:
        training_data (pd.DataFrame): Historic braking telemetry data
        optimal_brake_points (array): Optimal braking distances
        """
        features = [
            'approach_speed', 'track_gradient',
            'corner_type', 'tire_temp_fl', 'tire_temp_fr',
            'track_temp', 'fuel_load', 'brake_temp',
            'weather_condition', 'track_grip',
            'brake_pressure_history', 'entry_speed_delta',
            'downforce_level'  # Additional features for XGBoost
        ]
        
        # Split data for training with evaluation set
        X_train, X_eval, y_train, y_eval = train_test_split(
            training_data[features],
            optimal_brake_points,
            test_size=0.2,
            random_state=42
        )
        
        # Scale features
        X_train_scaled = self.braking_scaler.fit_transform(X_train)
        X_eval_scaled = self.braking_scaler.transform(X_eval)
        
        # Create DMatrix for faster training
        dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
        deval = xgb.DMatrix(X_eval_scaled, label=y_eval)
        
        # Train model with evaluation set
        self.braking_optimizer.fit(
            X_train_scaled, y_train,
            eval_set=[(X_eval_scaled, y_eval)],
            verbose=False
        )
        
        # Store feature importance
        self.braking_feature_importance = pd.DataFrame({
            'feature': features,
            'importance': self.braking_optimizer.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Calculate and store SHAP values for explainability
        self.shap_values = shap.TreeExplainer(self.braking_optimizer).shap_values(X_train_scaled)
    
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
        """Analyze braking performance using XGBoost"""
        if not braking_segments:
            return {'warning': 'No braking segments found'}
            
        braking_analysis = {}
        for i, segment in enumerate(braking_segments):
            # Extract enhanced braking features
            braking_features = self._extract_enhanced_braking_features(segment['data'])
            
            # Scale features
            scaled_features = self.braking_scaler.transform(braking_features.reshape(1, -1))
            
            # Get XGBoost prediction and contribution scores
            optimal_brake_point = self.braking_optimizer.predict(scaled_features)[0]
            
            # Get SHAP values for this prediction
            shap_values = shap.TreeExplainer(self.braking_optimizer).shap_values(scaled_features)
            
            # Calculate braking efficiency with enhanced metrics
            efficiency_metrics = self._calculate_enhanced_braking_efficiency(
                segment['data'],
                optimal_brake_point,
                shap_values
            )
            
            analysis = {
                'zone_number': i + 1,
                'current_brake_point': float(segment['data']['distance'].iloc[0]),
                'optimal_brake_point': float(optimal_brake_point),
                'brake_point_delta': float(optimal_brake_point - segment['data']['distance'].iloc[0]),
                'efficiency_metrics': efficiency_metrics,
                'feature_contributions': self._get_feature_contributions(
                    braking_features, 
                    shap_values
                ),
                'confidence_score': self._calculate_prediction_confidence(
                    scaled_features,
                    self.braking_optimizer
                ),
                'recommendations': self._get_enhanced_braking_recommendations(
                    efficiency_metrics,
                    shap_values
                )
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
    
    def _extract_enhanced_braking_features(self, data):
        """Extract enhanced features for XGBoost braking analysis"""
        features = np.zeros(13)  # 14 features as defined in train_braking_optimizer
        
        # Basic features
        features[0] = data['speed'].iloc[0]  # approach_speed
        features[1] = data['track_gradient'].mean()
        features[2] = self._encode_corner_type(data)
        features[3] = data['tire_temp_fl'].mean()
        features[4] = data['tire_temp_fr'].mean()
        features[5] = data['track_temp'].mean()
        features[6] = data['fuel_load'].iloc[0]
        features[7] = data['brake_temp'].mean()
        features[8] = self._encode_weather_condition(data)
        features[9] = data['track_grip'].mean()
        
        # Enhanced features for XGBoost
        features[10] = self._calculate_brake_pressure_history(data)
        features[11] = self._calculate_entry_speed_delta(data)
        features[12] = self._calculate_downforce_level(data)
        
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
    
    def _calculate_enhanced_braking_efficiency(self, data, optimal_point, shap_values):
        """Calculate enhanced braking efficiency metrics"""
        current_point = data['distance'].iloc[0]
        
        return {
            'brake_point_difference': float(optimal_point - current_point),
            'brake_pressure_efficiency': float(np.mean(data['brake_pressure']) / np.max(data['brake_pressure'])),
            'deceleration_efficiency': float(np.min(np.diff(data['speed'])) / -9.81),
            'brake_stability': float(1.0 - np.std(data['lateral_g']) / np.mean(np.abs(data['lateral_g']))),
            'brake_temperature_optimal': self._is_brake_temp_optimal(data['brake_temp']),
            'tire_grip_utilization': self._calculate_tire_grip_usage(data),
            'feature_importance_scores': dict(zip(
                self.braking_feature_importance['feature'],
                np.abs(shap_values[0])
            ))
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
    
    def _get_feature_contributions(self, features, shap_values):
        """Get feature contributions to the prediction"""
        return dict(zip(
            self.braking_feature_importance['feature'],
            shap_values[0].tolist()
        ))
    
    def _calculate_prediction_confidence(self, features, model):
        """Calculate confidence score for the prediction"""
        # Get prediction variance using XGBoost's built-in uncertainty estimation
        prediction_var = model.predict(features).var()
        confidence = 1 / (1 + prediction_var)
        return float(confidence)
    
    def _get_enhanced_braking_recommendations(self, efficiency_metrics, shap_values):
        """Generate enhanced recommendations based on XGBoost analysis"""
        recommendations = []
        feature_impacts = dict(zip(
            self.braking_feature_importance['feature'],
            np.abs(shap_values[0])
        ))
        
        # Sort features by impact
        sorted_impacts = sorted(
            feature_impacts.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        # Generate recommendations based on top impacting features
        for feature, impact in sorted_impacts[:3]:
            if impact > 0.1:  # Significant impact threshold
                recommendation = self._generate_feature_recommendation(
                    feature,
                    impact,
                    efficiency_metrics
                )
                if recommendation:
                    recommendations.append(recommendation)
        
        # Add specific recommendations based on efficiency metrics
        if efficiency_metrics['brake_point_difference'] > 10:
            recommendations.append(
                f"Brake {abs(efficiency_metrics['brake_point_difference']):.1f}m later"
            )
        elif efficiency_metrics['brake_point_difference'] < -10:
            recommendations.append(
                f"Brake {abs(efficiency_metrics['brake_point_difference']):.1f}m earlier"
            )
            
        if efficiency_metrics['brake_stability'] < 0.9:
            recommendations.append("Improve brake modulation for better stability")
            
        if not efficiency_metrics['brake_temperature_optimal']:
            recommendations.append("Monitor brake temperatures - not in optimal range")
            
        return recommendations if recommendations else ['Current braking technique is optimal']
    
    def _generate_feature_recommendation(self, feature, impact, metrics):
        """Generate specific recommendation based on feature impact"""
        recommendation_map = {
            'approach_speed': "Adjust approach speed for optimal braking",
            'track_grip': "Account for track grip conditions",
            'brake_temp': "Manage brake temperature",
            'tire_temp_fl': "Monitor front-left tire temperature",
            'tire_temp_fr': "Monitor front-right tire temperature",
            'fuel_load': "Adjust braking point for fuel load",
            'downforce_level': "Optimize downforce settings"
        }
        
        return recommendation_map.get(feature)
    
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

    def _encode_corner_type(self, data):
        """
        Encode corner type for model input.
        Options include ordinal encoding, one-hot encoding, or characteristic-based encoding.
        
        Parameters:
        data : pd.DataFrame or int
            Either the full data segment or a single corner type value
            Corner types:
            1: Slow speed / hairpin (0-120 km/h)
            2: Medium speed (120-200 km/h)
            3: Fast speed (200+ km/h)
            4: Chicane
            5: Increasing radius
            6: Decreasing radius
        
        Returns:
        float or np.array: Encoded corner type value(s)
        """
        # If input is a DataFrame, extract corner_type column
        if isinstance(data, pd.DataFrame):
            if 'corner_type' in data.columns:
                corner_type = data['corner_type'].iloc[0]
            else:
                return 0  # Default value if no corner type found
        else:
            corner_type = data
            
        # Option 1: Simple ordinal encoding (default)
        if hasattr(self, 'encoding_type') and self.encoding_type == 'ordinal':
            return float(corner_type)
        
        # Option 2: One-hot encoding
        elif hasattr(self, 'encoding_type') and self.encoding_type == 'one-hot':
            encoding = np.zeros(6)  # For 6 corner types
            encoding[int(corner_type) - 1] = 1
            return encoding
        
        # Option 3: Characteristic-based encoding (default)
        else:
            # Define corner characteristics
            corner_characteristics = {
                1: {  # Slow speed / hairpin
                    'speed_factor': 0.2,
                    'brake_importance': 0.9,
                    'traction_importance': 0.8,
                    'downforce_factor': 0.3
                },
                2: {  # Medium speed
                    'speed_factor': 0.5,
                    'brake_importance': 0.6,
                    'traction_importance': 0.6,
                    'downforce_factor': 0.6
                },
                3: {  # Fast speed
                    'speed_factor': 0.9,
                    'brake_importance': 0.4,
                    'traction_importance': 0.4,
                    'downforce_factor': 0.9
                },
                4: {  # Chicane
                    'speed_factor': 0.4,
                    'brake_importance': 0.7,
                    'traction_importance': 0.7,
                    'downforce_factor': 0.5
                },
                5: {  # Increasing radius
                    'speed_factor': 0.6,
                    'brake_importance': 0.5,
                    'traction_importance': 0.7,
                    'downforce_factor': 0.6
                },
                6: {  # Decreasing radius
                    'speed_factor': 0.5,
                    'brake_importance': 0.8,
                    'traction_importance': 0.6,
                    'downforce_factor': 0.7
                }
            }
            
            try:
                characteristics = corner_characteristics[int(corner_type)]
                
                # For simple encoding, return a weighted average of characteristics
                return (characteristics['speed_factor'] * 0.3 +
                    characteristics['brake_importance'] * 0.3 +
                    characteristics['traction_importance'] * 0.2 +
                    characteristics['downforce_factor'] * 0.2)
                    
            except KeyError:
                # Return default value for unknown corner types
                return 0.5

    def _get_corner_characteristics(self, corner_type):
        """
        Get detailed characteristics for a specific corner type.
        Useful for advanced analysis and recommendations.
        
        Parameters:
        corner_type : int
            Corner type identifier
        
        Returns:
        dict: Corner characteristics
        """
        characteristics = {
            1: {  # Slow speed / hairpin
                'name': 'Slow speed/Hairpin',
                'speed_range': '0-120 km/h',
                'key_factors': ['Braking stability', 'Traction on exit'],
                'typical_brake_pressure': 0.9,
                'typical_brake_duration': 'Long',
                'downforce_sensitivity': 'Low',
                'tire_stress': {
                    'front': 'Very High',
                    'rear': 'High'
                }
            },
            2: {  # Medium speed
                'name': 'Medium speed',
                'speed_range': '120-200 km/h',
                'key_factors': ['Balance', 'Line precision'],
                'typical_brake_pressure': 0.7,
                'typical_brake_duration': 'Medium',
                'downforce_sensitivity': 'Medium',
                'tire_stress': {
                    'front': 'High',
                    'rear': 'Medium'
                }
            },
            3: {  # Fast speed
                'name': 'Fast speed',
                'speed_range': '200+ km/h',
                'key_factors': ['Aero balance', 'Commitment'],
                'typical_brake_pressure': 0.5,
                'typical_brake_duration': 'Short',
                'downforce_sensitivity': 'High',
                'tire_stress': {
                    'front': 'Medium',
                    'rear': 'High'
                }
            },
            4: {  # Chicane
                'name': 'Chicane',
                'speed_range': 'Variable',
                'key_factors': ['Direction change', 'Kerb usage'],
                'typical_brake_pressure': 0.8,
                'typical_brake_duration': 'Multiple short',
                'downforce_sensitivity': 'Medium',
                'tire_stress': {
                    'front': 'High',
                    'rear': 'High'
                }
            },
            5: {  # Increasing radius
                'name': 'Increasing radius',
                'speed_range': 'Variable',
                'key_factors': ['Early apex', 'Progressive throttle'],
                'typical_brake_pressure': 0.6,
                'typical_brake_duration': 'Medium-Short',
                'downforce_sensitivity': 'Medium',
                'tire_stress': {
                    'front': 'Medium',
                    'rear': 'High'
                }
            },
            6: {  # Decreasing radius
                'name': 'Decreasing radius',
                'speed_range': 'Variable',
                'key_factors': ['Late apex', 'Progressive braking'],
                'typical_brake_pressure': 0.75,
                'typical_brake_duration': 'Medium-Long',
                'downforce_sensitivity': 'Medium-High',
                'tire_stress': {
                    'front': 'High',
                    'rear': 'Medium'
                }
            }
        }
        
        return characteristics.get(corner_type, {
            'name': 'Unknown',
            'speed_range': 'Unknown',
            'key_factors': [],
            'typical_brake_pressure': 0.7,
            'typical_brake_duration': 'Medium',
            'downforce_sensitivity': 'Medium',
            'tire_stress': {
                'front': 'Medium',
                'rear': 'Medium'
            }
        })

    def _encode_weather_condition(self, data):
        """
        Encode weather condition for model input.
        
        Parameters:
        data : pd.DataFrame or int
            Either the full data segment or a single weather condition value
            Weather conditions:
            0: Dry
            1: Damp
            2: Wet
            3: Light Rain
            4: Heavy Rain
            5: Mixed (changing conditions)
        
        Returns:
        float or np.array: Encoded weather condition value(s)
        """
        # If input is a DataFrame, extract weather_condition column
        if isinstance(data, pd.DataFrame):
            if 'weather_condition' in data.columns:
                weather_condition = data['weather_condition'].iloc[0]
            else:
                return 0  # Default to dry if no weather condition found
        else:
            weather_condition = data
            
        # Option 1: Simple ordinal encoding
        if hasattr(self, 'weather_encoding_type') and self.weather_encoding_type == 'ordinal':
            return float(weather_condition)
        
        # Option 2: One-hot encoding
        elif hasattr(self, 'weather_encoding_type') and self.weather_encoding_type == 'one-hot':
            encoding = np.zeros(6)  # For 6 weather conditions
            encoding[int(weather_condition)] = 1
            return encoding
        
        # Option 3: Characteristic-based encoding (default)
        else:
            # Define weather impact characteristics
            weather_characteristics = {
                0: {  # Dry
                    'grip_factor': 1.0,
                    'braking_efficiency': 1.0,
                    'visibility': 1.0,
                    'tire_temp_factor': 1.0,
                    'track_evolution': 1.0
                },
                1: {  # Damp
                    'grip_factor': 0.8,
                    'braking_efficiency': 0.85,
                    'visibility': 0.9,
                    'tire_temp_factor': 0.9,
                    'track_evolution': 0.7
                },
                2: {  # Wet
                    'grip_factor': 0.6,
                    'braking_efficiency': 0.7,
                    'visibility': 0.7,
                    'tire_temp_factor': 0.7,
                    'track_evolution': 0.5
                },
                3: {  # Light Rain
                    'grip_factor': 0.7,
                    'braking_efficiency': 0.8,
                    'visibility': 0.8,
                    'tire_temp_factor': 0.8,
                    'track_evolution': 0.6
                },
                4: {  # Heavy Rain
                    'grip_factor': 0.4,
                    'braking_efficiency': 0.6,
                    'visibility': 0.5,
                    'tire_temp_factor': 0.6,
                    'track_evolution': 0.3
                },
                5: {  # Mixed
                    'grip_factor': 0.65,
                    'braking_efficiency': 0.75,
                    'visibility': 0.75,
                    'tire_temp_factor': 0.75,
                    'track_evolution': 0.4
                }
            }
            
            try:
                characteristics = weather_characteristics[int(weather_condition)]
                
                # Return weighted average of characteristics
                return (characteristics['grip_factor'] * 0.35 +
                    characteristics['braking_efficiency'] * 0.25 +
                    characteristics['visibility'] * 0.15 +
                    characteristics['tire_temp_factor'] * 0.15 +
                    characteristics['track_evolution'] * 0.10)
                    
            except KeyError:
                # Return default value (dry conditions) for unknown weather
                return 1.0
                
    def _get_weather_characteristics(self, weather_condition):
        """
        Get detailed characteristics for a specific weather condition.
        Useful for advanced analysis and recommendations.
        
        Parameters:
        weather_condition : int
            Weather condition identifier
        
        Returns:
        dict: Weather characteristics and recommended adjustments
        """
        characteristics = {
            0: {  # Dry
                'name': 'Dry',
                'brake_bias_adjust': 0,
                'recommended_lines': ['Standard racing line'],
                'tire_pressure_adjust': 0,
                'brake_pressure_adjust': 0,
                'key_considerations': [
                    'Standard brake points',
                    'Maximum grip available',
                    'Normal tire temperature window'
                ]
            },
            1: {  # Damp
                'name': 'Damp',
                'brake_bias_adjust': -1,  # Move brake bias forward
                'recommended_lines': ['Slightly off racing line for more grip'],
                'tire_pressure_adjust': -1,
                'brake_pressure_adjust': -0.15,
                'key_considerations': [
                    'Earlier braking required',
                    'Reduced grip in normal racing line',
                    'Variable grip levels across track'
                ]
            },
            2: {  # Wet
                'name': 'Wet',
                'brake_bias_adjust': -2,
                'recommended_lines': ['Off racing line', 'Avoid standing water'],
                'tire_pressure_adjust': -2,
                'brake_pressure_adjust': -0.3,
                'key_considerations': [
                    'Significantly earlier braking',
                    'Avoid puddles and standing water',
                    'Gentle inputs required'
                ]
            },
            3: {  # Light Rain
                'name': 'Light Rain',
                'brake_bias_adjust': -1.5,
                'recommended_lines': ['Mixed line depending on grip'],
                'tire_pressure_adjust': -1.5,
                'brake_pressure_adjust': -0.2,
                'key_considerations': [
                    'Continuously changing grip levels',
                    'Monitor tire temperatures',
                    'Progressive brake application'
                ]
            },
            4: {  # Heavy Rain
                'name': 'Heavy Rain',
                'brake_bias_adjust': -3,
                'recommended_lines': ['Far off racing line', 'Higher ground'],
                'tire_pressure_adjust': -2.5,
                'brake_pressure_adjust': -0.4,
                'key_considerations': [
                    'Maximum caution required',
                    'Significant aquaplaning risk',
                    'Minimal brake pressure'
                ]
            },
            5: {  # Mixed
                'name': 'Mixed Conditions',
                'brake_bias_adjust': -1.5,
                'recommended_lines': ['Adaptive line choice'],
                'tire_pressure_adjust': -1.5,
                'brake_pressure_adjust': -0.25,
                'key_considerations': [
                    'Highly variable grip',
                    'Constant adaptation required',
                    'Conservative initial approach'
                ]
            }
        }
        
        return characteristics.get(weather_condition, {
            'name': 'Unknown',
            'brake_bias_adjust': 0,
            'recommended_lines': ['Standard'],
            'tire_pressure_adjust': 0,
            'brake_pressure_adjust': 0,
            'key_considerations': ['Standard approach']
        })
    
    def _calculate_entry_speed_delta(self, data):
        """
        Calculate the difference between actual and ideal corner entry speeds,
        taking into account various factors affecting optimal entry speed.
        
        Parameters:
        data : pd.DataFrame
            Telemetry data containing speed, corner, and condition information
            Expected columns: 'speed', 'corner_type', 'track_grip', 'fuel_load', etc.
            
        Returns:
        float: Entry speed delta in km/h (positive means faster than ideal, negative means slower)
        """
        try:
            # Get current entry speed (first speed value in the sequence)
            current_entry_speed = data['speed'].iloc[0]
            
            # Calculate ideal entry speed based on corner and conditions
            ideal_entry_speed = self._calculate_ideal_entry_speed(data)
            
            # Calculate raw delta
            speed_delta = current_entry_speed - ideal_entry_speed
            
            # Store detailed metrics for analysis if needed
            self.entry_speed_metrics = {
                'current_entry_speed': current_entry_speed,
                'ideal_entry_speed': ideal_entry_speed,
                'raw_delta': speed_delta,
                'normalized_delta': self._normalize_speed_delta(speed_delta),
                'conditions_factor': self._get_conditions_factor(data),
                'risk_factor': self._calculate_entry_risk_factor(data, speed_delta)
            }
            
            return speed_delta
            
        except Exception as e:
            print(f"Error calculating entry speed delta: {str(e)}")
            return 0.0
            
    def _calculate_ideal_entry_speed(self, data):
        """
        Calculate ideal corner entry speed based on corner type and conditions.
        """
        # Base entry speeds for different corner types (in km/h)
        base_entry_speeds = {
            1: 120,  # Slow corner
            2: 180,  # Medium corner
            3: 240,  # Fast corner
            4: 160,  # Chicane
            5: 200,  # Increasing radius
            6: 170   # Decreasing radius
        }
        
        corner_type = data['corner_type'].iloc[0]
        base_speed = base_entry_speeds.get(corner_type, 180)  # Default to medium corner speed
        
        # Adjust for conditions
        conditions_factor = self._get_conditions_factor(data)
        
        # Adjust for track specific factors
        track_factors = self._get_track_specific_factors(data)
        
        # Calculate adjusted ideal speed
        ideal_speed = base_speed * conditions_factor * track_factors
        
        return ideal_speed
        
    def _get_conditions_factor(self, data):
        """
        Calculate adjustment factor based on current conditions.
        """
        factor = 1.0
        
        # Grip adjustment
        if 'track_grip' in data.columns:
            grip = data['track_grip'].iloc[0]
            factor *= (0.7 + 0.3 * grip)  # Grip affects speed by up to 30%
            
        # Weather adjustment
        if 'weather_condition' in data.columns:
            weather = data['weather_condition'].iloc[0]
            weather_factors = {
                0: 1.0,    # Dry
                1: 0.9,    # Damp
                2: 0.8,    # Wet
                3: 0.85,   # Light Rain
                4: 0.7,    # Heavy Rain
                5: 0.8     # Mixed
            }
            factor *= weather_factors.get(weather, 1.0)
            
        # Fuel load adjustment
        if 'fuel_load' in data.columns:
            fuel = data['fuel_load'].iloc[0]
            fuel_factor = 1.0 - (fuel - 50) * 0.001  # 0.1% reduction per kg above 50kg
            factor *= fuel_factor
            
        # Tire condition adjustment
        if all(col in data.columns for col in ['tire_temp_fl', 'tire_temp_fr']):
            avg_temp = (data['tire_temp_fl'].iloc[0] + data['tire_temp_fr'].iloc[0]) / 2
            temp_factor = 1.0 - abs(avg_temp - 90) * 0.005  # Optimal temp around 90°C
            factor *= temp_factor
            
        # Track temperature adjustment
        if 'track_temp' in data.columns:
            track_temp = data['track_temp'].iloc[0]
            temp_factor = 1.0 - abs(track_temp - 30) * 0.003  # Optimal temp around 30°C
            factor *= temp_factor
            
        return factor
        
    def _get_track_specific_factors(self, data):
        """
        Calculate track-specific adjustment factors.
        """
        factor = 1.0
        
        # Gradient adjustment
        if 'track_gradient' in data.columns:
            gradient = data['track_gradient'].iloc[0]
            factor *= (1.0 - gradient * 0.01)  # 1% adjustment per degree of gradient
            
        # Banking adjustment
        if 'track_banking' in data.columns:
            banking = data['track_banking'].iloc[0]
            factor *= (1.0 + banking * 0.02)  # 2% adjustment per degree of banking
            
        # Wind effect
        if all(col in data.columns for col in ['wind_speed', 'wind_direction']):
            wind_effect = self._calculate_wind_effect(data)
            factor *= (1.0 + wind_effect)
            
        # Downforce level
        if 'downforce_level' in data.columns:
            downforce = data['downforce_level'].iloc[0]
            factor *= (0.8 + 0.4 * downforce)  # Downforce affects speed by up to 40%
            
        return factor
        
    def _calculate_wind_effect(self, data):
        """
        Calculate the effect of wind on entry speed.
        """
        wind_speed = data['wind_speed'].iloc[0]
        wind_direction = data['wind_direction'].iloc[0]
        
        # Assuming corner direction is available
        if 'corner_direction' in data.columns:
            corner_direction = data['corner_direction'].iloc[0]
            # Calculate relative wind angle
            relative_angle = abs(wind_direction - corner_direction)
            # Normalize to 0-180 degrees
            relative_angle = min(relative_angle, 360 - relative_angle)
            
            # Calculate wind effect (-0.1 to 0.1)
            wind_effect = wind_speed * np.cos(np.radians(relative_angle)) * 0.001
        else:
            wind_effect = 0.0
            
        return wind_effect
        
    def _normalize_speed_delta(self, delta):
        """
        Normalize speed delta to a -1 to 1 scale.
        """
        return np.clip(delta / 50.0, -1.0, 1.0)  # 50 km/h as maximum expected delta
        
    def _calculate_entry_risk_factor(self, data, speed_delta):
        """
        Calculate risk factor based on speed delta and conditions.
        """
        base_risk = abs(speed_delta) / 50.0  # Base risk from speed delta
        
        # Increase risk based on conditions
        conditions_risk = 0.0
        
        # Weather risk
        if 'weather_condition' in data.columns:
            weather = data['weather_condition'].iloc[0]
            weather_risk = {
                0: 0.0,    # Dry
                1: 0.2,    # Damp
                2: 0.4,    # Wet
                3: 0.3,    # Light Rain
                4: 0.6,    # Heavy Rain
                5: 0.5     # Mixed
            }
            conditions_risk += weather_risk.get(weather, 0.0)
            
        # Tire temperature risk
        if all(col in data.columns for col in ['tire_temp_fl', 'tire_temp_fr']):
            avg_temp = (data['tire_temp_fl'].iloc[0] + data['tire_temp_fr'].iloc[0]) / 2
            temp_risk = abs(avg_temp - 90) * 0.01  # Risk increases with deviation from optimal temp
            conditions_risk += temp_risk
            
        # Grip risk
        if 'track_grip' in data.columns:
            grip = data['track_grip'].iloc[0]
            grip_risk = (1 - grip) * 0.3
            conditions_risk += grip_risk
            
        # Combine risks
        total_risk = (base_risk * 0.7 + conditions_risk * 0.3)
        
        return np.clip(total_risk, 0.0, 1.0)
    
    def _calculate_downforce_level(self, data):
        """
        Calculate effective downforce level based on car setup and current conditions.
        Returns a normalized value between 0.0 (minimum downforce) and 1.0 (maximum downforce).
        
        Parameters:
        data : pd.DataFrame
            Telemetry data containing aero and condition information
            Expected columns: speed, wing_angles, ride_height, etc.
        
        Returns:
        float: Normalized downforce level (0.0 to 1.0)
        """
        try:
            # Initialize base downforce level
            base_downforce = 0.0
            factors_counted = 0
            
            # 1. Wing Configuration Analysis
            if all(col in data.columns for col in ['front_wing_angle', 'rear_wing_angle']):
                front_wing = data['front_wing_angle'].iloc[0]
                rear_wing = data['rear_wing_angle'].iloc[0]
                
                # Calculate wing contribution (typical F1 wing angles: 0-40 degrees)
                wing_downforce = (
                    0.4 * (front_wing / 40.0) +  # Front wing contribution
                    0.6 * (rear_wing / 40.0)     # Rear wing contribution
                )
                base_downforce += wing_downforce
                factors_counted += 1
                
            # 2. Ride Height Analysis
            if all(col in data.columns for col in ['front_ride_height', 'rear_ride_height']):
                front_height = data['front_ride_height'].iloc[0]
                rear_height = data['rear_ride_height'].iloc[0]
                
                # Calculate ride height contribution (optimal height around 30-50mm)
                height_efficiency = self._calculate_ride_height_efficiency(front_height, rear_height)
                base_downforce += height_efficiency
                factors_counted += 1
                
            # 3. Speed-Dependent Downforce
            if 'speed' in data.columns:
                speed = data['speed'].iloc[0]
                speed_factor = self._calculate_speed_dependent_downforce(speed)
                base_downforce += speed_factor
                factors_counted += 1
                
            # 4. Floor and Diffuser Effect
            if 'floor_damage' in data.columns:
                floor_condition = 1.0 - data['floor_damage'].iloc[0]  # 1.0 = perfect, 0.0 = damaged
                base_downforce += floor_condition
                factors_counted += 1
                
            # 5. Environmental Factors
            env_factor = self._calculate_environmental_downforce_factor(data)
            base_downforce += env_factor
            factors_counted += 1
            
            # Calculate average downforce level
            if factors_counted > 0:
                effective_downforce = base_downforce / factors_counted
            else:
                effective_downforce = 0.5  # Default if no factors available
                
            # Store detailed metrics
            self.downforce_metrics = {
                'effective_downforce': effective_downforce,
                'wing_configuration': wing_downforce if 'wing_downforce' in locals() else None,
                'ride_height_efficiency': height_efficiency if 'height_efficiency' in locals() else None,
                'speed_factor': speed_factor if 'speed_factor' in locals() else None,
                'floor_condition': floor_condition if 'floor_condition' in locals() else None,
                'environmental_factor': env_factor
            }
            
            return np.clip(effective_downforce, 0.0, 1.0)
        
        except Exception as e:
            print(f"Error calculating downforce level: {str(e)}")
            return 0.5
        
    def _calculate_ride_height_efficiency(self, front_height, rear_height):
        """
        Calculate efficiency of ride height for downforce generation.
        Optimal ride heights typically around 30-50mm for F1 cars.
        """
        # Define optimal ride height ranges
        optimal_front = 35
        optimal_rear = 45
        
        # Calculate deviations from optimal (normalized)
        front_deviation = abs(front_height - optimal_front) / optimal_front
        rear_deviation = abs(rear_height - optimal_rear) / optimal_rear
        
        # Calculate efficiency (1.0 = optimal, decreasing with deviation)
        front_efficiency = 1.0 / (1.0 + front_deviation)
        rear_efficiency = 1.0 / (1.0 + rear_deviation)
        
        # Combine with weight factors (rear typically more important)
        return 0.4 * front_efficiency + 0.6 * rear_efficiency
        
    def _calculate_speed_dependent_downforce(self, speed):
        """
        Calculate speed-dependent downforce factor.
        F1 cars generate maximum downforce at high speeds (>250 km/h).
        """
        # Normalize speed (assuming max meaningful speed for downforce is 350 km/h)
        norm_speed = min(speed / 350.0, 1.0)
        
        # Downforce increases with square of speed (simplified)
        return norm_speed ** 2
        
    def _calculate_environmental_downforce_factor(self, data):
        """
        Calculate environmental effects on downforce.
        """
        env_factor = 1.0
        factors_counted = 0
        
        # 1. Air Density Effect (based on temperature and altitude)
        if 'air_temperature' in data.columns:
            temp = data['air_temperature'].iloc[0]
            # Air density decreases with temperature
            density_factor = 1.0 - (temp - 20) * 0.002  # 0.2% reduction per degree above 20°C
            env_factor *= density_factor
            factors_counted += 1
            
        if 'altitude' in data.columns:
            altitude = data['altitude'].iloc[0]
            # Air density decreases with altitude
            altitude_factor = 1.0 - (altitude / 1000) * 0.1  # 10% reduction per 1000m
            env_factor *= altitude_factor
            factors_counted += 1
            
        # 2. Wind Effect
        if all(col in data.columns for col in ['wind_speed', 'wind_direction']):
            wind_factor = self._calculate_wind_effect_on_downforce(
                data['wind_speed'].iloc[0],
                data['wind_direction'].iloc[0]
            )
            env_factor *= wind_factor
            factors_counted += 1
            
        # 3. Track Condition Effect
        if 'track_temp' in data.columns:
            track_temp = data['track_temp'].iloc[0]
            # Hot track generally means less dense air
            track_factor = 1.0 - (track_temp - 30) * 0.001  # 0.1% reduction per degree above 30°C
            env_factor *= track_factor
            factors_counted += 1
            
        # Return average if factors counted, otherwise return neutral factor
        return env_factor if factors_counted > 0 else 1.0
        
    def _calculate_wind_effect_on_downforce(self, wind_speed, wind_direction):
        """
        Calculate wind effect on downforce.
        Headwind increases effective airspeed and thus downforce.
        Tailwind reduces effective airspeed.
        Crosswind can reduce overall downforce efficiency.
        """
        # Normalize wind speed (assuming max significant wind speed is 50 km/h)
        norm_wind_speed = min(wind_speed / 50.0, 1.0)
        
        # Calculate directional effect (assuming 0° is headwind)
        # Convert direction to radians
        direction_rad = np.radians(wind_direction)
        
        # Calculate components
        headwind_component = np.cos(direction_rad)  # 1 for headwind, -1 for tailwind
        crosswind_component = abs(np.sin(direction_rad))  # 0 for head/tail wind, 1 for pure crosswind
        
        # Calculate overall effect
        wind_effect = 1.0 + (
            0.1 * headwind_component * norm_wind_speed -  # Headwind boost
            0.2 * crosswind_component * norm_wind_speed   # Crosswind penalty
        )
        
        return np.clip(wind_effect, 0.7, 1.3)  # Limit wind effect to ±30%
    
    def _calculate_brake_pressure_history(self, data):
        """
        Calculate historical brake pressure characteristics and patterns.
        
        Parameters:
        data : pd.DataFrame
            Telemetry data containing brake pressure measurements
            Expected columns: 'brake_pressure', 'speed', 'timestamp'
        
        Returns:
        float: Normalized brake pressure history score (0.0 to 1.0)
        """
        if 'brake_pressure' not in data.columns:
            return 0.5  # Default value if no brake pressure data available
            
        try:
            # Extract brake pressure data
            brake_pressures = data['brake_pressure'].values
            
            # Calculate various brake pressure characteristics
            
            # 1. Initial brake application rate
            initial_rate = np.diff(brake_pressures[:10]).mean()
            
            # 2. Maximum brake pressure
            max_pressure = np.max(brake_pressures)
            
            # 3. Average brake pressure
            avg_pressure = np.mean(brake_pressures)
            
            # 4. Brake pressure stability (lower std = more stable)
            pressure_stability = 1 / (1 + np.std(brake_pressures))
            
            # 5. Brake release smoothness
            release_smoothness = self._calculate_brake_release_smoothness(brake_pressures)
            
            # 6. Brake modulation
            modulation_score = self._calculate_brake_modulation(brake_pressures)
            
            # 7. Calculate brake efficiency
            brake_efficiency = self._calculate_brake_efficiency(data)
            
            # Calculate pressure consistency across different speed ranges
            if 'speed' in data.columns:
                consistency_score = self._calculate_speed_based_consistency(
                    data['speed'].values,
                    brake_pressures
                )
            else:
                consistency_score = 0.5
            
            # Calculate temporal characteristics if timestamp available
            if 'timestamp' in data.columns:
                temporal_score = self._calculate_temporal_characteristics(
                    data['timestamp'].values,
                    brake_pressures
                )
            else:
                temporal_score = 0.5
                
            # Weighted combination of all factors
            history_score = (
                0.15 * self._normalize_rate(initial_rate) +
                0.15 * max_pressure +
                0.10 * (avg_pressure / max_pressure) +
                0.15 * pressure_stability +
                0.15 * release_smoothness +
                0.10 * modulation_score +
                0.10 * consistency_score +
                0.10 * temporal_score
            )
            
            # Store detailed metrics for analysis if needed
            self.brake_pressure_metrics = {
                'initial_rate': initial_rate,
                'max_pressure': max_pressure,
                'avg_pressure': avg_pressure,
                'pressure_stability': pressure_stability,
                'release_smoothness': release_smoothness,
                'modulation_score': modulation_score,
                'consistency_score': consistency_score,
                'temporal_score': temporal_score,
                'brake_efficiency': brake_efficiency
            }
            
            return np.clip(history_score, 0.0, 1.0)
            
        except Exception as e:
            print(f"Error calculating brake pressure history: {str(e)}")
            return 0.5
            
    def _normalize_rate(self, rate):
        """Normalize brake application rate to 0-1 range"""
        # Typical F1 brake application rates range from 0 to about 1.0
        return np.clip((rate + 1) / 2, 0, 1)
        
    def _calculate_brake_release_smoothness(self, brake_pressures):
        """Calculate smoothness of brake release"""
        # Find release phases (where pressure is decreasing)
        release_phases = np.where(np.diff(brake_pressures) < 0)[0]
        
        if len(release_phases) == 0:
            return 1.0
            
        # Calculate smoothness as inverse of variance in release rate
        release_rates = np.diff(brake_pressures[release_phases])
        smoothness = 1 / (1 + np.var(release_rates))
        
        return smoothness
        
    def _calculate_brake_modulation(self, brake_pressures):
        """Calculate brake modulation score"""
        # Find number of distinct pressure changes
        pressure_changes = np.diff(brake_pressures)
        modulations = np.where(np.abs(pressure_changes) > 0.05)[0]
        
        # Calculate modulation frequency
        modulation_freq = len(modulations) / len(brake_pressures)
        
        # Calculate modulation amplitude
        modulation_amp = np.std(pressure_changes)
        
        # Combine frequency and amplitude metrics
        modulation_score = (
            0.6 * (1 - modulation_freq) +  # Lower frequency is better
            0.4 * (1 / (1 + modulation_amp))  # Lower amplitude is better
        )
        
        return np.clip(modulation_score, 0.0, 1.0)
        
    def _calculate_brake_efficiency(self, data):
        """Calculate overall brake efficiency"""
        if 'speed' not in data.columns:
            return 0.5
            
        brake_pressures = data['brake_pressure'].values
        speeds = data['speed'].values
        
        # Calculate deceleration
        deceleration = -np.diff(speeds)
        
        # Calculate efficiency as deceleration per unit brake pressure
        brake_pressures_trimmed = brake_pressures[:-1]  # Match length with deceleration
        mask = brake_pressures_trimmed > 0.1  # Only consider significant brake applications
        
        if not any(mask):
            return 0.5
            
        efficiency = np.mean(deceleration[mask] / brake_pressures_trimmed[mask])
        
        # Normalize efficiency score
        return np.clip(efficiency / 10, 0.0, 1.0)  # Assuming max efficient deceleration of 10 m/s²
        
    def _calculate_speed_based_consistency(self, speeds, brake_pressures):
        """Calculate brake pressure consistency across speed ranges"""
        # Define speed ranges
        speed_ranges = [
            (0, 100),    # Low speed
            (100, 200),  # Medium speed
            (200, 300)   # High speed
        ]
        
        consistency_scores = []
        
        for speed_min, speed_max in speed_ranges:
            # Get brake pressures for this speed range
            mask = (speeds >= speed_min) & (speeds < speed_max)
            if any(mask):
                range_pressures = brake_pressures[mask]
                # Calculate consistency as inverse of variance
                consistency = 1 / (1 + np.var(range_pressures))
                consistency_scores.append(consistency)
                
        if not consistency_scores:
            return 0.5
            
        return np.mean(consistency_scores)
        
    def _calculate_temporal_characteristics(self, timestamps, brake_pressures):
        """Calculate temporal characteristics of brake applications"""
        try:
            # Convert timestamps to seconds if they're datetime
            if isinstance(timestamps[0], (pd.Timestamp, datetime)):
                timestamps = np.array([t.timestamp() for t in timestamps])
                
            # Calculate time intervals between brake applications
            brake_applications = np.where(np.diff(brake_pressures) > 0.1)[0]
            if len(brake_applications) < 2:
                return 0.5
                
            time_intervals = np.diff(timestamps[brake_applications])
            
            # Calculate regularity of brake applications
            temporal_regularity = 1 / (1 + np.std(time_intervals))
            
            # Calculate average duration of brake applications
            brake_durations = []
            for start_idx in brake_applications:
                end_idx = start_idx + 1
                while end_idx < len(brake_pressures) and brake_pressures[end_idx] > 0.1:
                    end_idx += 1
                duration = timestamps[end_idx - 1] - timestamps[start_idx]
                brake_durations.append(duration)
                
            avg_duration = np.mean(brake_durations)
            duration_consistency = 1 / (1 + np.std(brake_durations))
            
            # Combine metrics
            temporal_score = (
                0.4 * temporal_regularity +
                0.3 * (1 / (1 + abs(avg_duration - 2.0))) +  # Optimal duration around 2 seconds
                0.3 * duration_consistency
            )
            
            return np.clip(temporal_score, 0.0, 1.0)
            
        except Exception as e:
            print(f"Error in temporal analysis: {str(e)}")
            return 0.5
    
    def _is_brake_temp_optimal(self, brake_temps):
        """
        Analyze if brake temperatures are within optimal operating windows.
        
        Parameters:
        brake_temps : pd.Series or np.array
            Brake temperature readings in Celsius
        
        Returns:
        bool: True if temperatures are optimal, False otherwise
        """
        try:
            # Define temperature ranges
            TEMP_RANGES = {
                'optimal': (450, 750),    # Optimal operating window
                'warning': (350, 850),    # Warning range
                'critical': (200, 1000)   # Critical range
            }
            
            # Get current temperature
            if isinstance(brake_temps, pd.Series):
                current_temp = brake_temps.mean()
                temp_variance = brake_temps.std()
            else:
                current_temp = np.mean(brake_temps)
                temp_variance = np.std(brake_temps)
                
            # Store detailed temperature analysis
            self.brake_temp_analysis = {
                'current_temp': current_temp,
                'temp_variance': temp_variance,
                'status': self._get_temp_status(current_temp, TEMP_RANGES),
                'optimal_range': TEMP_RANGES['optimal'],
                'deviation': self._calculate_temp_deviation(current_temp, TEMP_RANGES['optimal']),
                'stability': self._calculate_temp_stability(temp_variance),
                'recommendations': []
            }
            
            # Add specific recommendations based on temperature
            self._add_temp_recommendations(current_temp, temp_variance, TEMP_RANGES)
            
            # Return True if within optimal range and stability is good
            is_optimal = (TEMP_RANGES['optimal'][0] <= current_temp <= TEMP_RANGES['optimal'][1] and
                        temp_variance < 50)  # Less than 50°C variance is considered stable
                        
            return is_optimal
            
        except Exception as e:
            print(f"Error analyzing brake temperatures: {str(e)}")
            return False
        
    def _get_temp_status(self, temp, ranges):
        """Determine temperature status"""
        if ranges['optimal'][0] <= temp <= ranges['optimal'][1]:
            return 'OPTIMAL'
        elif ranges['warning'][0] <= temp <= ranges['warning'][1]:
            if temp < ranges['optimal'][0]:
                return 'WARNING_COLD'
            else:
                return 'WARNING_HOT'
        elif ranges['critical'][0] <= temp <= ranges['critical'][1]:
            if temp < ranges['warning'][0]:
                return 'CRITICAL_COLD'
            else:
                return 'CRITICAL_HOT'
        else:
            return 'EXTREME'
            
    def _calculate_temp_deviation(self, temp, optimal_range):
        """Calculate deviation from optimal range"""
        if temp < optimal_range[0]:
            return temp - optimal_range[0]
        elif temp > optimal_range[1]:
            return temp - optimal_range[1]
        return 0
        
    def _calculate_temp_stability(self, variance):
        """Calculate temperature stability score"""
        # Convert variance to stability score (0-1)
        # Lower variance = higher stability
        return np.clip(1 - (variance / 100), 0, 1)
        
    def _add_temp_recommendations(self, temp, variance, ranges):
        """Add specific recommendations based on temperature analysis"""
        recommendations = self.brake_temp_analysis['recommendations']
        
        # Temperature too low
        if temp < ranges['optimal'][0]:
            if temp < ranges['warning'][0]:
                recommendations.append("URGENT: Increase brake energy - brakes severely cold")
                recommendations.append("Consider more aggressive brake warming procedure")
            else:
                recommendations.append("Increase brake energy to reach optimal temperature")
                recommendations.append("More frequent brake applications recommended")
                
        # Temperature too high
        elif temp > ranges['optimal'][1]:
            if temp > ranges['warning'][1]:
                recommendations.append("URGENT: Reduce brake energy - risk of brake fade")
                recommendations.append("Modify brake bias to reduce stress on hottest brakes")
            else:
                recommendations.append("Reduce brake energy to cool brakes")
                recommendations.append("Consider more brake cooling")
                
        # High variance
        if variance > 50:
            recommendations.append("Brake temperature inconsistent - review brake balance")
            
        # Optimal but close to edges
        if ranges['optimal'][0] <= temp <= ranges['optimal'][1]:
            if temp - ranges['optimal'][0] < 50:
                recommendations.append("Monitor brake temperatures - trending cold")
            elif ranges['optimal'][1] - temp < 50:
                recommendations.append("Monitor brake temperatures - trending hot")
                
    def get_brake_temp_report(self):
        """
        Generate detailed brake temperature report.
        Should be called after _is_brake_temp_optimal().
        
        Returns:
        dict: Detailed brake temperature analysis
        """
        if not hasattr(self, 'brake_temp_analysis'):
            return {"error": "No brake temperature analysis available"}
            
        return {
            'current_temperature': f"{self.brake_temp_analysis['current_temp']:.1f}°C",
            'status': self.brake_temp_analysis['status'],
            'deviation_from_optimal': f"{self.brake_temp_analysis['deviation']:.1f}°C",
            'temperature_stability': f"{self.brake_temp_analysis['stability']:.2f}",
            'recommendations': self.brake_temp_analysis['recommendations'],
            'optimal_range': f"{self.brake_temp_analysis['optimal_range'][0]}-{self.brake_temp_analysis['optimal_range'][1]}°C"
        }
        
    def _calculate_tire_grip_usage(self, data):
        """
        Calculate how effectively tire grip is being utilized based on multiple factors.
        
        Parameters:
        data : pd.DataFrame
            Telemetry data containing tire and grip-related information
            Expected columns: tire temperatures, pressures, slip angles, loads, etc.
            
        Returns:
        float: Tire grip usage efficiency (0.0 to 1.0)
        """
        try:
            # Initialize grip factors dictionary to store detailed analysis
            grip_factors = {
                'temperature_efficiency': 0.0,
                'pressure_efficiency': 0.0,
                'slip_efficiency': 0.0,
                'load_efficiency': 0.0,
                'compound_efficiency': 0.0
            }
            
            # 1. Temperature Analysis
            if all(col in data.columns for col in ['tire_temp_fl', 'tire_temp_fr']):
                grip_factors['temperature_efficiency'] = self._analyze_tire_temperatures(data)
                
            # 2. Pressure Analysis
            if all(col in data.columns for col in ['tire_pressure_fl', 'tire_pressure_fr']):
                grip_factors['pressure_efficiency'] = self._analyze_tire_pressures(data)
                
            # 3. Slip Angle Analysis
            if 'tire_slip_angle' in data.columns:
                grip_factors['slip_efficiency'] = self._analyze_slip_angles(data)
                
            # 4. Vertical Load Analysis
            if 'vertical_load' in data.columns:
                grip_factors['load_efficiency'] = self._analyze_tire_loads(data)
                
            # 5. Compound Analysis
            if 'tire_compound' in data.columns:
                grip_factors['compound_efficiency'] = self._analyze_compound_usage(data)
                
            # Calculate weighted average of available factors
            available_factors = {k: v for k, v in grip_factors.items() if v > 0}
            
            if not available_factors:
                return 0.5  # Default if no data available
                
            # Weight factors based on importance
            weights = {
                'temperature_efficiency': 0.3,
                'pressure_efficiency': 0.25,
                'slip_efficiency': 0.2,
                'load_efficiency': 0.15,
                'compound_efficiency': 0.1
            }
            
            # Calculate weighted average
            total_weight = sum(weights[k] for k in available_factors.keys())
            weighted_sum = sum(available_factors[k] * weights[k] for k in available_factors.keys())
            
            grip_usage = weighted_sum / total_weight
            
            # Store detailed analysis
            self.tire_grip_analysis = {
                'overall_grip_usage': grip_usage,
                'detailed_factors': grip_factors,
                'recommendations': self._generate_grip_recommendations(grip_factors)
            }
            
            return np.clip(grip_usage, 0.0, 1.0)
            
        except Exception as e:
            print(f"Error calculating tire grip usage: {str(e)}")
            return 0.5
        
    def _analyze_tire_temperatures(self, data):
        """
        Analyze tire temperature efficiency.
        Optimal temperature window depends on compound.
        """
        # Get tire temperatures
        temps = [
            data['tire_temp_fl'].iloc[0],
            data['tire_temp_fr'].iloc[0]
        ]
        
        # Define optimal temperature ranges (compound-specific if available)
        if 'tire_compound' in data.columns:
            optimal_range = self._get_compound_temp_range(data['tire_compound'].iloc[0])
        else:
            optimal_range = (80, 100)  # Default range in °C
            
        # Calculate how close temperatures are to optimal range
        temp_scores = [
            1.0 - min(abs(temp - sum(optimal_range)/2) / (optimal_range[1] - optimal_range[0]), 1.0)
            for temp in temps
        ]
        
        return np.mean(temp_scores)
        
    def _analyze_tire_pressures(self, data):
        """
        Analyze tire pressure efficiency.
        Optimal pressure depends on track conditions and tire load.
        """
        pressures = [
            data['tire_pressure_fl'].iloc[0],
            data['tire_pressure_fr'].iloc[0]
        ]
        
        # Define optimal pressure range (track-specific if available)
        if 'track_temp' in data.columns:
            optimal_range = self._get_optimal_pressure_range(data['track_temp'].iloc[0])
        else:
            optimal_range = (21.0, 23.0)  # Default range in PSI
            
        # Calculate pressure efficiency
        pressure_scores = [
            1.0 - min(abs(pressure - sum(optimal_range)/2) / (optimal_range[1] - optimal_range[0]), 1.0)
            for pressure in pressures
        ]
        
        return np.mean(pressure_scores)
        
    def _analyze_slip_angles(self, data):
        """
        Analyze tire slip angle efficiency.
        Optimal slip angle varies with speed and corner type.
        """
        slip_angle = data['tire_slip_angle'].iloc[0]
        
        # Get optimal slip angle range based on speed
        if 'speed' in data.columns:
            speed = data['speed'].iloc[0]
            optimal_range = self._get_optimal_slip_range(speed)
        else:
            optimal_range = (3.0, 5.0)  # Default range in degrees
            
        # Calculate slip efficiency
        slip_efficiency = 1.0 - min(abs(slip_angle - sum(optimal_range)/2) / 
                                  (optimal_range[1] - optimal_range[0]), 1.0)
        
        return slip_efficiency
        
    def _analyze_tire_loads(self, data):
        """
        Analyze vertical load efficiency.
        Optimal load depends on tire specification and track characteristics.
        """
        load = data['vertical_load'].iloc[0]
        
        # Define optimal load range
        if 'corner_type' in data.columns:
            optimal_range = self._get_optimal_load_range(data['corner_type'].iloc[0])
        else:
            optimal_range = (2000, 4000)  # Default range in Newtons
            
        # Calculate load efficiency
        load_efficiency = 1.0 - min(abs(load - sum(optimal_range)/2) / 
                                  (optimal_range[1] - optimal_range[0]), 1.0)
        
        return load_efficiency
        
    def _analyze_compound_usage(self, data):
        """
        Analyze tire compound usage efficiency.
        Different compounds have different optimal operating windows.
        """
        compound = data['tire_compound'].iloc[0]
        
        # Get compound-specific factors
        compound_factors = self._get_compound_factors(compound)
        
        # Calculate compound efficiency based on current conditions
        condition_score = self._calculate_compound_condition_score(data, compound_factors)
        
        return condition_score
        
    def _get_compound_temp_range(self, compound):
        """Get optimal temperature range for specific compound"""
        ranges = {
            'soft': (90, 110),
            'medium': (85, 105),
            'hard': (80, 100),
            'intermediate': (70, 90),
            'wet': (60, 80)
        }
        return ranges.get(compound.lower(), (80, 100))
        
    def _get_optimal_pressure_range(self, track_temp):
        """Get optimal pressure range based on track temperature"""
        base_range = (21.0, 23.0)
        temp_adjustment = (track_temp - 25) * 0.1  # 0.1 PSI per degree from 25°C
        return (base_range[0] + temp_adjustment, base_range[1] + temp_adjustment)
        
    def _get_optimal_slip_range(self, speed):
        """Get optimal slip angle range based on speed"""
        if speed > 200:
            return (2.0, 4.0)  # High speed range
        elif speed > 100:
            return (3.0, 5.0)  # Medium speed range
        else:
            return (4.0, 6.0)  # Low speed range
            
    def _get_optimal_load_range(self, corner_type):
        """Get optimal load range based on corner type"""
        ranges = {
            1: (3000, 5000),  # Slow corner
            2: (2500, 4500),  # Medium corner
            3: (2000, 4000)   # Fast corner
        }
        return ranges.get(corner_type, (2000, 4000))
        
    def _get_compound_factors(self, compound):
        """Get compound-specific performance factors"""
        return {
            'soft': {
                'peak_grip': 1.0,
                'temperature_sensitivity': 0.8,
                'wear_rate': 0.7
            },
            'medium': {
                'peak_grip': 0.9,
                'temperature_sensitivity': 0.6,
                'wear_rate': 0.8
            },
            'hard': {
                'peak_grip': 0.8,
                'temperature_sensitivity': 0.4,
                'wear_rate': 0.9
            }
        }.get(compound.lower(), {
            'peak_grip': 0.85,
            'temperature_sensitivity': 0.6,
            'wear_rate': 0.8
        })
        
    def _calculate_compound_condition_score(self, data, compound_factors):
        """Calculate compound-specific condition score"""
        score = compound_factors['peak_grip']
        
        if 'track_temp' in data.columns:
            temp_effect = 1.0 - compound_factors['temperature_sensitivity'] * \
                         abs(data['track_temp'].iloc[0] - 25) / 50
            score *= temp_effect
            
        return score
        
    def _generate_grip_recommendations(self, grip_factors):
        """Generate recommendations based on grip analysis"""
        recommendations = []
        
        if grip_factors['temperature_efficiency'] < 0.7:
            recommendations.append("Optimize tire temperature management")
            
        if grip_factors['pressure_efficiency'] < 0.7:
            recommendations.append("Adjust tire pressures to optimal range")
            
        if grip_factors['slip_efficiency'] < 0.7:
            recommendations.append("Modify driving style to optimize slip angles")
            
        if grip_factors['load_efficiency'] < 0.7:
            recommendations.append("Review suspension settings for better load distribution")
            
        if grip_factors['compound_efficiency'] < 0.7:
            recommendations.append("Consider compound strategy for conditions")
            
        return recommendations

    
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
        'track_grip': np.random.uniform(0.7, 1.0, n_samples),
        
        # New XGBoost-specific features
        'brake_pressure_history': np.random.uniform(0.4, 1.0, n_samples),  # Historical average brake pressure
        'entry_speed_delta': np.random.uniform(-20, 20, n_samples),  # Speed difference from ideal entry
        'downforce_level': np.random.uniform(0.6, 1.0, n_samples),  # Relative downforce setting
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