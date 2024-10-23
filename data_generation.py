import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_telemetry_data():
    """
    Generate realistic F1 telemetry data for one lap
    """
    # Create base timestamp array for one lap (approximately 80 seconds, 100Hz sampling)
    num_samples = 8000
    base_time = datetime(2024, 3, 15, 14, 30, 0)
    timestamps = [base_time + timedelta(milliseconds=10*i) for i in range(num_samples)]
    
    # Initialize random seed for reproducibility
    np.random.seed(42)
    
    # Generate distance array (5.5km track)
    distance = np.linspace(0, 5500, num_samples)
    
    # Generate speed profile (realistic F1 speeds between 80 and 340 km/h)
    speed = generate_speed_profile(num_samples)
    
    # Generate throttle position (0-100%)
    throttle = generate_throttle_profile(num_samples, speed)
    
    # Generate brake pressure (0-100%)
    brake = generate_brake_profile(num_samples, speed)
    
    # Generate steering angle (-40 to 40 degrees)
    steering = generate_steering_profile(num_samples)
    
    # Generate gear (1-8)
    gear = generate_gear_profile(num_samples, speed)
    
    # Generate engine RPM (4000-15000)
    rpm = generate_rpm_profile(num_samples, gear, speed)
    
    # Generate tire temperatures (60-110°C)
    tire_temp_fl = generate_tire_temp_profile(num_samples, speed, 'FL')
    tire_temp_fr = generate_tire_temp_profile(num_samples, speed, 'FR')
    tire_temp_rl = generate_tire_temp_profile(num_samples, speed, 'RL')
    tire_temp_rr = generate_tire_temp_profile(num_samples, speed, 'RR')
    
    # Generate GPS coordinates (fictional track)
    gps_lat, gps_lon = generate_gps_coordinates(num_samples)
    
    # Create DataFrame
    telemetry = pd.DataFrame({
        'timestamp': timestamps,
        'distance': distance,
        'speed': speed,
        'throttle_position': throttle,
        'brake_pressure': brake,
        'steering_angle': steering,
        'gear': gear,
        'engine_rpm': rpm,
        'tire_temp_fl': tire_temp_fl,
        'tire_temp_fr': tire_temp_fr,
        'tire_temp_rl': tire_temp_rl,
        'tire_temp_rr': tire_temp_rr,
        'gps_latitude': gps_lat,
        'gps_longitude': gps_lon,
        'drs': generate_drs_status(num_samples, distance),
        'ers_deployment': generate_ers_profile(num_samples, speed),
        'fuel_flow': generate_fuel_flow(num_samples, throttle),
        'g_lat': generate_lateral_g(num_samples, speed, steering),
        'g_lon': generate_longitudinal_g(num_samples, speed)
    })
    
    return telemetry

def generate_track_map():
    """
    Generate track mapping data including corner numbers, types, and racing line
    """
    # Define track segments (approximately matching the telemetry)
    num_points = 1000  # Less dense than telemetry data
    
    # Generate base coordinates for track centerline
    track_distance = np.linspace(0, 5500, num_points)
    gps_lat, gps_lon = generate_gps_coordinates(num_points)
    
    # Define corner data
    corners = define_corner_data()
    
    # Generate ideal racing line (offset from centerline)
    racing_line_lat, racing_line_lon = generate_racing_line(gps_lat, gps_lon)
    
    track_map = pd.DataFrame({
        'distance': track_distance,
        'centerline_lat': gps_lat,
        'centerline_lon': gps_lon,
        'racing_line_lat': racing_line_lat,
        'racing_line_lon': racing_line_lon,
        'track_width': generate_track_width(num_points),
        'corner_number': assign_corner_numbers(track_distance, corners),
        'corner_type': assign_corner_types(track_distance, corners),
        'drs_zone': generate_drs_zones(track_distance),
        'slope': generate_track_slope(num_points),
        'surface_grip': generate_surface_grip(num_points)
    })
    
    return track_map

def generate_speed_profile(num_samples):
    """Generate realistic F1 speed profile"""
    base_speed = np.zeros(num_samples)
    
    # Define speed zones
    zones = [
        (0, 500, 280, 320),    # Start/finish straight
        (500, 800, 80, 120),   # Turn 1
        (800, 1200, 180, 220), # Short straight
        (1200, 1500, 90, 130), # Turn 2
        (1500, 1800, 180, 220), # Short straight
        (1800, 2200, 180, 220), # Short straight
        (2200, 2500, 90, 130), # Turn 3
        (2500, 3000, 180, 220), # Long straight
        (3000, 3500, 80, 120), # Turn 4
        (3500, 4000, 280, 320), # Start/finish straight
    ]
    
    for start, end, min_speed, max_speed in zones:
        idx_start = int(start * num_samples/5500)
        idx_end = int(end * num_samples/5500)
        segment_length = idx_end - idx_start
        
        # Generate smooth speed transition
        speed_segment = np.linspace(min_speed, max_speed, segment_length)
        speed_segment += np.random.normal(0, 1, segment_length)  # Add small variations
        base_speed[idx_start:idx_end] = speed_segment
    
    return np.clip(base_speed, 80, 340)

# Helper functions for generating other telemetry components
def generate_throttle_profile(num_samples, speed):
    base_throttle = np.where(speed > 200, 100, 70 + speed/4)
    return np.clip(base_throttle + np.random.normal(0, 2, num_samples), 0, 100)

def generate_brake_profile(num_samples, speed):
    brake = np.zeros(num_samples)
    speed_diff = np.diff(speed, prepend=speed[0])
    brake[speed_diff < -1] = -speed_diff[speed_diff < -1] * 2
    return np.clip(brake, 0, 100)

def generate_steering_profile(num_samples):
    steering = np.zeros(num_samples)
    # Add corner sequences
    corner_points = [(500, 800), (1200, 1500)]  # Matching speed profile zones
    for start, end in corner_points:
        idx_start = int(start * num_samples/5500)
        idx_end = int(end * num_samples/5500)
        steering[idx_start:idx_end] = 30 * np.sin(np.linspace(0, np.pi, idx_end-idx_start))
    return steering

def generate_gear_profile(num_samples, speed):
    # Basic gear calculation based on speed
    gears = np.floor(speed/340 * 8) + 1
    return np.clip(gears, 1, 8).astype(int)

def generate_rpm_profile(num_samples, gear, speed):
    # Approximate RPM based on gear and speed
    rpm_base = speed * 60  # Base conversion
    rpm = rpm_base * (9-gear)/2 + 4000  # Adjust for gear
    return np.clip(rpm, 4000, 15000)

def generate_tire_temp_profile(num_samples, speed, position):
    base_temp = 80 + speed/10
    if position in ['FL', 'FR']:  # Front tires
        base_temp += 5
    return np.clip(base_temp + np.random.normal(0, 2, num_samples), 60, 110)

def generate_gps_coordinates(num_samples):
    # Generate simple oval track coordinates
    t = np.linspace(0, 2*np.pi, num_samples)
    center_lat, center_lon = 45.0, 7.0  # Example location
    radius_lat, radius_lon = 0.01, 0.015
    
    lat = center_lat + radius_lat * np.cos(t)
    lon = center_lon + radius_lon * np.sin(t)
    return lat, lon

def generate_racing_line(center_lat, center_lon):
    # Offset from centerline to create racing line
    racing_lat = center_lat + 0.0001 * np.sin(np.linspace(0, 4*np.pi, len(center_lat)))
    racing_lon = center_lon + 0.0001 * np.cos(np.linspace(0, 4*np.pi, len(center_lon)))
    return racing_lat, racing_lon

def generate_track_width(num_points):
    # Generate varying track width (12-15 meters)
    return np.clip(13 + np.random.normal(0, 0.5, num_points), 12, 15)

def define_corner_data():
    return [
        {'start': 500, 'end': 800, 'number': 1, 'type': 'Heavy Braking'},
        {'start': 1200, 'end': 1500, 'number': 2, 'type': 'Medium Speed'},
        {'start': 1800, 'end': 2200, 'number': 3, 'type': 'Medium Speed'},
        {'start': 2500, 'end': 3000, 'number': 4, 'type': 'Medium Speed'},
        {'start': 3500, 'end': 4000, 'number': 5, 'type': 'High Speed'},
        # Add more corners as needed
    ]

def assign_corner_numbers(distance, corners):
    corner_numbers = np.zeros_like(distance)
    for corner in corners:
        mask = (distance >= corner['start']) & (distance <= corner['end'])
        corner_numbers[mask] = corner['number']
    return corner_numbers

def assign_corner_types(distance, corners):
    corner_types = [''] * len(distance)
    for corner in corners:
        mask = (distance >= corner['start']) & (distance <= corner['end'])
        corner_types = [corner['type'] if m else t for t, m in zip(corner_types, mask)]
    return corner_types

def generate_drs_zones(distance):
    drs_zones = np.zeros_like(distance)
    # Define DRS zones (example: 2 zones)
    drs_zones[(distance >= 0) & (distance <= 500)] = 1
    drs_zones[(distance >= 2500) & (distance <= 3000)] = 1
    return drs_zones

def generate_track_slope(num_points):
    # Generate track elevation changes (-3% to +3%)
    return np.clip(np.random.normal(0, 1, num_points), -3, 3)

def generate_surface_grip(num_points):
    # Generate grip levels (0.8-1.0)
    return np.clip(0.9 + np.random.normal(0, 0.05, num_points), 0.8, 1.0)

def generate_drs_status(num_samples, distance):
    drs = np.zeros(num_samples)
    # DRS zones (matching track map)
    drs[(distance >= 0) & (distance <= 500)] = 1
    drs[(distance >= 2500) & (distance <= 3000)] = 1
    return drs

def generate_ers_profile(num_samples, speed):
    # Generate ERS deployment (0-100%)
    base_ers = np.where(speed > 200, 80, 40)
    return np.clip(base_ers + np.random.normal(0, 5, num_samples), 0, 100)

def generate_fuel_flow(num_samples, throttle):
    # Generate fuel flow rate (0-100 kg/hr)
    return np.clip(throttle * 0.8 + np.random.normal(0, 2, num_samples), 0, 100)

def generate_lateral_g(num_samples, speed, steering):
    # Generate lateral G-force (-4 to 4 G)
    return np.clip(steering * speed / 1000, -4, 4)

def generate_longitudinal_g(num_samples, speed):
    # Generate longitudinal G-force (-5 to 2 G)
    speed_diff = np.diff(speed, prepend=speed[0])
    return np.clip(speed_diff / 10, -5, 2)

# Generate and save the data
telemetry = generate_telemetry_data()
track_map = generate_track_map()

# Save to CSV files
telemetry.to_csv('data/telemetry.csv', index=False)
track_map.to_csv('data/track_map.csv', index=False)

print("Sample data stats:")
print("\nTelemetry data shape:", telemetry.shape)
print("\nTelemetry columns:", telemetry.columns.tolist())
print("\nTrack map data shape:", track_map.shape)
print("\nTrack map columns:", track_map.columns.tolist())

# Display sample statistics
print("\nTelemetry Statistics:")
print(telemetry.describe())
print("\nTrack Map Statistics:")
print(track_map.describe())