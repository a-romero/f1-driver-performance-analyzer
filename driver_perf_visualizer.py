import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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