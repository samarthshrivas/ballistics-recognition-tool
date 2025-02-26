import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class BulletVisualizer:
    def __init__(self):
        self.bullet_types = {
            "FMJ": {"nose_length": 0.3, "boat_tail": False},
            "HP": {"nose_length": 0.2, "cavity_depth": 0.15},
            "SP": {"nose_length": 0.25, "flat_nose": True},
            "BTHP": {"nose_length": 0.35, "boat_tail": True},
            "AP": {"nose_length": 0.4, "pointed": True}
        }
    
    def generate_bullet_profile(self, weight, caliber, bullet_type):
        """Generate 3D bullet profile"""
        try:
            # Calculate dimensions
            caliber_inch = float(caliber.replace(".", "")) / 100
            density = 0.321  # lb/in³ (approximate for lead)
            volume = weight / (7000 * density)  # in³
            radius = caliber_inch / 2
            length = volume / (np.pi * radius ** 2)
            
            # Get bullet type parameters
            params = self.bullet_types.get(bullet_type, {"nose_length": 0.3})
            
            # Generate profile points
            theta = np.linspace(0, 2*np.pi, 100)  # Increased resolution
            z = np.linspace(0, length, 100)
            theta, z = np.meshgrid(theta, z)
            
            # Basic cylinder
            r = np.ones_like(theta) * radius
            
            # Modify nose based on bullet type
            nose_length = length * params.get("nose_length", 0.3)
            nose_start = length - nose_length
            nose_mask = z > nose_start
            
            if bullet_type == "HP":
                cavity_depth = params.get("cavity_depth", 0.15) * length
                cavity_mask = (z > (length - cavity_depth))
                r[cavity_mask] *= (1 - (z[cavity_mask] - (length - cavity_depth))/cavity_depth * 0.5)
            
            elif bullet_type in ["AP", "BTHP"]:
                r[nose_mask] *= (1 - (z[nose_mask] - nose_start)/nose_length)
            
            elif bullet_type == "SP":
                r[nose_mask] *= 0.9
            
            # Convert to Cartesian coordinates
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            
            return x, y, z, length, radius
            
        except Exception as e:
            print(f"Error generating bullet profile: {e}")
            return None, None, None, None, None
    
    def create_visualization(self, weight, caliber, bullet_type):
        """Create interactive 3D visualization"""
        x, y, z, length, radius = self.generate_bullet_profile(weight, caliber, bullet_type)
        
        if x is None:
            return None
        
        # Create figure with subplots
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'surface'}, {'type': 'surface'}]],
            subplot_titles=('Side View', 'Top View'),
            horizontal_spacing=0.02
        )
        
        # Add surface plots
        # Side View
        fig.add_trace(
            go.Surface(
                x=x, y=y, z=z,
                colorscale='Viridis',
                showscale=False,
                lighting=dict(
                    ambient=0.8,
                    diffuse=0.9,
                    fresnel=0.2,
                    specular=1,
                    roughness=0.4
                ),
                hoverinfo='skip'
            ),
            row=1, col=1
        )
        
        # Top View (rotated 90 degrees)
        fig.add_trace(
            go.Surface(
                x=x, y=y, z=z,
                colorscale='Viridis',
                showscale=False,
                lighting=dict(
                    ambient=0.8,
                    diffuse=0.9,
                    fresnel=0.2,
                    specular=1,
                    roughness=0.4
                ),
                hoverinfo='skip'
            ),
            row=1, col=2
        )
        
        # Update layout with better camera positions and aspect ratios
        fig.update_layout(
            title=dict(
                text=f"{bullet_type} Bullet - {weight}gr {caliber}",
                x=0.5,
                y=0.95,
                xanchor='center',
                yanchor='top',
                font=dict(size=20)
            ),
            scene=dict(
                camera=dict(
                    eye=dict(x=2.5, y=0, z=0),  # Side view
                    up=dict(x=0, y=0, z=1)
                ),
                aspectratio=dict(x=1, y=1, z=3),
                xaxis=dict(showgrid=True, showbackground=True, gridcolor='white', gridwidth=1),
                yaxis=dict(showgrid=True, showbackground=True, gridcolor='white', gridwidth=1),
                zaxis=dict(showgrid=True, showbackground=True, gridcolor='white', gridwidth=1)
            ),
            scene2=dict(
                camera=dict(
                    eye=dict(x=0, y=0, z=2.5),  # Top view
                    up=dict(x=0, y=1, z=0)
                ),
                aspectratio=dict(x=1, y=1, z=3),
                xaxis=dict(showgrid=True, showbackground=True, gridcolor='white', gridwidth=1),
                yaxis=dict(showgrid=True, showbackground=True, gridcolor='white', gridwidth=1),
                zaxis=dict(showgrid=True, showbackground=True, gridcolor='white', gridwidth=1)
            ),
            showlegend=False,
            template="plotly_dark",
            margin=dict(t=100, b=20, l=20, r=20),
            height=600  # Increased height for better visibility
        )
        
        # Add dimension annotations
        fig.add_annotation(
            text=f"Length: {length:.3f} inches",
            xref="paper", yref="paper",
            x=0.02, y=1.05,
            showarrow=False,
            font=dict(size=14)
        )
        
        fig.add_annotation(
            text=f"Diameter: {radius*2:.3f} inches",
            xref="paper", yref="paper",
            x=0.02, y=1.02,
            showarrow=False,
            font=dict(size=14)
        )
        
        return fig 