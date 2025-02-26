import streamlit as st
import pandas as pd
import numpy as np
from ballistics_model import BallisticsModel
import plotly.express as px
import plotly.graph_objects as go
from bullet_visualizer import BulletVisualizer
import cv2

def load_data():
    """Load and prepare the ballistics data"""
    df = pd.read_csv('dataset/ballistics.csv')
    df_300blk = pd.read_csv('dataset/detail_300BLK.csv')
    return df, df_300blk

def train_model(df):
    """Train the ballistics model"""
    model = BallisticsModel()
    X, y = model.prepare_data(df)
    model.train(X, y)
    model.save_model('ballistics_model.pkl')
    return model

def create_trajectory_plot(velocity_data, distances, drop_data=None):
    """Create a trajectory visualization"""
    fig = go.Figure()
    
    # Velocity trace
    fig.add_trace(go.Scatter(
        x=distances, 
        y=velocity_data,
        mode='lines+markers',
        name='Velocity (fps)',
        line=dict(color='#2ecc71')
    ))
    
    # Bullet drop trace if provided
    if drop_data is not None:
        fig.add_trace(go.Scatter(
            x=distances,
            y=drop_data,
            mode='lines+markers',
            name='Bullet Drop (inches)',
            line=dict(color='#e74c3c'),
            yaxis='y2'
        ))
    
    fig.update_layout(
        title='Bullet Trajectory Analysis',
        xaxis_title='Distance (yards)',
        yaxis_title='Velocity (fps)',
        yaxis2=dict(
            title='Bullet Drop (inches)',
            overlaying='y',
            side='right'
        ),
        template='plotly_dark',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    return fig

def main():
    st.set_page_config(
        page_title="Advanced Ballistics Analyzer",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS with improved styling
    st.markdown("""
        <style>
        .main {
            background-color: #1E1E1E;
            color: white;
        }
        .stButton>button {
            background-color: #2ecc71;
            color: white;
            width: 100%;
            border-radius: 10px;
            padding: 10px 20px;
            font-size: 18px;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
            padding: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            padding: 10px 20px;
            background-color: #2C3E50;
            border-radius: 10px;
        }
        .stAlert {
            background-color: #34495e;
            border: none;
            border-radius: 10px;
        }
        .css-1d391kg {
            padding: 1rem;
            border-radius: 10px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header with improved description
    st.title("🎯 Advanced Ballistics Analyzer")
    st.markdown("""
    ### Your Complete Ballistics Analysis Tool
    This advanced tool helps shooters and ammunition enthusiasts:
    - Calculate and predict ballistic coefficients
    - Analyze bullet trajectories and performance
    - Compare with existing ammunition
    - Understand ballistic behavior
    """)
    
    # Load data and train model
    df, df_300blk = load_data()
    
    try:
        model = BallisticsModel.load_model('ballistics_model.pkl')
    except:
        with st.spinner("Initializing ballistics model..."):
            model = train_model(df)
    
    # Enhanced sidebar with more parameters
    with st.sidebar:
        st.header("📊 Ammunition Parameters")
        
        # Basic Parameters Section
        st.subheader("Basic Parameters")
        st.info("Essential bullet characteristics")
        
        weight = st.number_input(
            "Bullet Weight (grains)", 
            min_value=20, 
            max_value=500, 
            value=150,
            help="The weight of the bullet in grains (1 grain = 1/7000 pound)"
        )
        
        velocity = st.number_input(
            "Muzzle Velocity (fps)", 
            min_value=500, 
            max_value=4000, 
            value=2800,
            help="The initial velocity of the bullet when it leaves the barrel"
        )
        
        # Advanced Parameters Section
        st.subheader("Advanced Parameters")
        st.info("Optional parameters for detailed analysis")
        
        caliber = st.selectbox(
            "Caliber",
            options=[".223", ".308", ".30-06", ".338", ".50"],
            help="Bullet diameter in inches"
        )
        
        bullet_type = st.selectbox(
            "Bullet Type",
            options=["FMJ", "HP", "SP", "BTHP", "AP"],
            help="Type of bullet construction"
        )
        
        barrel_length = st.slider(
            "Barrel Length (inches)",
            min_value=10,
            max_value=30,
            value=20,
            help="Length of the firearm barrel"
        )
    
    # Main content with enhanced tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Ballistic Calculator",
        "📊 Ammunition Analysis",
        "📈 Advanced Metrics",
        "🔍 Bullet Visualization",
        "ℹ️ Help & Information"
    ])
    
    with tab1:
        st.header("Ballistic Coefficient Calculator")
        
        # Quick explanation
        st.info("""
        The Ballistic Calculator helps you understand how your bullet will perform in flight.
        Enter the parameters in the sidebar and click calculate to begin.
        """)
        
        if st.button("Calculate Ballistics", key="calc_bc"):
            with st.spinner("Analyzing ballistic performance..."):
                # Prepare input data
                input_data = pd.DataFrame({
                    'Weight': [weight],
                    'V0': [velocity],
                    'V100': [velocity * 0.95],
                    'E0': [weight * velocity**2 / 450400],
                    'E100': [weight * (velocity * 0.95)**2 / 450400]
                })
                
                # Make prediction
                bc = model.predict(input_data)[0]
                
                # Results in columns
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Ballistic Coefficient", f"{bc:.3f}")
                with col2:
                    st.metric("Sectional Density", f"{weight/(7000*0.308*0.308):.3f}")
                with col3:
                    st.metric("Muzzle Energy (ft-lbs)", f"{(weight * velocity**2)/(450400):.0f}")
                
                # Trajectory Analysis
                st.subheader("Trajectory Analysis")
                with st.expander("View Detailed Trajectory", expanded=True):
                    # Calculate trajectory
                    distances = np.array([0, 100, 200, 300, 400, 500])
                    velocities = velocity * (0.95 ** (distances/100))
                    drops = -0.5 * 32.2 * (distances/velocity)**2 * 12  # Simple bullet drop calculation
                    
                    fig = create_trajectory_plot(velocities, distances, drops)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Trajectory data table
                    trajectory_data = pd.DataFrame({
                        'Distance (yards)': distances,
                        'Velocity (fps)': velocities.round(0),
                        'Energy (ft-lbs)': (weight * velocities**2/450400).round(0),
                        'Drop (inches)': drops.round(1)
                    })
                    st.dataframe(trajectory_data, use_container_width=True)
    
    with tab2:
        st.header("Ammunition Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Similar Ammunition")
            st.info("Find commercial ammunition with similar characteristics")
            
            # Find similar ammunition
            df['Weight_diff'] = abs(df['Weight'] - weight)
            df['V0_diff'] = abs(df['V0'] - velocity)
            similar_ammo = df.nsmallest(5, ['Weight_diff', 'V0_diff'])
            
            st.dataframe(
                similar_ammo[['Type', 'Weight', 'V0', 'BC', 'E0']],
                use_container_width=True
            )
        
        with col2:
            st.subheader("Statistical Comparison")
            st.info("See how your ammunition compares to others")
            
            stats_fig = px.box(
                df, 
                y='BC',
                points="all",
                title="Ballistic Coefficient Distribution"
            )
            stats_fig.update_layout(template='plotly_dark')
            st.plotly_chart(stats_fig, use_container_width=True)
    
    with tab3:
        st.header("Advanced Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Performance Indicators")
            
            # Calculate additional metrics
            energy_retention = ((weight * (velocity * 0.95)**2) / (weight * velocity**2)) * 100
            velocity_loss_rate = (velocity - (velocity * 0.95)) / 100
            
            st.metric("Energy Retention at 100y", f"{energy_retention:.1f}%")
            st.metric("Velocity Loss Rate", f"{velocity_loss_rate:.1f} fps/yard")
            
            # Calculate BC for performance assessment
            input_data = pd.DataFrame({
                'Weight': [weight],
                'V0': [velocity],
                'V100': [velocity * 0.95],
                'E0': [weight * velocity**2 / 450400],
                'E100': [weight * (velocity * 0.95)**2 / 450400]
            })
            
            bc = model.predict(input_data)[0]
            
            # Display aerodynamic efficiency
            st.subheader("Aerodynamic Efficiency")
            if bc > 0.5:
                st.success(f"High aerodynamic efficiency (BC: {bc:.3f})")
            elif bc > 0.3:
                st.info(f"Moderate aerodynamic efficiency (BC: {bc:.3f})")
            else:
                st.warning(f"Low aerodynamic efficiency (BC: {bc:.3f})")
            
            # Add explanation
            st.markdown("""
            **Efficiency Ranges:**
            - High: BC > 0.5
            - Moderate: 0.3 < BC ≤ 0.5
            - Low: BC ≤ 0.3
            """)
        
        with col2:
            st.subheader("Environmental Effects")
            
            temperature = st.slider("Temperature (°F)", -20, 120, 59)
            altitude = st.slider("Altitude (ft)", 0, 10000, 0)
            
            # Simple environmental corrections
            velocity_temp_factor = 1 + (temperature - 59) * 0.002
            velocity_alt_factor = 1 + (altitude / 1000) * 0.003
            
            adjusted_velocity = velocity * velocity_temp_factor * velocity_alt_factor
            
            st.metric(
                "Temperature Adjusted Velocity",
                f"{adjusted_velocity:.0f} fps",
                f"{adjusted_velocity - velocity:+.0f} fps"
            )
            
            # Add environmental effects explanation
            st.markdown("""
            **Environmental Corrections:**
            - Temperature: ±0.2% per °F from 59°F
            - Altitude: +0.3% per 1000 ft
            """)
    
    with tab4:
        st.header("3D Bullet Visualization")
        
        st.info("""
        This section provides an interactive 3D visualization of your bullet.
        - Use mouse to rotate the view
        - Scroll to zoom
        - Double click to reset view
        The model is an approximation and may not exactly match actual bullet designs.
        """)
        
        try:
            visualizer = BulletVisualizer()
            
            # Create visualization
            with st.spinner("Generating 3D bullet visualization..."):
                fig = visualizer.create_visualization(
                    weight=weight,
                    caliber=caliber,
                    bullet_type=bullet_type
                )
            
            if fig is not None:
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    # Display the interactive 3D plot
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("Bullet Specifications")
                    st.markdown(f"""
                    - **Weight:** {weight} grains
                    - **Caliber:** {caliber}
                    - **Type:** {bullet_type}
                    - **Estimated Length:** {(weight/(7000 * 0.321 * np.pi * (float(caliber.replace('.', '')) / 200)**2)):.3f} inches
                    """)
                    
                    st.subheader("Type Characteristics")
                    if bullet_type == "FMJ":
                        st.markdown("""
                        - Full metal jacket design
                        - Consistent profile
                        - Standard target ammunition
                        """)
                    elif bullet_type == "HP":
                        st.markdown("""
                        - Hollow point cavity
                        - Expansion on impact
                        - Defensive ammunition
                        """)
                    elif bullet_type == "SP":
                        st.markdown("""
                        - Soft point design
                        - Controlled expansion
                        - Hunting ammunition
                        """)
                    elif bullet_type == "BTHP":
                        st.markdown("""
                        - Boat tail design
                        - Hollow point
                        - Match-grade accuracy
                        """)
                    elif bullet_type == "AP":
                        st.markdown("""
                        - Armor piercing design
                        - Penetrator core
                        - Military/specialized use
                        """)
            else:
                st.error("Could not generate bullet visualization. Please check your input parameters.")
                
        except Exception as e:
            st.error(f"Error in bullet visualization: {str(e)}")
            st.info("Please try adjusting the input parameters or refresh the page.")
    
    with tab5:
        st.header("Help & Information")
        
        with st.expander("Quick Start Guide", expanded=True):
            st.markdown("""
            ### Getting Started
            1. **Enter Basic Parameters** (sidebar)
               - Bullet Weight
               - Muzzle Velocity
            
            2. **Optional: Add Advanced Parameters**
               - Caliber
               - Bullet Type
               - Barrel Length
            
            3. **Click 'Calculate Ballistics'**
            
            4. **Explore Results**
               - View trajectory
               - Compare with similar ammunition
               - Check advanced metrics
            """)
        
        with st.expander("Understanding Ballistic Coefficient (BC)"):
            st.markdown("""
            ### Ballistic Coefficient Explained
            - **What is BC?** A measure of a bullet's ability to overcome air resistance
            - **Higher BC** (>0.5): Better long-range performance
            - **Lower BC** (<0.3): More affected by wind and drag
            
            #### Factors Affecting BC:
            - Bullet weight
            - Bullet shape
            - Bullet length
            - Cross-sectional area
            """)
        
        with st.expander("Terminology Guide"):
            st.markdown("""
            ### Common Ballistics Terms
            
            #### Basic Terms
            - **FPS**: Feet Per Second
            - **Grain**: Unit of weight (1/7000 pound)
            - **BC**: Ballistic Coefficient
            
            #### Advanced Terms
            - **Sectional Density**: Weight to caliber ratio
            - **Muzzle Energy**: Kinetic energy at the muzzle
            - **Trajectory**: Path of the bullet
            
            #### Bullet Types
            - **FMJ**: Full Metal Jacket
            - **HP**: Hollow Point
            - **SP**: Soft Point
            - **BTHP**: Boat Tail Hollow Point
            - **AP**: Armor Piercing
            """)
        
        with st.expander("Advanced Features"):
            st.markdown("""
            ### Advanced Analysis Features
            
            #### 1. Trajectory Analysis
            - Velocity at different ranges
            - Bullet drop calculation
            - Energy retention
            
            #### 2. Environmental Corrections
            - Temperature effects
            - Altitude adjustments
            - Air density impacts
            
            #### 3. Comparative Analysis
            - Similar ammunition finder
            - Statistical comparison
            - Performance metrics
            """)

if __name__ == "__main__":
    main() 