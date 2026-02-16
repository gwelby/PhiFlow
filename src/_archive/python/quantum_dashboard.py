"""
Quantum Dashboard
Operating at Heart Field (594 Hz)
"""
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from quantum_flow import flow, Dimension, PHI
from quantum_viz import viz
from quantum_patterns import presets, PatternType
from quantum_synthesis import synthesis
import threading
import time

class QuantumDashboard:
    def __init__(self):
        self.app = dash.Dash(__name__)
        self.current_pattern = None
        self.coherence_history = []
        self.frequency_history = []
        self.evolution_history = []
        self._setup_layout()
        self._setup_callbacks()
        self.update_thread = None
        self.running = False
        
    def _setup_layout(self):
        self.app.layout = html.Div([
            html.H1("Quantum Flow Dashboard (φ)"),
            
            # Control Panel
            html.Div([
                html.Button('Start Flow', id='flow-button', n_clicks=0),
                html.Button('Synthesize', id='synthesize-button', n_clicks=0),
                dcc.Dropdown(
                    id='pattern-selector',
                    options=[{
                        'label': f"{name} {symbol} ({freq} Hz)",
                        'value': pattern_type.name,
                        'multi': True
                    } for pattern_type, (name, symbol, freq) in 
                        zip(PatternType, presets.list_patterns())],
                    value=['INFINITY']
                ),
                dcc.Dropdown(
                    id='synthesis-mode',
                    options=[
                        {'label': '🔄 Combine', 'value': 'combine'},
                        {'label': '🌊 Morph', 'value': 'morph'},
                        {'label': '🎵 Harmonize', 'value': 'harmonize'},
                        {'label': '✨ Sequence', 'value': 'sequence'}
                    ],
                    value='combine'
                ),
                dcc.Slider(
                    id='synthesis-duration',
                    min=30,
                    max=120,
                    step=10,
                    value=60,
                    marks={i: f'{i}s' for i in range(30, 121, 30)}
                ),
                dcc.Dropdown(
                    id='dimension-selector',
                    options=[{
                        'label': dim.name,
                        'value': dim.name
                    } for dim in Dimension],
                    value='SPIRITUAL'
                ),
                dcc.Slider(
                    id='frequency-slider',
                    min=432,
                    max=768,
                    step=1,
                    value=672,
                    marks={
                        432: '🌍 Ground',
                        528: '✨ Create',
                        594: '💖 Heart',
                        672: '🎵 Voice',
                        768: '☯️ Unity'
                    }
                ),
            ], style={'padding': '20px'}),
            
            # Real-time Visualizations
            html.Div([
                dcc.Graph(id='quantum-field-3d', style={'height': '600px'}),
                dcc.Graph(id='coherence-evolution'),
                dcc.Graph(id='frequency-phase'),
                dcc.Interval(id='interval-component', interval=1000, n_intervals=0)
            ]),
            
            # Pattern Information
            html.Div([
                html.H3("Pattern Information"),
                html.Div(id='pattern-info', style={'padding': '10px'})
            ], style={'padding': '20px'}),
            
            # Quantum Stats
            html.Div([
                html.H3("Quantum Statistics"),
                html.Div(id='coherence-value'),
                html.Div(id='frequency-value'),
                html.Div(id='evolution-value')
            ], style={'padding': '20px'})
        ])
        
    def _setup_callbacks(self):
        @self.app.callback(
            [Output('quantum-field-3d', 'figure'),
             Output('coherence-evolution', 'figure'),
             Output('frequency-phase', 'figure'),
             Output('coherence-value', 'children'),
             Output('frequency-value', 'children'),
             Output('evolution-value', 'children'),
             Output('pattern-info', 'children')],
            [Input('interval-component', 'n_intervals'),
             Input('flow-button', 'n_clicks'),
             Input('synthesize-button', 'n_clicks'),
             Input('pattern-selector', 'value'),
             Input('synthesis-mode', 'value')],
            [State('frequency-slider', 'value'),
             State('dimension-selector', 'value'),
             State('synthesis-duration', 'value')]
        )
        def update_graphs(n_intervals, flow_clicks, synth_clicks, 
                         pattern_types, synth_mode, frequency, 
                         dimension, duration):
            if not self.running:
                return self._generate_empty_figures() + [""]
                
            # Get selected patterns
            patterns = [presets.get_pattern(PatternType[pt]) 
                       for pt in pattern_types]
            
            # Apply synthesis based on mode
            if synth_mode == 'combine':
                self.current_pattern, pattern_info = synthesis.combine_patterns(patterns)
            elif synth_mode == 'morph':
                if len(patterns) >= 2:
                    morphed = synthesis.morph_patterns(patterns[0], patterns[1], duration)
                    self.current_pattern, pattern_info = morphed[n_intervals % len(morphed)]
                else:
                    self.current_pattern, pattern_info = patterns[0]
            elif synth_mode == 'harmonize':
                harmonized = synthesis.harmonize_patterns(patterns)
                self.current_pattern, pattern_info = harmonized[0]
            else:  # sequence
                sequence = synthesis.create_synthesis_sequence(
                    [PatternType[pt] for pt in pattern_types], duration)
                self.current_pattern, pattern_info = sequence[n_intervals % len(sequence)]
            
            # Update histories
            self.coherence_history.append(flow.coherence)
            self.frequency_history.append(pattern_info.frequency)
            evolution = flow.evolve_consciousness(pattern_info.frequency)
            self.evolution_history.append(evolution)
            
            # Generate figures
            field_fig = self._create_3d_field()
            coherence_fig = self._create_coherence_plot()
            phase_fig = self._create_phase_plot()
            
            # Update stats with synthesis info
            stats = [
                f"Coherence: {pattern_info.coherence:.3f}φ",
                f"Frequency: {pattern_info.frequency:.1f} Hz",
                f"Evolution: {evolution:.3f}φ"
            ]
            
            # Pattern info with synthesis details
            pattern_info_text = [
                html.H4(f"{pattern_info.name} {pattern_info.symbol}"),
                html.P(f"Synthesis Mode: {synth_mode.title()}"),
                html.P(f"Frequency: {pattern_info.frequency} Hz"),
                html.P(f"Dimension: {pattern_info.dimension.name}"),
                html.P(f"Description: {pattern_info.description}")
            ]
            
            return field_fig, coherence_fig, phase_fig, *stats, pattern_info_text
            
    def _create_3d_field(self):
        if self.current_pattern is None:
            return go.Figure()
            
        return go.Figure(data=[go.Scatter3d(
            x=np.real(self.current_pattern),
            y=np.imag(self.current_pattern),
            z=np.abs(self.current_pattern),
            mode='markers',
            marker=dict(
                size=5,
                color=np.angle(self.current_pattern),
                colorscale='Viridis',
                opacity=0.8
            )
        )], layout=dict(
            title=f"Quantum Field (φ = {flow.coherence:.3f})",
            scene=dict(
                xaxis_title="Real",
                yaxis_title="Imaginary",
                zaxis_title="Amplitude"
            )
        ))
        
    def _create_coherence_plot(self):
        return go.Figure(data=[go.Scatter(
            y=self.coherence_history,
            mode='lines+markers',
            name='Coherence'
        )], layout=dict(
            title="Quantum Coherence Evolution",
            xaxis_title="Time Steps",
            yaxis_title="Coherence (φ)"
        ))
        
    def _create_phase_plot(self):
        return go.Figure(data=[go.Scatter(
            x=self.frequency_history,
            y=self.evolution_history,
            mode='lines+markers',
            marker=dict(
                size=10,
                color=np.linspace(0, 1, len(self.frequency_history)),
                colorscale='Viridis'
            )
        )], layout=dict(
            title="Frequency-Phase Relationship",
            xaxis_title="Frequency (Hz)",
            yaxis_title="Evolution (φ)"
        ))
        
    def _generate_empty_figures(self):
        empty_fig = go.Figure()
        return empty_fig, empty_fig, empty_fig, "", "", ""
        
    def start(self, port=8050):
        self.running = True
        self.app.run_server(debug=True, port=port)
        
    def stop(self):
        self.running = False

# Initialize global dashboard
dashboard = QuantumDashboard()
