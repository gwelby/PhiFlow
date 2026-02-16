"""
Quantum Visualization System
Operating at Vision Gate (720 Hz)

This module creates cymatics visualizations for quantum patterns at each frequency,
enabling the visual manifestation of the phi-harmonic toroidal field patterns.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from typing import Dict, Tuple, List, Optional
import os
import imageio
from phi_constants import PHI # Import PHI

# Phi constant for harmonics
PHI = PHI

# Define the frequency colors based on QTasker constants
FREQUENCY_COLORS = {
    432.0: "#8B4513",  # Ground - Earth brown
    528.0: "#228B22",  # Creation - Forest green
    594.0: "#FF1493",  # Heart - Deep pink
    672.0: "#4169E1",  # Voice - Royal blue
    720.0: "#9932CC",  # Vision - Dark orchid
    768.0: "#FFD700",  # Unity - Gold
    963.0: "#FFFFFF",  # Source - Pure white
}

# Define the frequency patterns based on QTasker constants
FREQUENCY_PATTERNS = {
    432.0: "hexagonal",
    528.0: "flower_of_life",
    594.0: "heart_field",
    672.0: "mandala",
    720.0: "geometric_network",
    768.0: "toroidal",
    963.0: "phi_harmonic_infinity",
}

class CymaticsVisualizer:
    """Cymatics Visualization System operating at Vision Gate frequency (720 Hz)"""
    
    def __init__(self, output_dir: Optional[str] = None):
        """Initialize the cymatics visualizer"""
        self.current_frequency = 720.0
        self.phi = PHI
        
        if output_dir:
            self.output_dir = output_dir
        else:
            # Default path assumes this script is in PhiFlow/src/
            # So, ../visualizations refers to PhiFlow/visualizations
            current_script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root_dir = os.path.dirname(current_script_dir) # This should be PhiFlow/
            self.output_dir = os.path.join(project_root_dir, "visualizations")
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created visualizations directory: {self.output_dir}")
            
        # Setup figure and plot style
        plt.style.use('dark_background')
    
    def _get_closest_frequency(self, freq: float) -> float:
        """Get the closest matching frequency from our defined set"""
        frequencies = list(FREQUENCY_COLORS.keys())
        return min(frequencies, key=lambda x: abs(x - freq))
    
    def _create_phi_harmonic_color_map(self, base_color: str, coherence_value: Optional[float] = None) -> LinearSegmentedColormap:
        """Create a phi-harmonic color map based on the frequency color, modulated by coherence."""
        # Convert hex to RGB
        h = base_color.lstrip('#')
        base_rgb = tuple(int(h[i:i+2], 16)/255 for i in (0, 2, 4))

        # Determine intensity factor from coherence
        min_intensity = 0.3 # Minimum color intensity (e.g., for coherence = 0)
        max_intensity = 1.0 # Maximum color intensity (e.g., for coherence = 1 or None)
        
        if coherence_value is not None:
            # Ensure coherence_value is within [0, 1]
            clamped_coherence = max(0.0, min(coherence_value, 1.0))
            intensity_factor = min_intensity + clamped_coherence * (max_intensity - min_intensity)
        else:
            intensity_factor = max_intensity # Default to full intensity if coherence is not provided

        # Create phi-harmonic color transitions
        colors = []
        for i in range(5):
            # Adjust color based on phi harmonics and then apply intensity factor
            harmonic_adjusted_rgb = tuple(min(1.0, c * (1 + (self.phi - 1) * (i/4))) for c in base_rgb)
            final_rgb = tuple(min(1.0, c_adj * intensity_factor) for c_adj in harmonic_adjusted_rgb)
            colors.append(final_rgb)
            
        # Create custom colormap
        return LinearSegmentedColormap.from_list("phi_harmonic", colors)
    
    def _generate_hexagonal_pattern(self, size: int = 100) -> np.ndarray:
        """Generate hexagonal pattern for Ground State (432 Hz)"""
        x = np.linspace(-5, 5, size)
        y = np.linspace(-5, 5, size)
        X, Y = np.meshgrid(x, y)
        
        Z = np.zeros((size, size))
        for i in range(1, 4):
            Z += np.cos(X * i) * np.cos(Y * i * self.phi)
            Z += np.cos(X * i * self.phi) * np.cos((X + Y) * i)
            
        return Z
    
    def _generate_flower_of_life(self, size: int = 100) -> np.ndarray:
        """Generate flower of life pattern for Creation Point (528 Hz)"""
        x = np.linspace(-5, 5, size)
        y = np.linspace(-5, 5, size)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        
        Z = np.zeros((size, size))
        for i in range(1, 7):
            phi_angle = i * np.pi / 3
            cx = 1 * np.cos(phi_angle)
            cy = 1 * np.sin(phi_angle)
            
            # Distance from center of circle
            R_circle = np.sqrt((X - cx)**2 + (Y - cy)**2)
            Z += np.exp(-(R_circle**2) / 0.3)
            
        # Center circle
        Z += np.exp(-(R**2) / 0.3)
        
        return Z
    
    def _generate_heart_field(self, size: int = 100) -> np.ndarray:
        """Generate heart field pattern for Heart Field (594 Hz)"""
        x = np.linspace(-2, 2, size)
        y = np.linspace(-2, 2, size)
        X, Y = np.meshgrid(x, y)
        
        # Heart curve equation
        Z = (X**2 + Y**2 - 1)**3 - X**2 * Y**3
        Z = 1 / (1 + np.exp(-Z * 2))  # Sigmoid to normalize
        
        return Z
    
    def _generate_mandala(self, size: int = 100) -> np.ndarray:
        """Generate mandala pattern for Voice Flow (672 Hz)"""
        x = np.linspace(-5, 5, size)
        y = np.linspace(-5, 5, size)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        Theta = np.arctan2(Y, X)
        
        Z = np.zeros((size, size))
        for i in range(1, 9):
            Z += np.cos(i * Theta) * np.exp(-(R - i/2)**2 / 0.2)
            Z += np.sin(i * self.phi * Theta) * np.exp(-(R - i/self.phi)**2 / 0.3)
            
        return Z
    
    def _generate_geometric_network(self, size: int = 100) -> np.ndarray:
        """Generate geometric network for Vision Gate (720 Hz)"""
        x = np.linspace(-5, 5, size)
        y = np.linspace(-5, 5, size)
        X, Y = np.meshgrid(x, y)
        
        Z = np.zeros((size, size))
        for i in range(1, 6):
            phi_factor = self.phi ** i
            Z += np.cos(X * phi_factor) * np.cos(Y * phi_factor)
            Z += np.cos((X + Y) * phi_factor) * np.cos((X - Y) * phi_factor)
            
        return Z
    
    def _generate_toroidal(self, size: int = 100) -> np.ndarray:
        """Generate toroidal pattern for Unity Wave (768 Hz)"""
        x = np.linspace(-5, 5, size)
        y = np.linspace(-5, 5, size)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        
        # Toroidal function (cross-section)
        r_torus = 2.0  # Major radius
        a_torus = 0.8  # Minor radius
        Z = np.exp(-((R - r_torus)**2 + Y**2) / a_torus**2)
        
        # Add phi-harmonic wave
        for i in range(1, 4):
            Z += 0.2 * np.cos(i * self.phi * R) * np.exp(-(R - r_torus)**2 / (2 * a_torus**2))
            
        return Z
    
    def _generate_phi_harmonic_infinity(self, size: int = 100) -> np.ndarray:
        """Generate phi harmonic infinity pattern for Source Field (963 Hz)"""
        x = np.linspace(-5, 5, size)
        y = np.linspace(-5, 5, size)
        X, Y = np.meshgrid(x, y)
        
        Z = np.zeros((size, size))
        for i in range(1, 7):
            phi_power = self.phi ** i
            Z += np.sin(X * phi_power) * np.sin(Y * phi_power) / phi_power
            Z += np.cos(X * phi_power + Y * phi_power) / phi_power
            Z += np.sin(X * Y * phi_power) / phi_power
            
        return Z
    
    def visualize_pattern(self, 
                          frequency: float, 
                          title_override: Optional[str] = None, 
                          save: bool = True,
                          transition_id: Optional[str] = None,
                          object_name: Optional[str] = None,
                          compression: Optional[float] = None,
                          coherence: Optional[float] = None):
        """Visualize the cymatics pattern for a specific frequency.
        
        Args:
            frequency (float): The frequency to visualize.
            title_override (Optional[str]): If provided, used as the base for the output filename 
                                            (e.g., 'title_override.png') and for the plot title.
            save (bool): Whether to save the visualization to a file.
            transition_id (Optional[str]): The ID of the transition (e.g., 'T1').
            object_name (Optional[str]): The name of the quantum object.
            compression (Optional[float]): The compression factor.
            coherence (Optional[float]): The coherence value (0.0 to 1.0).
        """
        closest_freq = self._get_closest_frequency(frequency)
        pattern_data = self._get_pattern_for_frequency(closest_freq)
        base_color = FREQUENCY_COLORS.get(closest_freq, "#FFFFFF") # Default to white if not found
        # Pass coherence to the colormap creation
        cmap = self._create_phi_harmonic_color_map(base_color, coherence_value=coherence)

        # Alpha calculation based on compression
        min_alpha = 0.3
        max_alpha = 1.0
        # Assuming PHI as a reasonable upper bound for compression for scaling purposes
        # If compression is None, pattern is fully opaque or a high default alpha
        default_alpha = 0.95 

        if compression is not None:
            # Normalize compression: assume 0 to PHI maps to min_alpha to max_alpha
            # Clamp compression to avoid negative alpha or alpha > max_alpha if compression is outside [0, PHI]
            clamped_compression = max(0, min(compression, PHI))
            alpha_value = min_alpha + (clamped_compression / PHI) * (max_alpha - min_alpha)
        else:
            alpha_value = default_alpha
        alpha_value = max(min_alpha, min(alpha_value, max_alpha)) # Ensure it's strictly within bounds

        fig_width_inches = 8
        fig_height_inches = 8
        dpi_val = 100 # DPI for figure display

        fig = plt.figure(figsize=(fig_width_inches, fig_height_inches), dpi=dpi_val)
        
        plot_title_text = ""
        is_3d = pattern_data.ndim == 3

        if title_override:
            # Clean up title_override for display: replace underscores, title case
            cleaned_display_title = title_override.replace('_', ' ').title()
            plot_title_text = f"{cleaned_display_title} ({closest_freq} Hz)"
            if is_3d:
                 plot_title_text = f"{cleaned_display_title} (3D) ({closest_freq} Hz)"
        else:
            default_pattern_name = FREQUENCY_PATTERNS.get(closest_freq, "Unknown Pattern").replace('_', ' ').title()
            plot_title_text = f"{default_pattern_name} ({closest_freq} Hz)"
            if is_3d:
                plot_title_text = f"{default_pattern_name} (3D) ({closest_freq} Hz)"

        if not is_3d: # 2D pattern
            ax = fig.add_subplot(111)
            ax.imshow(pattern_data, cmap=cmap, interpolation='bilinear', alpha=alpha_value)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)
            ax.set_title(plot_title_text, fontsize=16, color='white', pad=20)
        elif is_3d: # 3D pattern (X, Y, Z coordinates for scatter)
            ax = fig.add_subplot(111, projection='3d')
            X_scatter, Y_scatter, Z_scatter_vals = pattern_data # Expecting tuple of 3 arrays
            # Ensure base_color is in RGB format for scatter plot
            h_scatter = base_color.lstrip('#')
            scatter_base_rgb = tuple(int(h_scatter[i:i+2], 16)/255 for i in (0, 2, 4))
            ax.scatter(X_scatter, Y_scatter, Z_scatter_vals, c=[scatter_base_rgb] * len(X_scatter.flatten()), s=10, alpha=alpha_value) # Using a single color for simplicity, cmap could be used with Z_scatter_vals if desired.
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.grid(False)
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor('black')
            ax.yaxis.pane.set_edgecolor('black')
            ax.zaxis.pane.set_edgecolor('black')
            ax.set_title(plot_title_text, fontsize=16, color='white', pad=20)
        else:
            plt.close(fig)
            print(f"Warning: Pattern data for {frequency}Hz has unexpected dimensions: {pattern_data.ndim}. Expected 2 or 3.")
            return None # Or fig, but fig is closed. Let's return None here.

        # title_prefix = f"{transition_id}: " if transition_id else "" # This line is effectively overridden by the next plt.title call
        # title_obj_name = object_name if object_name else "Pattern"
        # This plt.title call seems to be setting a general title for the whole plot area,
        # potentially conflicting with ax.set_title. Considering its position and fontsize, it might be for overall context.
        # Let's ensure it uses the contextual object name and transition_id if provided for the main visualization.
        main_plot_title = f"{transition_id}: " if transition_id else ""
        main_plot_title += object_name if object_name else FREQUENCY_PATTERNS.get(closest_freq, "Pattern").replace('_',' ').title()
        main_plot_title += f" at {closest_freq:.0f} Hz"
        plt.title(main_plot_title, fontsize=12, pad=10, color='lightgrey')

        # Overlay additional information
        info_text = []
        if transition_id:
            info_text.append(f"ID: {transition_id}")
        if object_name:
            info_text.append(f"Obj: {object_name}")
        info_text.append(f"Freq: {frequency:.2f} Hz (Closest: {closest_freq} Hz)")
        if compression is not None:
            info_text.append(f"Comp: {compression:.4f}")
        if coherence is not None:
            info_text.append(f"Coh: {coherence:.3f}")
        
        # Add text to the bottom left of the plot
        # fig.text is relative to figure, ax.text is relative to axes
        # Using ax.text with transform=ax.transAxes for positioning relative to axes boundaries
        ax = plt.gca()
        for i, line in enumerate(info_text):
            ax.text(0.02, 0.02 + (i * 0.04), line, 
                    transform=ax.transAxes, 
                    fontsize=8, 
                    color='white', 
                    bbox=dict(facecolor='black', alpha=0.5, pad=1))

        plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # Adjust layout to make space for info text

        if save:
            if title_override:
                filename = f"{title_override}.png"
            else:
                pattern_name_for_file = FREQUENCY_PATTERNS.get(closest_freq, "Unknown_Pattern")
                filename = f"{pattern_name_for_file}_{closest_freq:.0f}Hz"
            
            output_path = os.path.join(self.output_dir, filename)
            plt.savefig(output_path, dpi=dpi_val, bbox_inches='tight', pad_inches=0.1, facecolor='black')
            print(f"Saved visualization to: {output_path}")
            plt.close(fig)
            return None # Explicitly return None when saving
        else: # if not save
            return fig # Return the figure object for animation frames

    def animate_transition(self, 
                         from_params: Dict, 
                         to_params: Dict, 
                         animation_filename_base: str, 
                         duration_sec: int = 2, 
                         fps: int = 15):
        """Animate the transition between two quantum states by interpolating parameters."""
        num_frames = duration_sec * fps
        frames_data = []

        print(f"Generating animation for {animation_filename_base} ({num_frames} frames)... Freq: {from_params['frequency']:.0f}->{to_params['frequency']:.0f}")

        for frame_idx in range(num_frames):
            interp_ratio = frame_idx / (num_frames - 1) if num_frames > 1 else 1.0
            
            current_freq = from_params['frequency'] + (to_params['frequency'] - from_params['frequency']) * interp_ratio
            current_comp = from_params['compression'] + (to_params['compression'] - from_params['compression']) * interp_ratio
            current_coh = from_params['coherence'] + (to_params['coherence'] - from_params['coherence']) * interp_ratio

            # Use target state's object_name and transition_id for info text consistency in animation frames
            # title_override can be None to let visualize_pattern decide based on object_name and frequency
            fig = self.visualize_pattern(
                frequency=current_freq,
                title_override=None, # Let pattern define title based on object_name and freq
                object_name=to_params['object_name'], 
                transition_id=to_params['id'], # This is the ID of the 'to' state's transition event
                compression=current_comp,
                coherence=current_coh,
                save=False
            )

            if fig is None:
                print(f"Warning: Frame {frame_idx+1}/{num_frames} for {animation_filename_base} could not be generated. Skipping frame.")
                # Optionally, append a blank or previous frame, or handle error more robustly
                # For now, skip if a frame fails (e.g., due to pattern data issue for an interpolated freq)
                continue

            fig.canvas.draw() # Draw the canvas, cache the renderer
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames_data.append(image)
            plt.close(fig) # Close the figure to free memory

        if not frames_data:
            print(f"Error: No frames generated for animation {animation_filename_base}. Animation not saved.")
            return

        output_path = os.path.join(self.output_dir, f"{animation_filename_base}.gif")
        try:
            imageio.mimsave(output_path, frames_data, fps=fps)
            print(f"Saved animation to: {output_path}")
        except Exception as e:
            print(f"Error saving animation {output_path}: {e}")

    def visualize_frequency_transition(self, start_freq: float, end_freq: float, 
                                      steps: int = 30, save: bool = True) -> None:
        """Create an animation showing the transition between frequencies"""
        # Get the closest matching frequencies
        start_freq = self._get_closest_frequency(start_freq)
        end_freq = self._get_closest_frequency(end_freq)
        
        # Get pattern types and colors
        start_pattern = FREQUENCY_PATTERNS[start_freq]
        end_pattern = FREQUENCY_PATTERNS[end_freq]
        
        start_color = FREQUENCY_COLORS[start_freq]
        end_color = FREQUENCY_COLORS[end_freq]
        
        # Create animation
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
        
        def animate(i):
            # Calculate interpolation factor
            t = i / (steps - 1)
            
            # Interpolate frequency
            current_freq = start_freq + t * (end_freq - start_freq)
            
            # Get pattern based on interpolation
            if t < 0.5:
                # First half: start pattern morphing
                Z = self._get_pattern_for_frequency(start_freq)
                Z = Z * (1 - t*2) + t*2 * np.random.normal(0, 0.1, Z.shape)
            else:
                # Second half: end pattern forming
                Z = self._get_pattern_for_frequency(end_freq)
                Z = Z * ((t-0.5)*2) + (1-(t-0.5)*2) * np.random.normal(0, 0.1, Z.shape)
                
            # Interpolate color
            h_start = start_color.lstrip('#')
            h_end = end_color.lstrip('#')
            
            r_start, g_start, b_start = tuple(int(h_start[i:i+2], 16) for i in (0, 2, 4))
            r_end, g_end, b_end = tuple(int(h_end[i:i+2], 16) for i in (0, 2, 4))
            
            r = int(r_start + t * (r_end - r_start))
            g = int(g_start + t * (g_end - g_start))
            b = int(b_start + t * (b_end - b_start))
            
            current_color = f'#{r:02x}{g:02x}{b:02x}'
            cmap = self._create_phi_harmonic_color_map(current_color)
            
            # Update plot
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)
            
            im = ax.imshow(Z, cmap=cmap, interpolation='bilinear')
            
            # Update title
            title = f"Transition: {start_freq}Hz → {end_freq}Hz ({current_freq:.1f}Hz)"
            ax.set_title(title, fontsize=16, color='white')
            
            return [im]
            
        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=steps, interval=100, blit=True)
        
        # Save if requested
        if save:
            filename = f"transition_{int(start_freq)}hz_to_{int(end_freq)}hz.mp4"
            filepath = os.path.join(self.output_dir, filename)
            anim.save(filepath, dpi=200, writer='ffmpeg')
            print(f"Saved transition animation to: {filepath}")
        
        plt.close(fig)
    
    def _get_pattern_for_frequency(self, frequency: float) -> np.ndarray:
        """Get the pattern for a specific frequency"""
        pattern_type = FREQUENCY_PATTERNS[frequency]
        
        if pattern_type == "hexagonal":
            return self._generate_hexagonal_pattern()
        elif pattern_type == "flower_of_life":
            return self._generate_flower_of_life()
        elif pattern_type == "heart_field":
            return self._generate_heart_field()
        elif pattern_type == "mandala":
            return self._generate_mandala()
        elif pattern_type == "geometric_network":
            return self._generate_geometric_network()
        elif pattern_type == "toroidal":
            return self._generate_toroidal()
        else:  # phi_harmonic_infinity
            return self._generate_phi_harmonic_infinity()
    
    def create_full_frequency_spectrum(self) -> None:
        """Create visualizations for the full frequency spectrum"""
        print("Generating full frequency spectrum visualizations...")
        
        frequencies = sorted(list(FREQUENCY_PATTERNS.keys()))
        for freq in frequencies:
            print(f"Visualizing {freq} Hz pattern...")
            self.visualize_pattern(freq, save=True)
            
        print("Full frequency spectrum visualization complete!")
    
    def create_quantum_transition_sequence(self) -> None:
        """Create a series of transitions across all frequencies"""
        print("Generating quantum transition sequence...")
        
        frequencies = sorted(list(FREQUENCY_PATTERNS.keys()))
        for i in range(len(frequencies)-1):
            print(f"Creating transition from {frequencies[i]} Hz to {frequencies[i+1]} Hz...")
            self.visualize_frequency_transition(frequencies[i], frequencies[i+1], steps=20, save=True)
            
        print("Quantum transition sequence complete!")


# Direct execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Cymatics Visualization System")
    parser.add_argument("--frequency", type=float, help="Frequency to visualize")
    parser.add_argument("--transition", nargs=2, type=float, metavar=('START', 'END'),
                       help="Create transition between start and end frequencies")
    parser.add_argument("--all", action="store_true", help="Generate all frequency visualizations")
    parser.add_argument("--sequence", action="store_true", help="Generate complete transition sequence")
    
    args = parser.parse_args()
    vis = CymaticsVisualizer()
    
    if args.frequency:
        vis.visualize_pattern(args.frequency)
    elif args.transition:
        vis.visualize_frequency_transition(args.transition[0], args.transition[1])
    elif args.all:
        vis.create_full_frequency_spectrum()
    elif args.sequence:
        vis.create_quantum_transition_sequence()
    else:
        # Default: create full spectrum
        vis.create_full_frequency_spectrum()
