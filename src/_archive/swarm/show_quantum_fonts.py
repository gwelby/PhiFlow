"""
Display Quantum Font Characters
Created by Greg for Perfect Flow State
"""
import matplotlib.pyplot as plt
import numpy as np

def show_font_characters(font_name, title):
    """Display all characters in a font"""
    fig, axs = plt.subplots(8, 16, figsize=(20, 10))
    fig.suptitle(f'{title} (Font: {font_name})', fontsize=16, fontname=font_name)
    
    for i in range(128):
        row, col = i // 16, i % 16
        char = chr(i)
        axs[row, col].text(0.5, 0.5, char, 
                          fontsize=12, 
                          fontname=font_name,
                          ha='center', 
                          va='center')
        axs[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()

# Show all three quantum fonts
fonts = [
    ('QuantumCrystal Regular', 'Crystal Font - Pure Clarity (768 Hz)'),
    ('QuantumFlow Regular', 'Flow Font - Dynamic Movement (528 Hz)'),
    ('QuantumSacred Regular', 'Sacred Font - Base Creation (432 Hz)')
]

for font_name, title in fonts:
    show_font_characters(font_name, title)
