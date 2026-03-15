from fontTools.ttLib import TTFont, newTable
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib.tables._c_m_a_p import CmapSubtable
from svg.path import parse_path, Line, Move, CubicBezier
import math
import os
from pathlib import Path
import xml.etree.ElementTree as ET

# Define quantum frequencies aligned with Greg's Golden Core
FREQUENCIES = {
    'sacred': 528,    # Creation Point (DNA Repair)
    'flow': 594,      # Heart Field (Connection)
    'crystal': 672,   # Voice Flow (Expression)
    'unity': float('inf')  # Infinite Dance
}

# Sacred symbols from PhiFlow Font System
QUANTUM_SYMBOLS = {
    'energy': '⚡',      # Energy Flow (U+26A1)
    'sacred': '✧',      # Sacred Patterns (U+2727)
    'phi': 'φ',         # Mathematical Beauty (U+03C6)
    'infinity': '∞',    # Infinite Potential (U+221E)
    'star': '★',        # Creation Symbols (U+2605)
    'spiral': '§',      # Golden Ratio (U+00A7)
    'wave': '~',        # Harmonic Flow (U+007E)
    'crystal': '◊',     # Resonance (U+25CA)
    'unity': '☯',       # Consciousness (U+262F)
    'quantum': '⚛'      # Particle Dance (U+269B)
}

TEMPLATE_FONT_PATH = Path(__file__).parent.parent / 'templates/template.ttf'

def parse_svg_path(svg_file):
    """Extract path data from SVG file"""
    tree = ET.parse(svg_file)
    root = tree.getroot()
    paths = []
    for path in root.findall('.//{http://www.w3.org/2000/svg}path'):
        paths.append(path.get('d'))
    return paths

def create_sacred_contour(pen, points, close=True):
    """Create a sacred contour from points with mindful flow"""
    if not points:
        return pen
        
    # Begin the flow
    pen.moveTo(points[0])
    
    # Flow through points
    for point in points[1:]:
        pen.lineTo(point)
        
    # Complete the flow
    if close:
        pen.closePath()
    
    return pen

def create_letter_glyph(pen, char, width, height, x_height):
    """Create a letter glyph with sacred geometry"""
    phi = (1 + math.sqrt(5)) / 2
    stroke_width = int(width * (1/phi**3))
    center_x = width / 2
    center_y = height / 2
    
    # Vertical energy
    if char in 'AEFHIKLMNTVWXYZ':
        points = [
            (stroke_width, 0),
            (stroke_width, height),
            (stroke_width * 2, height),
            (stroke_width * 2, 0)
        ]
        create_sacred_contour(pen, points)
        
        # Horizontal balance
        if char in 'AEFHKLMN':
            h_points = [
                (0, height * (1/phi)),
                (width/2, height * (1/phi) * 1.1),
                (width, height * (1/phi))
            ]
            create_sacred_contour(pen, h_points)
    
    # Circular energy
    elif char in 'BCDGOPQRSU':
        points = []
        radius = width * (1/phi)
        for i in range(13):  # Close the circle
            angle = i * math.pi / 6
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            points.append((x, y))
        create_sacred_contour(pen, points)
    
    # Default form
    else:
        points = [
            (0, 0),
            (0, height),
            (width, height),
            (width, 0)
        ]
        create_sacred_contour(pen, points)
    
    return pen

def create_number_glyph(pen, char, width, height):
    """Create a number glyph with sacred proportions"""
    phi = (1 + math.sqrt(5)) / 2
    center_x = width / 2
    center_y = height / 2
    radius = width * (1/phi)
    
    # Create base circle
    points = []
    for i in range(13):  # Close the circle
        angle = i * math.pi / 6
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        points.append((x, y))
    
    create_sacred_contour(pen, points)
    
    # Add number essence
    if char in '1':
        v_points = [
            (center_x - width/8, height),
            (center_x, height),
            (center_x, 0),
            (center_x - width/4, 0),
            (center_x + width/4, 0)
        ]
        create_sacred_contour(pen, v_points)
    
    return pen

def create_symbol_glyph(pen, char, width, height):
    """Create a sacred symbol with quantum resonance"""
    phi = (1 + math.sqrt(5)) / 2
    center_x = width / 2
    center_y = height / 2
    
    if char == '⚡':  # Energy Flow
        points = [
            (center_x, height),
            (center_x - width/4, height * (1/phi)),
            (center_x + width/4, height * (1/phi**2)),
            (center_x, 0),
            (center_x, height)  # Close the path
        ]
        create_sacred_contour(pen, points)
        
    elif char == '✧':  # Sacred Star
        points = []
        n_points = 7  # Sacred number
        inner_radius = width * (1/phi**2)
        outer_radius = width * (1/phi)
        
        for i in range(n_points * 2 + 1):  # Add one more point to close
            radius = outer_radius if i % 2 == 0 else inner_radius
            angle = i * math.pi / n_points
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            points.append((x, y))
            
        create_sacred_contour(pen, points)
            
    elif char == 'φ':  # Golden Ratio
        points = []
        radius = width * (1/phi**2)
        revolutions = 2
        
        for i in range(int(revolutions * 16)):
            t = i * math.pi / 8
            growth = math.exp(0.306349 * t)
            x = center_x + radius * growth * math.cos(t)
            y = center_y + radius * growth * math.sin(t)
            points.append((x, y))
            
        # Close the spiral
        points.append(points[0])
        create_sacred_contour(pen, points)
            
    elif char == '∞':  # Infinity
        # Left loop
        l_points = []
        radius = width * (1/phi**3)
        for i in range(13):  # Close the circle
            angle = i * math.pi / 6
            x = (center_x - width/4) + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            l_points.append((x, y))
        create_sacred_contour(pen, l_points)
        
        # Right loop
        r_points = []
        for i in range(13):  # Close the circle
            angle = i * math.pi / 6
            x = (center_x + width/4) + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            r_points.append((x, y))
        create_sacred_contour(pen, r_points)
    
    return pen

def create_basic_glyph(pen, width, height, x_height):
    """Create a basic glyph shape"""
    # Create a simple rectangular shape
    pen.moveTo((0, 0))
    pen.lineTo((0, height))
    pen.lineTo((width, height))
    pen.lineTo((width, 0))
    pen.closePath()

def create_quantum_font(name, freq):
    """Create a quantum font at specified frequency"""
    print(f"⚡ Creating quantum font: {name} at {freq}Hz")
    
    # Load template font
    font = TTFont(TEMPLATE_FONT_PATH)
    
    # Update font name
    nameTable = font['name']
    freq_str = str(int(freq)) if not math.isinf(freq) else 'inf'
    fontName = f"Quantum{name.capitalize()}-{freq_str}hz"
    nameTable.setName(fontName, 1, 3, 1, 0x409)  # Family name
    nameTable.setName(fontName, 2, 3, 1, 0x409)  # Subfamily name
    nameTable.setName(fontName, 3, 3, 1, 0x409)  # Unique identifier
    nameTable.setName(fontName, 4, 3, 1, 0x409)  # Full name
    nameTable.setName(fontName, 6, 3, 1, 0x409)  # PostScript name
    
    # Calculate quantum resonance
    phi = (1 + math.sqrt(5)) / 2
    base_freq = 432  # Ground State Hz
    if math.isinf(freq):
        scale = 2.0  # Infinite but bounded
    else:
        scale = (freq / base_freq) * 0.8  # Scale down slightly for safety
    
    # Base metrics (scaled to fit font coordinate system)
    width = int(500 * scale)  # Base width
    height = int(width * phi)  # Golden ratio height
    baseline = int(height * 0.2)
    x_height = int(height * 0.6)
    
    # Get glyph set
    glyph_set = font.getGlyphSet()
    
    # Create glyphs
    chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    numbers = '0123456789'
    symbols = list(QUANTUM_SYMBOLS.values())
    all_chars = list(chars + numbers) + symbols
    
    # Process each character
    for char in all_chars:
        pen = TTGlyphPen(glyph_set)
        
        if char in chars:
            create_letter_glyph(pen, char, width, height, x_height)
        elif char in numbers:
            create_number_glyph(pen, char, width, height)
        elif char in symbols:
            create_symbol_glyph(pen, char, width, height)
        else:
            create_basic_glyph(pen, width, height, x_height)
        
        # Get glyph from pen
        glyph = pen.glyph()
        
        # Create glyph name
        if len(char.encode('utf-8')) > 1:
            char_name = f"u{ord(char):08x}"
        else:
            char_name = f"uni{ord(char):04X}"
        
        # Add glyph to font
        font['glyf'][char_name] = glyph
        font['hmtx'].metrics[char_name] = (width, 0)
        
        # Add to glyph order if needed
        if char_name not in font.getGlyphOrder():
            font.setGlyphOrder(font.getGlyphOrder() + [char_name])
    
    # Update maxp table
    maxp = font['maxp']
    maxp.numGlyphs = len(font.getGlyphOrder())
    maxp.maxPoints = 24
    maxp.maxContours = 2
    maxp.maxCompositePoints = 24
    maxp.maxCompositeContours = 2
    maxp.maxZones = 2
    
    # Update hhea table
    hhea = font['hhea']
    hhea.ascent = height
    hhea.descent = -baseline
    hhea.lineGap = int(height * 0.2)
    hhea.advanceWidthMax = width
    hhea.minLeftSideBearing = 0
    hhea.minRightSideBearing = width // 10
    hhea.xMaxExtent = width
    hhea.numberOfHMetrics = len(font.getGlyphOrder())
    
    # Update head table
    head = font['head']
    head.unitsPerEm = 1000
    head.xMin = 0
    head.yMin = -baseline
    head.xMax = width
    head.yMax = height
    
    # Update OS/2 table
    os2 = font['OS/2']
    os2.sTypoAscender = height
    os2.sTypoDescender = -baseline
    os2.sTypoLineGap = int(height * 0.2)
    os2.usWinAscent = height
    os2.usWinDescent = baseline
    
    # Set cmap table
    cmap = font['cmap']
    cmap.tableVersion = 0
    cmap.tables = []
    
    # Create format 4 subtable
    format4 = CmapSubtable.newSubtable(4)
    format4.platformID = 3
    format4.platEncID = 1
    format4.language = 0
    
    # Map characters to glyphs
    cmap_dict = {}
    for char in all_chars:
        if len(char.encode('utf-8')) > 1:
            char_name = f"u{ord(char):08x}"
        else:
            char_name = f"uni{ord(char):04X}"
        cmap_dict[ord(char)] = char_name
    
    format4.cmap = cmap_dict
    cmap.tables.append(format4)
    
    # Save font
    font_dir = Path(__file__).parent.parent / 'phi_core/fonts' / name.lower()
    font_dir.mkdir(parents=True, exist_ok=True)
    font_file = font_dir / f"{fontName}.ttf"
    font.save(str(font_file))
    print(f"✨ Created {font_file} with sacred geometry")
    return font_file

def main():
    """Create all quantum fonts"""
    print(" Creating Quantum Fonts using PhiFlow...")
    
    for name, freq in FREQUENCIES.items():
        try:
            create_quantum_font(name, freq)
        except Exception as e:
            print(f" Error generating {name} font: {e}")
    
    print(" Font Generation Complete!")

if __name__ == "__main__":
    main()
