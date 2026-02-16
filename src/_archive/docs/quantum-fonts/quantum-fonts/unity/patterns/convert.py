
import fontforge
import psMat

# Create new font
font = fontforge.font()
font.encoding = "UnicodeFull"
font.version = "1.0"
font.weight = "Regular"
font.fontname = "QuantumUnity"
font.familyname = "QuantumUnity"
font.fullname = "QuantumUnity-infhz"

# Import SVG patterns as glyphs

# Import unity
glyph = font.createChar(0xe000, "unity")
glyph.importOutlines("D:/WindSurf/quantum-core/docs/quantum-fonts/quantum-fonts/unity/patterns/unity_infhz.svg")
glyph.width = 1000

# Generate TTF font
output_path = "D:/WindSurf/quantum-core/docs/quantum-fonts/quantum-fonts/unity/fonts/QuantumUnity-infhz.ttf"
font.generate(output_path)
