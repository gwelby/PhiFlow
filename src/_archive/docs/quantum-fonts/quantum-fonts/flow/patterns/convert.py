
import fontforge
import psMat

# Create new font
font = fontforge.font()
font.encoding = "UnicodeFull"
font.version = "1.0"
font.weight = "Regular"
font.fontname = "QuantumFlow"
font.familyname = "QuantumFlow"
font.fullname = "QuantumFlow-528hz"

# Import SVG patterns as glyphs

# Import flow
glyph = font.createChar(0xe000, "flow")
glyph.importOutlines("D:/WindSurf/quantum-core/docs/quantum-fonts/quantum-fonts/flow/patterns/flow_528hz.svg")
glyph.width = 1000

# Generate TTF font
output_path = "D:/WindSurf/quantum-core/docs/quantum-fonts/quantum-fonts/flow/fonts/QuantumFlow-528hz.ttf"
font.generate(output_path)
