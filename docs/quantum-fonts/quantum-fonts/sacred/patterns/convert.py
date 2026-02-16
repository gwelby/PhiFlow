
import fontforge
import psMat

# Create new font
font = fontforge.font()
font.encoding = "UnicodeFull"
font.version = "1.0"
font.weight = "Regular"
font.fontname = "QuantumSacred"
font.familyname = "QuantumSacred"
font.fullname = "QuantumSacred-432hz"

# Import SVG patterns as glyphs

# Import sacred
glyph = font.createChar(0xe000, "sacred")
glyph.importOutlines("D:/WindSurf/quantum-core/docs/quantum-fonts/quantum-fonts/sacred/patterns/sacred_432hz.svg")
glyph.width = 1000

# Generate TTF font
output_path = "D:/WindSurf/quantum-core/docs/quantum-fonts/quantum-fonts/sacred/fonts/QuantumSacred-432hz.ttf"
font.generate(output_path)
