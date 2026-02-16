
import fontforge
import psMat

# Create new font
font = fontforge.font()
font.encoding = "UnicodeFull"
font.version = "1.0"
font.weight = "Regular"
font.fontname = "QuantumCrystal"
font.familyname = "QuantumCrystal"
font.fullname = "QuantumCrystal-768hz"

# Import SVG patterns as glyphs

# Import crystal
glyph = font.createChar(0xe000, "crystal")
glyph.importOutlines("D:/WindSurf/quantum-core/docs/quantum-fonts/quantum-fonts/crystal/patterns/crystal_768hz.svg")
glyph.width = 1000

# Generate TTF font
output_path = "D:/WindSurf/quantum-core/docs/quantum-fonts/quantum-fonts/crystal/fonts/QuantumCrystal-768hz.ttf"
font.generate(output_path)
