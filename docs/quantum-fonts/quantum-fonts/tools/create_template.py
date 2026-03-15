from fontTools.ttLib import TTFont, newTable
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib.tables._c_m_a_p import CmapSubtable
from fontTools.ttLib.tables.O_S_2f_2 import Panose

def create_template_font():
    # Initialize font
    font = TTFont()
    
    # Add required tables
    font['head'] = newTable('head')
    font['hhea'] = newTable('hhea')
    font['maxp'] = newTable('maxp')
    font['OS/2'] = newTable('OS/2')
    font['cmap'] = newTable('cmap')
    font['name'] = newTable('name')
    font['post'] = newTable('post')
    font['glyf'] = newTable('glyf')
    font['loca'] = newTable('loca')
    font['hmtx'] = newTable('hmtx')
    
    # Initialize glyf and hmtx tables
    font['glyf'].glyphs = {}
    font['hmtx'].metrics = {}
    
    # Create minimal glyph set
    glyphs = ['.notdef', 'space']
    font.setGlyphOrder(glyphs)
    
    # Add empty glyphs
    for glyph_name in glyphs:
        pen = TTGlyphPen(font.getGlyphSet())
        # Create an empty contour for each glyph
        if glyph_name == '.notdef':
            # Create a simple square for .notdef glyph
            pen.moveTo((0, 0))
            pen.lineTo((0, 1000))
            pen.lineTo((1000, 1000))
            pen.lineTo((1000, 0))
            pen.closePath()
        glyph = pen.glyph()
        font['glyf'][glyph_name] = glyph
        font['hmtx'].metrics[glyph_name] = (500, 50)
    
    # Set basic font info
    head = font['head']
    head.tableVersion = 1.0
    head.fontRevision = 1.0
    head.magicNumber = 0x5F0F3CF5
    head.flags = 0
    head.unitsPerEm = 1000
    head.created = 0
    head.modified = 0
    head.xMin = 0
    head.yMin = 0
    head.xMax = 1000
    head.yMax = 1000
    head.macStyle = 0
    head.lowestRecPPEM = 8
    head.fontDirectionHint = 2
    head.indexToLocFormat = 0
    head.glyphDataFormat = 0
    head.checkSumAdjustment = 0
    
    # Set horizontal metrics
    hhea = font['hhea']
    hhea.tableVersion = 0x00010000
    hhea.ascent = 800
    hhea.descent = -200
    hhea.lineGap = 200
    hhea.advanceWidthMax = 1000
    hhea.minLeftSideBearing = 0
    hhea.minRightSideBearing = 0
    hhea.xMaxExtent = 1000
    hhea.caretSlopeRise = 1
    hhea.caretSlopeRun = 0
    hhea.caretOffset = 0
    hhea.reserved0 = 0
    hhea.reserved1 = 0
    hhea.reserved2 = 0
    hhea.reserved3 = 0
    hhea.metricDataFormat = 0
    hhea.numberOfHMetrics = len(glyphs)
    
    # Set maxp table version
    maxp = font['maxp']
    maxp.tableVersion = 0x00010000
    maxp.numGlyphs = len(glyphs)
    maxp.maxPoints = 4  # For .notdef square
    maxp.maxContours = 1  # For .notdef square
    maxp.maxCompositePoints = 0
    maxp.maxCompositeContours = 0
    maxp.maxZones = 2
    maxp.maxTwilightPoints = 0
    maxp.maxStorage = 0
    maxp.maxFunctionDefs = 0
    maxp.maxInstructionDefs = 0
    maxp.maxStackElements = 0
    maxp.maxSizeOfInstructions = 0
    maxp.maxComponentElements = 0
    maxp.maxComponentDepth = 0
    
    # Set post table version
    post = font['post']
    post.formatType = 3.0
    post.italicAngle = 0
    post.underlinePosition = -100
    post.underlineThickness = 50
    post.isFixedPitch = 0
    post.minMemType42 = 0
    post.maxMemType42 = 0
    post.minMemType1 = 0
    post.maxMemType1 = 0
    
    # Set OS/2 table version
    os2 = font['OS/2']
    os2.version = 0x0004
    os2.xAvgCharWidth = 500
    os2.usWeightClass = 400
    os2.usWidthClass = 5
    os2.fsType = 0
    os2.ySubscriptXSize = 650
    os2.ySubscriptYSize = 699
    os2.ySubscriptXOffset = 0
    os2.ySubscriptYOffset = 140
    os2.ySuperscriptXSize = 650
    os2.ySuperscriptYSize = 699
    os2.ySuperscriptXOffset = 0
    os2.ySuperscriptYOffset = 479
    os2.yStrikeoutSize = 49
    os2.yStrikeoutPosition = 258
    os2.sFamilyClass = 0
    panose = Panose()
    panose.bFamilyType = 2
    panose.bSerifStyle = 0
    panose.bWeight = 5
    panose.bProportion = 3
    panose.bContrast = 0
    panose.bStrokeVariation = 0
    panose.bArmStyle = 0
    panose.bLetterForm = 0
    panose.bMidline = 0
    panose.bXHeight = 0
    os2.panose = panose
    os2.ulUnicodeRange1 = 0x00000001
    os2.ulUnicodeRange2 = 0x00000000
    os2.ulUnicodeRange3 = 0x00000000
    os2.ulUnicodeRange4 = 0x00000000
    os2.achVendID = 'NONE'
    os2.fsSelection = 0x0040
    os2.usFirstCharIndex = 0x0020
    os2.usLastCharIndex = 0x007E
    os2.sTypoAscender = 800
    os2.sTypoDescender = -200
    os2.sTypoLineGap = 200
    os2.usWinAscent = 1000
    os2.usWinDescent = 200
    os2.ulCodePageRange1 = 0x00000001
    os2.ulCodePageRange2 = 0x00000000
    os2.sxHeight = 500
    os2.sCapHeight = 700
    os2.usDefaultChar = 0x0020
    os2.usBreakChar = 0x0020
    os2.usMaxContext = 0
    
    # Set name table
    nameTable = font['name']
    nameTable.names = []
    nameTable.addName("Template Font")
    
    # Set cmap table
    cmap = font['cmap']
    cmap.tableVersion = 0
    cmap.tables = []
    
    # Create format 4 subtable
    format4 = CmapSubtable.newSubtable(4)
    format4.platformID = 3
    format4.platEncID = 1
    format4.language = 0
    format4.cmap = {0x20: 'space'}  # Map space character
    cmap.tables.append(format4)
    
    # Save font
    font.save("../templates/template.ttf")

if __name__ == "__main__":
    create_template_font()
