#!/bin/bash
# Convert CogniSense Report from Markdown to PDF

echo "CogniSense Report PDF Conversion"
echo "================================="
echo ""

# Check if pandoc is installed
if ! command -v pandoc &> /dev/null; then
    echo "❌ Pandoc is not installed."
    echo ""
    echo "Please install pandoc:"
    echo "  Ubuntu/Debian: sudo apt-get install pandoc texlive-latex-base"
    echo "  Mac: brew install pandoc basictex"
    echo "  Windows: Download from pandoc.org"
    echo ""
    echo "Alternatively, use online converter: https://www.markdowntopdf.com/"
    exit 1
fi

echo "✓ Pandoc found"
echo ""

# Convert to PDF
echo "Converting CogniSense_Report.md to PDF..."

pandoc CogniSense_Report.md -o CogniSense_Report.pdf \
    --pdf-engine=pdflatex \
    -V geometry:margin=1in \
    -V fontsize=11pt \
    -V documentclass=article \
    --highlight-style=tango \
    2>&1

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ PDF generated successfully: CogniSense_Report.pdf"
    echo ""

    # Check file size
    if [ -f "CogniSense_Report.pdf" ]; then
        SIZE=$(ls -lh CogniSense_Report.pdf | awk '{print $5}')
        echo "File size: $SIZE"

        # Count pages (if pdfinfo available)
        if command -v pdfinfo &> /dev/null; then
            PAGES=$(pdfinfo CogniSense_Report.pdf 2>/dev/null | grep Pages | awk '{print $2}')
            if [ ! -z "$PAGES" ]; then
                echo "Pages: $PAGES"

                if [ $PAGES -ge 2 ] && [ $PAGES -le 3 ]; then
                    echo ""
                    echo "✅ Page count is within hackathon requirements (2-3 pages)"
                elif [ $PAGES -lt 2 ]; then
                    echo ""
                    echo "⚠️  Warning: Page count is less than 2 pages"
                else
                    echo ""
                    echo "⚠️  Warning: Page count exceeds 3 pages. Consider reducing content."
                fi
            fi
        fi
    fi
else
    echo ""
    echo "❌ PDF conversion failed"
    echo ""
    echo "This might be due to missing LaTeX packages."
    echo "Try installing: sudo apt-get install texlive-latex-extra"
    echo ""
    echo "Or use online converter: https://www.markdowntopdf.com/"
    exit 1
fi

echo ""
echo "Done!"
