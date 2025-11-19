# CogniSense Report

This directory contains the hackathon submission report for CogniSense.

---

## Files

- `CogniSense_Report.md` - Main report in Markdown format (2-3 pages)
- `CogniSense_Report.pdf` - PDF version for submission (generated)

---

## Converting to PDF

### Option 1: Using Pandoc (Recommended)

```bash
# Install pandoc (if not already installed)
# Ubuntu/Debian: sudo apt-get install pandoc texlive-latex-base texlive-fonts-recommended
# Mac: brew install pandoc basictex
# Windows: Download from pandoc.org

# Convert to PDF
pandoc CogniSense_Report.md -o CogniSense_Report.pdf \
    --pdf-engine=pdflatex \
    -V geometry:margin=1in \
    -V fontsize=11pt \
    -V documentclass=article \
    --highlight-style=tango

# With table of contents
pandoc CogniSense_Report.md -o CogniSense_Report.pdf \
    --pdf-engine=pdflatex \
    -V geometry:margin=1in \
    -V fontsize=11pt \
    --toc \
    --highlight-style=tango
```

### Option 2: Using Markdown to PDF Online

1. Go to https://www.markdowntopdf.com/
2. Upload `CogniSense_Report.md`
3. Download the generated PDF

### Option 3: Using Google Docs

1. Open Google Docs
2. Import `CogniSense_Report.md`
3. Format as needed
4. File → Download → PDF

### Option 4: Using Python (md2pdf)

```bash
pip install md2pdf

md2pdf CogniSense_Report.md
```

---

## Report Contents

The report (2-3 pages) includes:

1. **Abstract** - Problem, solution, key results
2. **Introduction** - Problem statement, our approach
3. **Methods** - Dataset, model architecture, training
4. **Results** - Individual models, fusion performance, ablation study
5. **Discussion** - Clinical implications, comparison, limitations, future work
6. **Conclusion** - Key contributions and impact
7. **References** - Academic sources
8. **Appendix** - Reproducibility information

---

## Page Count

The Markdown report is designed to be approximately 2.5 pages when converted to PDF with standard formatting:
- Font: 11pt
- Margins: 1 inch
- Line spacing: Single
- Includes: 3 tables, references

---

## Customization

To adjust formatting, modify the pandoc command:

```bash
# Smaller font (fit more content)
pandoc CogniSense_Report.md -o CogniSense_Report.pdf -V fontsize=10pt

# Larger margins
pandoc CogniSense_Report.md -o CogniSense_Report.pdf -V geometry:margin=1.5in

# Different style
pandoc CogniSense_Report.md -o CogniSense_Report.pdf -V documentclass=report
```

---

## Submission Checklist

- [ ] Convert Markdown to PDF
- [ ] Verify page count (2-3 pages)
- [ ] Check all tables render correctly
- [ ] Verify references are formatted
- [ ] Include in hackathon submission with:
  - [ ] Jupyter notebook (`notebooks/CogniSense_Demo.ipynb`)
  - [ ] GitHub repository link
  - [ ] This PDF report

---

For questions, see main README.md in repository root.
