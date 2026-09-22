"""Extracts the figures and result tables of the NWRD paper (Anwar et al., Sensors 2023,
doi:10.3390/s23156942) into md/figures/paper/, for md/paper_summary.md.

    pip install pymupdf
    python scripts/extract_paper_figures.py path/to/sensors-23-06942.pdf

Figures are the PDF's embedded images, copied at native resolution (downscaled only if wider
than MAX_WIDTH). Tables are vector text, so they are rendered as page crops. Nothing is redrawn.
"""
import io
import os
import sys

import fitz  # PyMuPDF
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'md', 'figures', 'paper')
MAX_WIDTH = 1600

# (output name, page number, embedded image xref) as found in the published PDF.
FIGURES = [
    ('fig1_data_collection_site.png', 5, 67),
    ('fig2_sample_images_and_masks.png', 6, 71),
    ('fig3_pipeline_flow_diagram.png', 7, 87),
    ('fig4_adaptive_patching.png', 9, 94),
    ('fig5_patch_examples.png', 9, 95),
    ('fig6_qualitative_results.png', 11, 102),
]
# (output name, page number, clip rectangle in PDF points: x0, y0, x1, y1), caption included.
TABLES = [
    ('table1_dataset_comparison.png', 3, (40, 367, 557, 560)),
    ('table2_patch_counts.png', 10, (160, 94, 561, 232)),
    ('table3_patching_results.png', 12, (38, 307, 561, 449)),
    ('table4_sota_comparison.png', 13, (40, 128, 561, 195)),
]


def main(pdf_path):
    os.makedirs(OUT, exist_ok=True)
    doc = fitz.open(pdf_path)
    for name, page_no, xref in FIGURES:
        found = {i['xref'] for i in doc[page_no - 1].get_image_info(xrefs=True)}
        assert xref in found, f'{name}: xref {xref} not on page {page_no}; is this the right PDF?'
        img = Image.open(io.BytesIO(doc.extract_image(xref)['image'])).convert('RGB')
        if img.width > MAX_WIDTH:
            img = img.resize((MAX_WIDTH, round(img.height * MAX_WIDTH / img.width)), Image.LANCZOS)
        img.save(os.path.join(OUT, name), optimize=True)
        print(f'{name}: page {page_no}, {img.width}x{img.height}')
    for name, page_no, clip in TABLES:
        pix = doc[page_no - 1].get_pixmap(matrix=fitz.Matrix(3, 3), clip=fitz.Rect(*clip))
        pix.save(os.path.join(OUT, name))
        print(f'{name}: page {page_no}, {pix.width}x{pix.height}')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, '..', 'sensors-23-06942.pdf'))
