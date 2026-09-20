# /// script
# requires-python = ">=3.10"
# dependencies = ["reportlab>=4"]
# ///
"""Generate Resources/sample-application.pdf: a fillable job application with real text
labels and standard AcroForm widgets (with appearance streams, so Preview renders
checked boxes). Run: uv run Tools/make_sample_pdf.py Sources/CuaFormsDemo/Resources/sample-application.pdf
"""
import sys

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

output = sys.argv[1]
width, height = letter
margin, gap, field_h = 54, 20, 22
col_w = (width - 2 * margin - gap) / 2

sections = [
    ("Personal information", [
        ("First name", False), ("Last name", False), ("Email address", False), ("Phone number", False),
        ("Street address", True), ("City", False), ("State", False), ("ZIP code", False), ("Country", False),
        ("Emergency contact phone", False), ("Referral code (optional)", False),
    ]),
    ("Professional profile", [
        ("LinkedIn profile URL", False), ("Website", False), ("Current employer", False), ("Job title", False),
        ("Desired salary", False), ("Earliest start date", False), ("University", False), ("Degree", False),
        ("Graduation year", False), ("How did you hear about us?", False),
    ]),
]
checkboxes = [
    "I am legally authorized to work in this country",
    "I certify the information above is accurate",
    "Subscribe to our newsletter",
]

c = canvas.Canvas(output, pagesize=letter)
c.setTitle("Example Robotics - Job Application")
form = c.acroForm
ink, muted, line = colors.HexColor("#111827"), colors.HexColor("#404040"), colors.HexColor("#cccccc")

c.setFillColor(colors.HexColor("#121728"))
c.rect(0, height - 78, width, 78, stroke=0, fill=1)
c.setFillColor(colors.white)
c.setFont("Helvetica-Bold", 20)
c.drawString(margin, height - 42, "Example Robotics")
c.setFillColor(colors.HexColor("#cccccc"))
c.setFont("Helvetica", 10)
c.drawString(margin, height - 62, "Job Application · Senior Software Engineer, Perception · Portland, OR")

y = height - 78 - 30
field_index = 0
for title, fields in sections:
    c.setFillColor(ink)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, title)
    c.setStrokeColor(line)
    c.setLineWidth(0.5)
    c.line(margin, y - 4, width - margin, y - 4)
    y -= 24
    column = 0
    for label, wide in fields:
        if wide and column == 1:
            column, y = 0, y - (field_h + 22)
        x = margin + column * (col_w + gap)
        w = width - 2 * margin if wide else col_w
        c.setFillColor(muted)
        c.setFont("Helvetica-Bold", 9)
        c.drawString(x, y, label)
        form.textfield(
            name=f"text_{field_index}", tooltip=label, x=x, y=y - 4 - field_h, width=w, height=field_h,
            borderStyle="solid", borderWidth=0.75, borderColor=colors.HexColor("#a6a6a6"),
            fillColor=colors.HexColor("#f5f7ff"), textColor=ink, fontSize=11, forceBorder=True)
        field_index += 1
        if wide or column == 1:
            column, y = 0, y - (field_h + 22)
        else:
            column = 1
    if column == 1:
        y -= field_h + 22
    y -= 6

c.setFillColor(ink)
c.setFont("Helvetica-Bold", 12)
c.drawString(margin, y, "Declarations")
y -= 22
for index, label in enumerate(checkboxes):
    form.checkbox(
        name=f"check_{index}", tooltip=label, x=margin, y=y - 2, size=12, buttonStyle="check",
        borderColor=colors.HexColor("#666666"), fillColor=colors.white, textColor=ink, borderWidth=0.75,
        forceBorder=True)
    c.setFillColor(ink)
    c.setFont("Helvetica", 10)
    c.drawString(margin + 20, y, label)
    y -= 20

c.setFillColor(colors.HexColor("#808080"))
c.setFont("Helvetica", 8)
c.drawString(margin, 18, "Form ER-JA-2026 · Page 1 of 1")
c.showPage()
c.save()
print(f"wrote {output}: {field_index} text fields, {len(checkboxes)} checkboxes")
