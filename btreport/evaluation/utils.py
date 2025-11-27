import re


TOP_LEVEL = [
    "CLINICAL INDICATION",
    "TECHNIQUE",
    "CONTRAST",
    "COMPARISON",
    "FINDINGS",
    "HEAD MRA",
    "IMPRESSION"
]

SUBFINDINGS = [
    "MASS EFFECT & VENTRICLES",
    "BRAIN",
    "ENHANCEMENT",
    "VASCULAR",
    "EXTRA-AXIAL",
    "EXTRA-CRANIAL"
]

def parse_radiology_report(text):
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)

    header_regex = r"(?im)^(" + "|".join(TOP_LEVEL) + r")\s*:?\s*$"
    matches = list(re.finditer(header_regex, text))

    sections = {}
    for i,m in enumerate(matches):
        key = m.group(1).strip().upper()
        start = m.end()
        end   = matches[i+1].start() if i+1 < len(matches) else len(text)
        sections[key] = text[start:end].strip()

    if "FINDINGS" not in sections:
        found = re.search(r"(?is)FINDINGS\s*:\s*(.+?)(IMPRESSION|CONCLUSION|$)", text)
        if found:
            sections["FINDINGS"] = found.group(1).strip()
        else:
            sections["FINDINGS"] = ""  # gracefully return empty

    block = sections.get("FINDINGS", "")
    if block:
        sub_regex = r"(?im)^(" + "|".join(SUBFINDINGS) + r")\s*:?\s*"
        sub = list(re.finditer(sub_regex, block))

        for i,m in enumerate(sub):
            key = m.group(1).strip().upper()
            start = m.end()
            end   = sub[i+1].start() if i+1 < len(sub) else len(block)
            sections[key] = block[start:end].strip()

        sections["FINDINGS"] = ""  # remove summary container

    return sections
