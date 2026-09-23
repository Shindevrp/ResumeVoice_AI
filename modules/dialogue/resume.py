from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("resume")

SPOKEN_ALIASES: dict[str, tuple[str, ...]] = {
    # (canonical resume name) -> spoken / STT-transcribed variants
    "IIIT Hyderabad": ("Triple IT Hyderabad", "Triple I T Hyderabad",
                       "IIIT-H", "IIITH", "IIIT"),
    "Nabh Technologies": ("Nabe", "Nabh", "Nabhe Technologies",
                          "Nabh Technologies Pvt Ltd"),
    "Plausibility Solutions": ("Plausibility", "Plausible Solutions"),
    "Eunoia Innovations": ("Eunoia", "Unola Innovations"),
    "BYOL Academy": ("BYOL", "Byol Academy"),
    "Patents": ("Patents & Publications", "Publications",
                "Patents and Publications"),
    "Certifications": ("Certifications, Awards & Honors", "Awards"),
    "Technical Skills": ("Tech Stack", "Tech Skills", "Technologies"),
}

PERSONA_INSTRUCTION = (
    "You are speaking as the candidate whose resume is shown below. "
    "The user is interviewing you about your background. Answer in "
    "FIRST PERSON ('I have...', 'I built...', 'I worked at...') using "
    "only the resume facts shown. Never break character, never say "
    "'according to my resume', never refer to the resume as an external "
    "document, and never invent or guess details.\n"
    "RESOLVE SPOKEN NAME VARIANTS: speech-to-text often renders "
    "organization names oddly (e.g. IIIT as 'Triple IT', Nabh as "
    "'Nabe'). When the user mentions a place, company, or section using "
    "a variant, map it to the matching resume entry and answer from "
    "that entry's real facts — never claim you don't recognize it and "
    "never invent generic filler (no 'industry partnerships', no "
    "'internship programs', no vague praise) if the resume does not "
    "actually list it.\n"
    "Be concrete and specific: when asked about companies you worked for, "
    "roles, technologies, schools, projects, publications, or years, name "
    "them explicitly from the profile and list every relevant item rather "
    "than giving a generic summary. Do NOT deflect with phrases like 'I "
    "don't have that listed' when the detail IS present in the profile. "
    "Only say a detail is unavailable if it truly is not there.\n"
    "Keep answers conversational and spoken-friendly — one to four "
    "sentences unless the question genuinely needs more depth. When the "
    "user asks about your technical stack or project work, impress with "
    "concrete specifics from the profile (tooling, model names, "
    "pipelines, quantisation targets) instead of vague phrasing."
)

PROFILE_PROMPT = (
    "\n\nCandidate profile (you are this person):\n"
    "Name: {name}\n"
    "Headline: {headline}\n"
    "Contact: {contact}\n"
    "Summary: {summary}"
)

SECTION_HEADERS = [
    "PROFESSIONAL SUMMARY",
    "PROFESSIONAL EXPERIENCE",
    "EDUCATION",
    "TECHNICAL SKILLS",
    "PATENTS & PUBLICATIONS",
    "KEY PROJECTS",
    "LEADERSHIP & COMMUNITY",
    "CERTIFICATIONS, AWARDS & HONORS",
    "LANGUAGES",
]

# Sections where only the entry header line matters for the compact profile
# index (e.g. "AI Engineer, Nabhe (Jan 2026 - Present)"); the full bullets
# are served on demand via retrieval.
INDEX_HEADERS_ONLY = {
    "PROFESSIONAL EXPERIENCE",
    "KEY PROJECTS",
    "LEADERSHIP & COMMUNITY",
    "PATENTS & PUBLICATIONS",
}

# Minimum length below which adjacent entries are merged so retrieval chunks
# stay meaningfully searchable (e.g. EDUCATION lines, skill categories).
MIN_ENTRY_CHARS = 100

_RESUME_TXT_NAME = "resume/resume.txt"


def _split_entries(body: str) -> list[str]:
    """Split a section body into logical entries.

    Entry headers are lines that are not bullet items (- / * / •); their
    following bullet lines belong to the same entry.
    """
    entries: list[str] = []
    current: list[str] = []
    for raw in body.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        if line.startswith(("-", "*", "•")):
            current.append(line)
        else:
            if current:
                entries.append("\n".join(current))
            current = [line]
    if current:
        entries.append("\n".join(current))

    merged: list[str] = []
    for entry in entries:
        if merged and len(merged[-1]) < MIN_ENTRY_CHARS:
            merged[-1] += "\n" + entry
        else:
            merged.append(entry)
    return merged


@dataclass
class ResumeData:
    name: str
    headline: str
    contact: str
    sections: list[tuple[str, str]] = field(default_factory=list)

    @property
    def summary(self) -> str:
        return self.get_section("PROFESSIONAL SUMMARY")

    def get_section(self, title: str) -> str:
        for section_title, body in self.sections:
            if section_title == title:
                return body
        return ""

    def _compact_index(self) -> list[tuple[str, str]]:
        """Per-section profile text: entry headers only for long sections,
        full body for the rest."""
        out: list[tuple[str, str]] = []
        for title, body in self.sections:
            if title == "PROFESSIONAL SUMMARY" or not body.strip():
                continue
            if title in INDEX_HEADERS_ONLY:
                heads = [
                    entry.splitlines()[0].strip()
                    for entry in _split_entries(body)
                    if entry.strip()
                ]
                if heads:
                    out.append((title, "\n".join(f"- {h}" for h in heads)))
            else:
                out.append((title, body))
        return out

    def to_prompt_block(self) -> str:
        block = PROFILE_PROMPT.format(
            name=self.name,
            headline=self.headline,
            contact=self.contact,
            summary=self.summary or "N/A",
        )
        for title, text in self._compact_index():
            block += f"\n\n{title}:\n{text}"
        block += "\n\nCOMMON SPOKEN / TRANSCRIBED NAME VARIANTS:\n"
        for canonical, variants in SPOKEN_ALIASES.items():
            block += f"- {canonical} = {', '.join(variants)}\n"
        return PERSONA_INSTRUCTION + block

    def retrieval_sections(self) -> list[str]:
        """Granular chunks suitable for seeding the vector retrieval store.

        Each section is split into per-entry chunks (company/role, project,
        publication, ...) so a query like "which companies did you intern
        at?" scores directly against the matching entry instead of being
        diluted across a whole multi-company section.
        """
        chunks: list[str] = []
        for title, body in self.sections:
            if title == "PROFESSIONAL SUMMARY" or not body.strip():
                continue
            entries = _split_entries(body)
            if len(entries) <= 1:
                chunks.append(f"{title}\n{body}")
            else:
                for entry in entries:
                    chunks.append(f"{title}\n{entry}")
        return chunks


def _split_sections(text: str) -> tuple[str, list[tuple[str, str]]]:
    lines = [line.rstrip() for line in text.splitlines()]
    header_index: dict[str, int] = {}
    for i, line in enumerate(lines):
        stripped = line.strip().upper()
        if stripped in SECTION_HEADERS:
            header_index[stripped] = i

    ordered: list[tuple[str, int]] = sorted(
        header_index.items(), key=lambda item: item[1]
    )
    sections: list[tuple[str, str]] = []
    for idx, (title, start) in enumerate(ordered):
        end = ordered[idx + 1][1] if idx + 1 < len(ordered) else len(lines)
        body = "\n".join(
            line for line in lines[start + 1 : end] if line.strip()
        ).strip()
        sections.append((title, body))

    header_lines_end = ordered[0][1] if ordered else len(lines)
    header_block = "\n".join(line for line in lines[:header_lines_end] if line.strip())
    return header_block, sections


def _build(text: str) -> ResumeData | None:
    if not text or not text.strip():
        return None
    header_block, sections = _split_sections(text)
    header_lines = header_block.splitlines()
    name = header_lines[0].strip() if header_lines else ""
    headline = header_lines[1].strip() if len(header_lines) > 1 else ""
    contact = header_lines[2].strip() if len(header_lines) > 2 else ""
    return ResumeData(
        name=name,
        headline=headline,
        contact=contact,
        sections=sections,
    )


def _default_resume_path() -> Path:
    root = Path(__file__).resolve().parent.parent.parent
    txt = root / _RESUME_TXT_NAME
    if txt.exists():
        return txt
    pdfs = sorted(root.glob("*.pdf"))
    if pdfs:
        return pdfs[0]
    return txt


def load_resume_data(path: str | Path | None = None) -> ResumeData | None:
    """Load resume knowledge from a text file or PDF.

    Text files (.txt/.md) are read directly. PDFs are parsed with pypdf
    (optional dependency) and cached next to the source file. When `path`
    is None, `resume/resume.txt` is used if present, otherwise the first
    PDF found at the project root.
    """
    resume_path = Path(path) if path else _default_resume_path()
    if not resume_path.exists():
        logger.warning("resume not found at %s", resume_path)
        return None

    try:
        if resume_path.suffix.lower() == ".pdf":
            text = _extract_pdf_text(resume_path)
            if not text:
                logger.warning("no text extracted from %s", resume_path)
                return None
            cache_path = resume_path.with_suffix(".txt")
            try:
                cache_path.write_text(text, encoding="utf-8")
            except OSError as e:
                logger.debug("could not cache resume text: %s", e)
        else:
            text = resume_path.read_text(encoding="utf-8", errors="ignore")
        data = _build(text)
        if data is None:
            logger.warning("resume at %s was empty", resume_path)
        elif data.name:
            logger.info("resume loaded: %s (%s)", data.name, resume_path)
        return data
    except Exception as e:
        logger.error("failed to load resume from %s: %s", resume_path, e)
        return None


def _extract_pdf_text(pdf_path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as e:
        logger.warning(
            "pypdf is not installed; cannot parse %s (%s). "
            "Install with: pip install 'resumevoice-ai[resume]'",
            pdf_path,
            e,
        )
        return ""
    reader = PdfReader(str(pdf_path))
    pages: list[str] = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)
