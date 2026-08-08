from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("resume")

PERSONA_INSTRUCTION = (
    "You are speaking as the candidate whose resume is shown below. "
    "The user is interviewing you about your background. Answer in "
    "FIRST PERSON ('I have...', 'I built...', 'I worked at...') using "
    "the resume facts. Never break character as an AI assistant, never "
    "say 'according to my resume', and never refer to the resume as an "
    "external document. Keep answers conversational and spoken-friendly, "
    "one to four sentences unless the question genuinely needs more depth. "
    "If a detail is not listed, say so naturally instead of guessing."
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

_RESUME_TXT_NAME = "resume/resume.txt"


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

    def to_prompt_block(self) -> str:
        return PERSONA_INSTRUCTION + PROFILE_PROMPT.format(
            name=self.name,
            headline=self.headline,
            contact=self.contact,
            summary=self.summary or "N/A",
        )

    def retrieval_sections(self) -> list[str]:
        """Chunks suitable for seeding the vector retrieval store."""
        return [
            f"{title}\n{body}"
            for title, body in self.sections
            if body.strip() and title != "PROFESSIONAL SUMMARY"
        ]


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
