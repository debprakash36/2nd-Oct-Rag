"""Generate a synthetic corpus for the Phase 1 exit gate.

Documents are generated rather than committed: 100 real policy documents would be
noise in the repository, and the properties under test are properties of the
*shape* of the input (headings, wrapping, multiple formats, duplicate content),
not of any specific document.

Content is varied deliberately so the run exercises the paths that matter:

* multiple formats, since each has its own extractor
* documents long enough to produce more than one chunk, so the chunker and the
  offset invariant are exercised rather than only the trivial single-chunk case
* headings at multiple depths, for breadcrumb construction
* deliberate near-duplicates, so the dedupe path runs at corpus scale
"""

from __future__ import annotations

import argparse
from pathlib import Path

#: Each topic is a *pair*: a document title, and the sentence that states the
#: topic's parameter. Both `{scope}` and `{days}` are substituted per document.
#:
#: `{scope}` is what makes a title-only question answerable. A corpus of N
#: documents all titled "Refund Policy", each stating a different figure, has N
#: equally valid answers to "what is the refund window?" — so an eval set that
#: labels one of them measures luck, not retrieval. Real policy corpora avoid
#: this by scoping each document to a subject; `{scope}` does the same here, and
#: it is also what gives the vector stage something to discriminate on beyond the
#: shared title.
TOPICS = [
    ("Refund Policy", "customers may request a refund within {days} days of purchase for {scope}"),
    ("Shipping Policy", "standard shipping arrives within {days} business days for {scope}"),
    ("Warranty Terms", "products are covered for {days} months from the delivery date for {scope}"),
    ("Privacy Notice",
     "personal data is retained for {days} days after account closure for {scope}"),
    ("Acceptable Use", "automated access is limited to {days} requests per hour for {scope}"),
    ("Data Retention", "logs are retained for {days} days and then deleted for {scope}"),
    ("Billing Terms", "invoices are due within {days} days of issue for {scope}"),
    ("Service Levels", "uptime of 99.{days} percent is committed for paid plans covering {scope}"),
    ("Cancellation Policy", "subscriptions may be cancelled within {days} days for {scope}"),
    ("Support Policy", "first response is provided within {days} business hours for {scope}"),
]

#: Per-topic subject scopes. Twelve per topic so the 100-document corpus gives
#: every topic a distinct subject per document. These are the discriminators that
#: let a title-and-subject question single out exactly one document.
SCOPES = [
    ["Digital Goods", "Physical Goods", "Subscription Services", "Gift Cards",
     "Marketplace Orders", "Business Accounts", "Pre-Orders", "Clearance Items",
     "International Orders", "Third-Party Resellers", "Installment Plans",
     "Educational Licences"],
    ["Domestic Standard", "International Express", "Overnight Delivery",
     "Freight and Pallets", "Digital Downloads", "Cold Chain",
     "Hazmat Materials", "Oversized Items", "Returns Processing",
     "Scheduled Slots", "Courier Handoff", "Last Mile Delivery"],
    ["Manufacturing Defects", "Extended Coverage", "Consumable Parts",
     "Structural Components", "Software and Firmware", "Electrical Components",
     "Outdoor Equipment", "Vehicle Parts", "Labour Costs", "Accidental Damage",
     "Wear and Tear", "Transfer of Ownership"],
    ["Analytics and Telemetry", "Marketing Communications", "Payment Data",
     "Health Information", "Location Tracking", "Account Credentials",
     "Support Records", "Partner Sharing", "Subject Access Requests",
     "Retention and Deletion", "International Transfers", "Minor Users"],
    ["API Access", "Automated Agents", "High-Volume Crawling", "Account Sharing",
     "Credential Storage", "Network Scanning", "Bulk Export",
     "Resale Accounts", "Malicious Code", "Circumventing Limits",
     "Third-Party Integration", "Training on Data"],
    ["Application Logs", "Billing Records", "Support Tickets", "Security Events",
     "Backup Snapshots", "Analytics Events", "Consent Records", "Audit Trails",
     "Financial Reports", "Marketing Events", "Crash Reports", "Training Data"],
    ["Monthly Invoices", "Annual Prepayment", "Usage Billing", "Credit Terms",
     "Overdue Accounts", "Tax Documentation", "Purchase Orders",
     "Multi-Currency", "Disputed Charges", "Automatic Renewal",
     "Purchase Cards", "Purchase Order Amendments"],
    ["Scheduled Maintenance", "Incident Response", "Data Residency",
     "Network Latency", "API Uptime", "Storage Durability",
     "Recovery Objectives", "Support Coverage", "Status Notifications",
     "Capacity Limits", "Regional Failover", "Change Management"],
    ["Monthly Subscriptions", "Annual Subscriptions", "Free Trials",
     "Enterprise Contracts", "Gift Subscriptions", "Student Discounts",
     "Promotional Offers", "Legacy Plans", "Partner Resellers",
     "Non-Renewal Notices", "Data Export on Exit", "Auto-Renewal Opt-Out"],
    ["Standard Support", "Priority Support", "Enterprise Support",
     "Technical Incidents", "Billing Disputes", "Account Access",
     "Security Questions", "Feature Requests", "Documentation Access",
     "Response Times", "Escalation Paths", "Weekend Coverage"],
]


def scope_for(topic: int, index: int) -> str:
    """The subject scope for one document.

    Determined by `index` rather than drawn at random so the corpus, and
    therefore every label derived from it, is reproducible.
    """
    scopes = SCOPES[topic % len(SCOPES)]
    return scopes[index % len(scopes)]


def document_title(topic: int, index: int) -> str:
    """Full document title, scope included.

    The scope goes in the title as well as the body because the title is what a
    human would search on, and a title that cannot distinguish the document from
    its eleven siblings is not a useful one.
    """
    return f"{TOPICS[topic % len(TOPICS)][0]} — {scope_for(topic, index)}"

BODY_SENTENCES = [
    "This clause applies to all customers unless a separate written agreement states otherwise.",
    "Where local law requires more, local law takes precedence over this document.",
    "Exceptions require written approval from an authorised account manager.",
    "Requests received outside the stated window are evaluated at our discretion.",
    "Nothing in this document limits your statutory consumer rights.",
    "We may update this document; the version in effect at the time of your request governs.",
    "Contact support with your account identifier before submitting any dispute.",
    "Records supporting this decision are retained for audit purposes only.",
]

#: Short document codes, one per topic. Combined with the document index they
#: form the unique identifier `POL-<CODE>-<n>`, which is what makes the
#: exact-identifier class of the Phase 2 eval set answerable at all. See
#: eval/build_dataset.py.
TOPIC_CODES = ["RFD", "SHP", "WAR", "PRV", "AUP", "RET", "BIL", "SLV", "CAN", "SUP"]

#: Invented surnames for the accountable owner line. Chosen as names that do not
#: appear anywhere else in the corpus, so a question naming one has exactly one
#: correct answer document.
OWNERS = [
    "Vasquez", "Okonkwo", "Lindqvist", "Nakamura", "Ferreira",
    "Halloran", "Petrossian", "Achterberg", "Mbeki", "Sorensen",
]


def _identifier(topic: int, index: int) -> str:
    """Unique document identifier, e.g. ``POL-RFD-042``."""
    return f"POL-{TOPIC_CODES[topic % len(TOPIC_CODES)]}-{index:03d}"


def _effective_date(index: int) -> str:
    """Deterministic ISO date derived from the document index.

    Real dates rather than 'Day 42' so the identifier-matching code paths are
    exercised on the same token shapes a real corpus would contain.
    """
    from datetime import date, timedelta

    return (date(2024, 1, 1) + timedelta(days=(index * 37) % 700)).isoformat()


IDENTIFIER_SENTENCES = [
    "Reference: {ref}.",
    "This document is {ref} and takes effect on {date}.",
    "The accountable owner for {ref} is {owner}.",
]


def _doc_text(
    title: str,
    topic: int,
    days: int,
    paragraphs: int = 40,
    *,
    index: int = 0,
    with_identifiers: bool = True,
) -> str:
    """Build a document body.

    `paragraphs` is deliberately large. With the default 1000-token target, a
    document under roughly 1000 tokens produces exactly one chunk, which would
    leave the chunker, the overlap logic, and the offset invariant exercised only
    in unit tests. The exit gate is only meaningful if the corpus actually spans
    multiple chunks per document.

    `with_identifiers` adds the unique reference, date and owner lines that the
    Phase 2 eval set labels against. They go into section 1 only, so the eval
    set can point at a specific section rather than at "somewhere in the
    document" -- a question with an unlocatable answer cannot distinguish a
    retrieval miss from a labelling error.
    """
    lines = [f"# {title}", ""]
    if with_identifiers:
        ref = _identifier(topic, index)
        lines.extend(
            [
                IDENTIFIER_SENTENCES[0].format(ref=ref),
                IDENTIFIER_SENTENCES[1].format(ref=ref, date=_effective_date(index)),
                IDENTIFIER_SENTENCES[2].format(
                    ref=ref, owner=OWNERS[index % len(OWNERS)]
                ),
                "",
            ]
        )
    scope = scope_for(topic, index)
    for section in range(8):
        lines.extend([f"## Section {section + 1}", ""])
        for p in range(paragraphs // 8):
            # `days` is rendered unchanged in every paragraph. It was previously
            # `days + p`, which made a single document state a different figure
            # in each section -- 14 days in section 1, 18 in section 5 -- so
            # "the stated limit" had no single answer even for a reader holding
            # only this document. That made every label derived from it a
            # guess about which figure was meant.
            template = TOPICS[topic][1].format(days=days, scope=scope)
            body = " ".join(
                BODY_SENTENCES[(topic + section + p) % len(BODY_SENTENCES)]
                for _ in range(3)
            )
            # Hard-wrapped at ~70 chars, like a real document. Exercises the
            # cleaner's line joining, which changes offsets.
            wrapped = []
            line = ""
            for word in f"{template} {body}".split():
                if len(line) + len(word) + 1 > 70:
                    wrapped.append(line)
                    line = word
                else:
                    line = f"{line} {word}".strip()
            wrapped.append(line)
            lines.extend(wrapped)
            lines.append("")
    return "\n".join(lines)


def write_corpus(out_dir: Path, count: int) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for i in range(count):
        topic = i % len(TOPICS)
        title = document_title(topic, i)
        text = _doc_text(
            title,
            topic,
            days=14 + (i % 60),
            index=i,
        )

        suffix = [".md", ".txt", ".html"][i % 3]
        path = out_dir / f"doc-{i:03d}{suffix}"

        if suffix == ".html":
            path.write_text(_to_html(title, text), encoding="utf-8")
        else:
            path.write_text(text, encoding="utf-8")

        written.append(path)

    # Near-duplicates: same content, different filename. Exercises the dedupe
    # path at corpus scale rather than only in a unit test.
    for src in written[: max(count // 10, 1)]:
        dup = src.with_name("dup-" + src.name)
        dup.write_bytes(src.read_bytes())
        written.append(dup)

    return written


def _to_html(title: str, text: str) -> str:
    """Render the same content as HTML, for the HTML extractor to read.

    Headings become real `h1`/`h2` elements rather than markdown syntax inside
    `<p>` tags. The point of including HTML in the corpus is that the extractor
    must recover document structure from markup; feeding it markdown-in-HTML
    would test the wrong path.
    """
    parts = [f"<html><head><title>{title}</title></head><body>"]
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        escaped = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        if escaped.startswith("### "):
            parts.append(f"<h3>{escaped[4:]}</h3>")
        elif escaped.startswith("## "):
            parts.append(f"<h2>{escaped[3:]}</h2>")
        elif escaped.startswith("# "):
            parts.append(f"<h1>{escaped[2:]}</h1>")
        else:
            parts.append(f"<p>{escaped}</p>")
    parts.append("</body></html>")
    return "\n".join(parts)


def write_pdf(out_dir: Path, name: str, text: str) -> Path:
    """Minimal single-page PDF, hand-built to avoid adding a PDF-writing dependency.

    A one-line summary: only the extraction path needs exercising here, and
    pypdf already covers reading a real PDF, so generating one with a heavier
    dependency would add risk without adding coverage.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    body = text.replace("(", r"\(").replace(")", r"\)")
    content = f"BT /F1 9 Tf 40 750 Td ({body[:1200]}) Tj ET".encode("latin-1", "replace")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body_obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body_obj + b"\nendobj\n"

    xref_pos = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n"
        f"{xref_pos}\n%%EOF\n"
    ).encode()

    path = out_dir / name
    path.write_bytes(bytes(out))
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a synthetic test corpus")
    parser.add_argument("--out", default="./samples", help="output directory")
    parser.add_argument("--count", type=int, default=100, help="number of documents")
    args = parser.parse_args()

    out_dir = Path(args.out)
    written = write_corpus(out_dir, args.count)
    for i in range(min(3, args.count)):
        # Offset past the text documents' index range so the PDF variants do not
        # reuse an identifier already assigned to doc-<i>. The same offset is
        # applied to the scope so the PDF does not collide with doc-<i> either.
        written.append(
            write_pdf(
                out_dir,
                f"doc-pdf-{i:03d}.pdf",
                _doc_text(
                    document_title(i, 1000 + i), i, days=30, index=1000 + i
                ),
            )
        )

    print(f"wrote {len(written)} files to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())