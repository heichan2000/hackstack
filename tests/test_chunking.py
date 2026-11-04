from hackstack.chunking.markdown import MarkdownChunker, count_tokens
from hackstack.models import ACL, ParsedDoc, RawDoc
from hackstack.parsing.sections import split_sections
from hackstack.registry import CHUNKER, get


def raw(title="Handbook", acl=("group:everyone",)) -> RawDoc:
    return RawDoc(
        doc_id="local_files:handbook.md",
        source="local_files",
        title=title,
        content="",
        acl=ACL(principals=list(acl)),
    )


def parsed(md: str) -> ParsedDoc:
    return ParsedDoc(markdown=md, sections=split_sections(md))


def words(n: int, prefix: str = "w") -> str:
    return " ".join(f"{prefix}{i}" for i in range(n))


def test_breadcrumb_and_acl_and_ordinals():
    md = "# Handbook\n\n## PTO\n\n### Carryover\n\n" + words(80) + "\n"
    chunks = MarkdownChunker().chunk(parsed(md), raw())
    assert len(chunks) == 1
    c = chunks[0]
    assert c.section_path == ["Handbook", "PTO", "Carryover"]
    # the H1 repeats the title, so the breadcrumb does not say it twice
    assert c.embed_text.startswith("Handbook > PTO > Carryover\n\n")
    assert c.embed_text.endswith(c.text)
    assert c.acl_principals == ["group:everyone"]
    assert c.ordinal == 0 and c.source == "local_files"


def test_breadcrumb_without_sections_is_the_title():
    chunks = MarkdownChunker().chunk(parsed(words(60)), raw(title="Notes"))
    assert chunks[0].embed_text.startswith("Notes\n\n")


def test_long_sections_split_near_target_and_never_exceed_max():
    paras = "\n\n".join(words(90, prefix=f"p{i}x") for i in range(20))  # ~1800 tokens
    chunker = MarkdownChunker(size=400, max_size=512)
    chunks = chunker.chunk(parsed("# Big\n\n" + paras), raw())
    sizes = [count_tokens(c.text) for c in chunks]
    assert len(chunks) >= 4
    assert max(sizes) <= 512
    assert all(s >= 200 for s in sizes[:-1])  # packed, not one paragraph per chunk
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_one_huge_paragraph_is_split_recursively():
    chunks = MarkdownChunker(size=100, max_size=128).chunk(parsed(words(1000)), raw())
    assert all(count_tokens(c.text) <= 128 for c in chunks)
    assert " ".join(c.text for c in chunks).split() == words(1000).split()


def test_chunk_ids_carry_chunker_version_and_are_stable():
    md = "# A\n\n" + words(100) + "\n\n# B\n\n" + words(100, "v")
    one = MarkdownChunker().chunk(parsed(md), raw())
    two = MarkdownChunker().chunk(parsed(md), raw())
    assert [c.chunk_id for c in one] == [c.chunk_id for c in two]
    version = MarkdownChunker().version
    assert all(c.chunk_id == f"local_files:handbook.md#{version}-{c.ordinal}" for c in one)
    # re-chunking with another size must not silently reuse the same ids
    assert MarkdownChunker(size=200).version != version


def test_heading_only_sections_make_no_fragments():
    md = (
        "# Handbook\n\n## Benefits\n\n### Health\n\n"
        + words(120)
        + "\n\n## Empty\n\n## Also empty\n"
    )
    chunks = MarkdownChunker().chunk(parsed(md), raw())
    assert len(chunks) == 1
    assert chunks[0].text.strip() and "Empty" not in chunks[0].text
    assert all(c.text.strip() for c in chunks)


def test_tiny_sections_merge_into_a_neighbour():
    md = (
        "# Handbook\n\n## PTO\n\nSee below.\n\n### Carryover\n\n" + words(120) + "\n\n"
        "### Holidays\n\n" + words(120, "h") + "\n"
    )
    chunks = MarkdownChunker(min_size=20).chunk(parsed(md), raw())
    assert all(count_tokens(c.text) >= 20 for c in chunks)
    first = chunks[0]
    assert first.text.startswith("See below.")
    assert "Carryover" in first.text  # the child heading is kept when merging
    assert first.section_path == ["Handbook", "PTO"]


def test_tables_are_kept_whole():
    rows = "\n".join(f"| row{i} | {words(12, f'r{i}c')} |" for i in range(15))  # ~300 tokens
    table = "| name | notes |\n|---|---|\n" + rows
    md = "# T\n\n" + words(300) + "\n\n" + table + "\n\n" + words(50, "z")
    chunks = MarkdownChunker(size=400, max_size=512).chunk(parsed(md), raw())
    holders = [c for c in chunks if "| row0 |" in c.text]
    assert len(holders) == 1
    assert "| row14 |" in holders[0].text  # the whole table in one chunk


def test_oversized_tables_split_by_rows_repeating_the_header():
    rows = "\n".join(f"| row{i} | {words(20, f'r{i}c')} |" for i in range(60))  # >1000 tokens
    table = "| name | notes |\n|---|---|\n" + rows
    chunks = MarkdownChunker(size=400, max_size=512).chunk(parsed("# T\n\n" + table), raw())
    assert len(chunks) > 1
    for c in chunks:
        assert c.text.startswith("| name | notes |\n|---|---|\n")
        assert count_tokens(c.text) <= 512
    body = "\n".join(c.text for c in chunks)
    assert all(f"| row{i} |" in body for i in range(60))


def test_overlap_repeats_the_tail_of_the_previous_chunk():
    paras = "\n\n".join(words(150, prefix=f"p{i}x") for i in range(6))
    chunks = MarkdownChunker(size=200, overlap=20).chunk(parsed(paras), raw())
    assert len(chunks) >= 3
    tail = chunks[0].text.split()[-5:]
    assert chunks[1].text.split()[: len(tail) + 20] != []
    assert " ".join(tail) in " ".join(chunks[1].text.split()[:25])


def test_chunker_is_registered_and_built_from_settings():
    from hackstack.config import load_settings

    cls = get(CHUNKER, "markdown")
    assert cls is MarkdownChunker
    settings = load_settings(profile="default")
    settings.chunk.size = 300
    settings.chunk.overlap = 10
    c = MarkdownChunker.from_settings(settings)
    assert (c.size, c.overlap) == (300, 10)
