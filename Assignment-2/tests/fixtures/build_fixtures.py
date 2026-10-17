#!/usr/bin/env python3
"""Turn tests/fixtures/<dataset>/source.json into the on-disk layout the parsers
actually read.

EB-NeRD fixtures are committed as JSON, not Parquet, for two reasons: `*.parquet`
is in config.FORBIDDEN_PATH_PATTERNS (Q8), and a binary fixture cannot be
reviewed in a diff. This script materialises the Parquet form at test time, so
the committed source of truth stays small and human-checkable.

**The dtypes below are part of the fixture, not an implementation detail.**
`pl.DataFrame(json_dicts)` infers `Utf8` for an ISO timestamp string and `Int64`
for every integer, so a naively built fixture has a different schema from the
real files — and a parser tested against it is being tested on input it will
never see. The first version of this script did exactly that, and `ebnerd.py`
failed on the real dtypes only after passing on the fixture.

The maps mirror `Data/EB-NeRD/*/**.parquet` as observed on 2026-08-20.
`tests/test_ebnerd.py::test_fixture_dtypes_match_the_real_files` asserts they
still do, so this cannot drift silently.

    python tests/fixtures/build_fixtures.py ebnerd_demo <dest-dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent


def _dtypes():
    import polars as pl

    articles = {
        "article_id": pl.Int32,
        "title": pl.String,
        "subtitle": pl.String,
        "body": pl.String,
        "category": pl.Int16,
        "category_str": pl.String,
        "subcategory": pl.List(pl.Int16),
        "article_type": pl.String,
        "published_time": pl.Datetime("us"),
        "last_modified_time": pl.Datetime("us"),
        "premium": pl.Boolean,
        "url": pl.String,
        "ner_clusters": pl.List(pl.String),
        "entity_groups": pl.List(pl.String),
        "topics": pl.List(pl.String),
        "sentiment_label": pl.String,
        "sentiment_score": pl.Float32,
        "total_inviews": pl.Int32,
        "total_pageviews": pl.Int32,
        "total_read_time": pl.Float32,
    }
    behaviors = {
        "impression_id": pl.UInt32,
        "user_id": pl.UInt32,
        "article_id": pl.Int32,
        "article_ids_inview": pl.List(pl.Int32),
        "article_ids_clicked": pl.List(pl.Int32),
        "impression_time": pl.Datetime("us"),
        "read_time": pl.Float32,
        "scroll_percentage": pl.Float32,
        "next_read_time": pl.Float32,
        "next_scroll_percentage": pl.Float32,
        "device_type": pl.Int8,
        "session_id": pl.UInt32,
        "is_sso_user": pl.Boolean,
        "is_subscriber": pl.Boolean,
        "gender": pl.Int8,
        "age": pl.Int8,
        "postcode": pl.Int8,
    }
    history = {
        "user_id": pl.UInt32,
        "article_id_fixed": pl.List(pl.Int32),
        "impression_time_fixed": pl.List(pl.Datetime("us")),
        "read_time_fixed": pl.List(pl.Float32),
        "scroll_percentage_fixed": pl.List(pl.Float32),
    }
    return articles, behaviors, history


def _frame(rows: list[dict], dtypes: dict):
    """Build a frame with the declared dtypes, parsing ISO strings to datetimes.

    Casting straight from Utf8 to Datetime does not parse; it errors. So string
    columns destined for a datetime are converted explicitly first.
    """
    import polars as pl

    df = pl.DataFrame(rows, infer_schema_length=None)
    exprs = []
    for name, dtype in dtypes.items():
        if name not in df.columns:
            continue
        col = pl.col(name)
        if isinstance(dtype, pl.Datetime) and df.schema[name] == pl.String:
            col = col.str.to_datetime(time_unit="us")
        elif isinstance(dtype, pl.List) and isinstance(dtype.inner, pl.Datetime):
            inner = df.schema[name]
            if isinstance(inner, pl.List) and inner.inner == pl.String:
                col = col.list.eval(pl.element().str.to_datetime(time_unit="us"))
        exprs.append(col.cast(dtype).alias(name))
    return df.select(exprs)


def build_ebnerd(source: Path, dest: Path) -> None:
    articles_dt, behaviors_dt, history_dt = _dtypes()
    data = json.loads(source.read_text(encoding="utf-8"))
    dest.mkdir(parents=True, exist_ok=True)

    _frame(data["articles"], articles_dt).write_parquet(dest / "articles.parquet")
    for split in ("train", "validation"):
        d = dest / split
        d.mkdir(exist_ok=True)
        _frame(data[split]["behaviors"], behaviors_dt).write_parquet(d / "behaviors.parquet")
        _frame(data[split]["history"], history_dt).write_parquet(d / "history.parquet")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    dataset, dest = argv[0], Path(argv[1])
    source = HERE / dataset / "source.json"
    if not source.exists():
        print(f"no source.json for {dataset}", file=sys.stderr)
        return 1
    build_ebnerd(source, dest)
    print(f"built {dataset} fixtures -> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
