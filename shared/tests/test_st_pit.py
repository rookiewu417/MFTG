"""build_base.py marks ST from max(start_date, ann_date); check the lake still satisfies the assumption noted there."""
from pathlib import Path
import pytest

NAME = Path('/home/ivenwu/projects/data/ashare/meta/name.parquet')


@pytest.mark.skipif(not NAME.exists(), reason='data lake not present')
def test_st_names_are_announced_by_their_start_date():
    import duckdb
    late, = duckdb.sql(f"""select count(*) from '{NAME}' where regexp_matches(name, 'ST|退')
                           and (ann_date is null or ann_date > start_date)""").fetchone()
    assert late == 0, f'{late} ST/退 name records become known after start_date; rebuild base and re-check univ'
