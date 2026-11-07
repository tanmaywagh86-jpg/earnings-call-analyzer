import pytest

from concall.manifest import COLUMNS, load_manifest, make_call_id

HEADER = ",".join(COLUMNS)


def write(tmp_path, *rows):
    p = tmp_path / "calls.csv"
    p.write_text("\n".join([HEADER, *rows]) + "\n", encoding="utf-8")
    return p


def test_call_id_is_built_when_blank(tmp_path):
    p = write(tmp_path, ",HDFC Bank,HDFCBANK,Banks,FY25Q2,2024-10-19,16:00,x.mp3,")
    df = load_manifest(p)
    assert list(df.index) == ["HDFCBANK_FY25Q2"]
    assert make_call_id("infy ", "fy25q1") == "INFY_FY25Q1"


def test_bad_rows_are_reported(tmp_path):
    p = write(tmp_path,
              "A_1,A,A,IT,2025Q1,2024-10-19,16:00,x,",      # bad quarter format
              "B_1,B,B,IT,FY25Q1,not-a-date,4pm,x,")        # bad date and time
    with pytest.raises(ValueError) as e:
        load_manifest(p)
    msg = str(e.value)
    assert "FY25Q2" in msg and "call_date" in msg and "HH:MM" in msg


def test_duplicate_ids_rejected(tmp_path):
    row = "X_FY25Q1,X,X,IT,FY25Q1,2024-07-10,16:00,x,"
    with pytest.raises(ValueError, match="duplicate"):
        load_manifest(write(tmp_path, row, row))
