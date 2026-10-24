"""PIT record clock: an announcement may be stamped before its event."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from quant_fund.schemas.pit import PITRecord


def test_pit_record_accepts_availability_before_and_after_the_event() -> None:
    event = datetime(2020, 1, 2, tzinfo=UTC)
    available_later = PITRecord(
        event_time=event,
        available_time=event + timedelta(hours=1),
        ingested_time=event + timedelta(hours=2),
        source="fixture",
        security_id="AAA",
        symbol="AAA",
    )
    assert available_later.available_time > available_later.event_time

    announced_early = available_later.model_copy(
        update={"available_time": event - timedelta(hours=1)}
    )
    assert announced_early.available_time < announced_early.event_time
    assert announced_early.security_id == "AAA"
