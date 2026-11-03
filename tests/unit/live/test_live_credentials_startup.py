from __future__ import annotations

import pytest

from quant_fund.live.brokers import alpaca_credentials
from quant_fund.live.errors import LiveGateError

_CREDENTIAL_NAMES = (
    "ALPACA_API_KEY_ID",
    "ALPACA_API_SECRET_KEY",
    "APCA_API_KEY_ID",
    "APCA_API_SECRET_KEY",
)
_KEY = "SYNTHETIC-KEY-DO-NOT-USE"
_SECRET = "SYNTHETIC-SECRET-DO-NOT-USE"
_ALT_KEY = "SYNTHETIC-ALT-KEY-DO-NOT-USE"
_ALT_SECRET = "SYNTHETIC-ALT-SECRET-DO-NOT-USE"
_SENSITIVE = "SYNTHETIC-CREDENTIAL-VALUE-DO-NOT-LEAK"


@pytest.fixture(autouse=True)
def _clear_credential_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _CREDENTIAL_NAMES:
        monkeypatch.delenv(name, raising=False)


def _set_values(monkeypatch: pytest.MonkeyPatch, values: dict[str, str]) -> None:
    for name, value in values.items():
        monkeypatch.setenv(name, value)


def _set_pair(
    monkeypatch: pytest.MonkeyPatch,
    namespace: str,
    key: str = _KEY,
    secret: str = _SECRET,
) -> None:
    _set_values(
        monkeypatch,
        {
            f"{namespace}_API_KEY_ID": key,
            f"{namespace}_API_SECRET_KEY": secret,
        },
    )


def _set_pair_with_override(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
) -> None:
    namespace = name.split("_", maxsplit=1)[0]
    _set_pair(monkeypatch, namespace)
    monkeypatch.setenv(name, value)


def test_absent_credentials_return_none() -> None:
    assert alpaca_credentials() is None


@pytest.mark.parametrize(
    "value",
    [pytest.param("", id="empty"), pytest.param(" \t\r\n", id="whitespace")],
)
def test_blank_credentials_are_absent(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    for name in _CREDENTIAL_NAMES:
        monkeypatch.setenv(name, value)

    assert alpaca_credentials() is None


@pytest.mark.parametrize("namespace", ["ALPACA", "APCA"])
def test_each_complete_namespace_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
    namespace: str,
) -> None:
    _set_pair(monkeypatch, namespace)

    assert alpaca_credentials() == (_KEY, _SECRET)


@pytest.mark.parametrize("namespace", ["ALPACA", "APCA"])
def test_credentials_are_stripped(
    monkeypatch: pytest.MonkeyPatch,
    namespace: str,
) -> None:
    _set_pair(monkeypatch, namespace, key=f" \t{_KEY} ", secret=f"\t{_SECRET} \t")

    assert alpaca_credentials() == (_KEY, _SECRET)


@pytest.mark.parametrize("name", _CREDENTIAL_NAMES)
def test_each_single_credential_value_is_incomplete(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    monkeypatch.setenv(name, _SENSITIVE)

    with pytest.raises(LiveGateError, match="^broker credentials are incomplete$"):
        alpaca_credentials()


@pytest.mark.parametrize(
    "values",
    [
        pytest.param(
            {"ALPACA_API_KEY_ID": _KEY, "APCA_API_SECRET_KEY": _SECRET},
            id="alpaca-key-apca-secret",
        ),
        pytest.param(
            {"APCA_API_KEY_ID": _KEY, "ALPACA_API_SECRET_KEY": _SECRET},
            id="apca-key-alpaca-secret",
        ),
        pytest.param(
            {"ALPACA_API_KEY_ID": _KEY, "APCA_API_KEY_ID": _ALT_KEY},
            id="keys-without-secrets",
        ),
        pytest.param(
            {"ALPACA_API_SECRET_KEY": _SECRET, "APCA_API_SECRET_KEY": _ALT_SECRET},
            id="secrets-without-keys",
        ),
    ],
)
def test_cross_namespace_values_cannot_form_a_pair(
    monkeypatch: pytest.MonkeyPatch,
    values: dict[str, str],
) -> None:
    _set_values(monkeypatch, values)

    with pytest.raises(LiveGateError, match="^broker credentials are incomplete$"):
        alpaca_credentials()


@pytest.mark.parametrize("namespace", ["ALPACA", "APCA"])
@pytest.mark.parametrize("orphan_field", ["KEY_ID", "SECRET_KEY"])
def test_complete_pair_does_not_excuse_an_orphan_alias_value(
    monkeypatch: pytest.MonkeyPatch,
    namespace: str,
    orphan_field: str,
) -> None:
    other_namespace = "APCA" if namespace == "ALPACA" else "ALPACA"
    orphan_value = _KEY if orphan_field == "KEY_ID" else _SECRET
    _set_pair(monkeypatch, namespace)
    monkeypatch.setenv(f"{other_namespace}_API_{orphan_field}", orphan_value)

    with pytest.raises(LiveGateError, match="^broker credentials are incomplete$"):
        alpaca_credentials()


def test_identical_stripped_duplicate_namespace_pairs_are_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_pair(monkeypatch, "ALPACA", key=f" {_KEY} ", secret=f" {_SECRET} ")
    _set_pair(monkeypatch, "APCA")

    assert alpaca_credentials() == (_KEY, _SECRET)


@pytest.mark.parametrize(
    ("apca_key", "apca_secret"),
    [
        pytest.param(_ALT_KEY, _SECRET, id="different-key"),
        pytest.param(_KEY, _ALT_SECRET, id="different-secret"),
        pytest.param(_ALT_KEY, _ALT_SECRET, id="different-pair"),
    ],
)
def test_conflicting_duplicate_namespace_pairs_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
    apca_key: str,
    apca_secret: str,
) -> None:
    _set_pair(monkeypatch, "ALPACA")
    _set_pair(monkeypatch, "APCA", key=apca_key, secret=apca_secret)

    with pytest.raises(
        LiveGateError,
        match="^broker credentials conflict across namespaces$",
    ):
        alpaca_credentials()


@pytest.mark.parametrize("name", _CREDENTIAL_NAMES)
@pytest.mark.parametrize("control", ["\r", "\n", "\x00"])
def test_embedded_control_characters_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    control: str,
) -> None:
    namespace = name.split("_", maxsplit=1)[0]
    values = {
        f"{namespace}_API_KEY_ID": _KEY,
        f"{namespace}_API_SECRET_KEY": _SECRET,
        name: f"{_SENSITIVE}{control}TAIL",
    }
    monkeypatch.setattr("quant_fund.live.brokers.os.environ", values)

    with pytest.raises(LiveGateError, match="^broker credentials are malformed$"):
        alpaca_credentials()


@pytest.mark.parametrize("name", _CREDENTIAL_NAMES)
def test_200_character_credentials_are_accepted(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    value = ("SYNTHETIC-" + "X" * 200)[:200]
    _set_pair_with_override(monkeypatch, name, value)
    expected_key = value if name.endswith("KEY_ID") else _KEY
    expected_secret = value if name.endswith("SECRET_KEY") else _SECRET

    assert alpaca_credentials() == (expected_key, expected_secret)


@pytest.mark.parametrize("name", _CREDENTIAL_NAMES)
def test_credentials_over_200_characters_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    value = ("SYNTHETIC-" + "X" * 201)[:201]
    _set_pair_with_override(monkeypatch, name, value)

    with pytest.raises(LiveGateError, match="^broker credentials are incomplete$"):
        alpaca_credentials()


@pytest.mark.parametrize(
    ("values", "message"),
    [
        pytest.param(
            {"ALPACA_API_KEY_ID": f"{_SENSITIVE}-INCOMPLETE"},
            "broker credentials are incomplete",
            id="incomplete",
        ),
        pytest.param(
            {
                "ALPACA_API_KEY_ID": f"{_SENSITIVE}-KEY",
                "ALPACA_API_SECRET_KEY": f"{_SENSITIVE}-SECRET\nTAIL",
            },
            "broker credentials are malformed",
            id="malformed",
        ),
        pytest.param(
            {
                "ALPACA_API_KEY_ID": f"{_SENSITIVE}-{'X' * 201}",
                "ALPACA_API_SECRET_KEY": f"{_SENSITIVE}-SECRET",
            },
            "broker credentials are incomplete",
            id="too-long",
        ),
        pytest.param(
            {
                "ALPACA_API_KEY_ID": f"{_SENSITIVE}-ALPACA-KEY",
                "ALPACA_API_SECRET_KEY": f"{_SENSITIVE}-ALPACA-SECRET",
                "APCA_API_KEY_ID": f"{_SENSITIVE}-APCA-KEY",
                "APCA_API_SECRET_KEY": f"{_SENSITIVE}-APCA-SECRET",
            },
            "broker credentials conflict across namespaces",
            id="conflicting",
        ),
    ],
)
def test_credential_errors_never_include_values(
    monkeypatch: pytest.MonkeyPatch,
    values: dict[str, str],
    message: str,
) -> None:
    _set_values(monkeypatch, values)

    with pytest.raises(LiveGateError, match=f"^{message}$") as caught:
        alpaca_credentials()

    assert _SENSITIVE not in str(caught.value)
