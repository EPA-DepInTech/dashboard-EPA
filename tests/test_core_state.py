import importlib
import sys
import types


def test_state_helpers_store_and_get_file():
    dummy = types.SimpleNamespace(session_state={})
    sys.modules["streamlit"] = dummy

    from app.core import state
    importlib.reload(state)

    state.init_session_state()
    assert state.get_uploaded_file() is None

    token = object()
    state.set_uploaded_file(token)
    assert state.get_uploaded_file() is token
