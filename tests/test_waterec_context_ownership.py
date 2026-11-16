"""Retained wrappers may compare equal while owning distinct stack entries."""
from types import SimpleNamespace
import pytest


def fake_contexts(monkeypatch):
    from parseq_ocr import waterec_runtime as module

    stack = []

    class Context:
        handle = 7

        def __init__(self, token):
            self.token = token

        def __hash__(self):
            return self.token

        def __eq__(self, other):
            return self.handle == other.handle

        def push(self):
            stack.append(self)

    api = SimpleNamespace(
        get_current=lambda: stack[-1] if stack else None, pop=lambda: stack.pop()
    )
    monkeypatch.setattr(module, "cuda", SimpleNamespace(Context=api))
    return module, Context, stack


def test_exact_wrapper_temporarily_suspends_others(monkeypatch):
    module, Context, stack = fake_contexts(monkeypatch)
    below, owned, above, top = [Context(i) for i in range(4)]
    stack.extend([below, owned, above, top])
    original = [id(x) for x in stack]
    assert owned == above and hash(owned) != hash(above)
    with module._owned_context(owned):
        assert stack[-1] is owned
        assert len(stack) == 2
    assert [id(x) for x in stack] == original


def test_missing_owned_wrapper_does_not_release_unrelated_context(monkeypatch):
    module, Context, stack = fake_contexts(monkeypatch)
    stack.extend([Context(1), Context(2)])
    original = [id(x) for x in stack]

    def unexpected_driver_call(*args):
        raise AssertionError("An unrelated driver entry must not be released")

    monkeypatch.setattr(module.cuda_driver, "cuCtxGetCurrent", unexpected_driver_call)
    monkeypatch.setattr(module.cuda_driver, "cuCtxPopCurrent", unexpected_driver_call)
    with pytest.raises(RuntimeError, match="not on this thread"):
        module._release_context(Context(3))
    assert [id(x) for x in stack] == original


def test_operation_failure_restores_suspended_wrappers(monkeypatch):
    module, Context, stack = fake_contexts(monkeypatch)
    owned, above = Context(1), Context(2)
    stack.extend([owned, above])
    with pytest.raises(ValueError, match="operation failed"):
        with module._owned_context(owned):
            raise ValueError("operation failed")
    assert stack[0] is owned and stack[1] is above
