"""Tests for registry module — LazyDict and helpers."""

from policy_arena.registry import (
    BRAIN_FACTORIES,
    MODEL_CLASSES,
    _filter_kwargs,
    _LazyDict,
    _make_filtered_factory,
)


class TestLazyDict:
    def test_lazy_loading(self):
        """Dict should not call factory until first access."""
        call_count = 0

        def factory():
            nonlocal call_count
            call_count += 1
            return {"a": 1, "b": 2}

        d = _LazyDict(factory)
        assert call_count == 0

        # Access triggers loading
        assert d["a"] == 1
        assert call_count == 1

        # Subsequent access doesn't re-load
        assert d["b"] == 2
        assert call_count == 1

    def test_contains(self):
        d = _LazyDict(lambda: {"x": 10})
        assert "x" in d
        assert "y" not in d

    def test_iter(self):
        d = _LazyDict(lambda: {"a": 1, "b": 2})
        assert set(d) == {"a", "b"}

    def test_len(self):
        d = _LazyDict(lambda: {"a": 1, "b": 2, "c": 3})
        assert len(d) == 3

    def test_keys_values_items(self):
        d = _LazyDict(lambda: {"x": 1, "y": 2})
        assert set(d.keys()) == {"x", "y"}
        assert set(d.values()) == {1, 2}
        assert set(d.items()) == {("x", 1), ("y", 2)}

    def test_get(self):
        d = _LazyDict(lambda: {"a": 1})
        assert d.get("a") == 1
        assert d.get("b") is None
        assert d.get("b", 42) == 42


class TestFilterKwargs:
    def test_filters_correctly(self):
        result = _filter_kwargs(
            {"a": 1, "b": 2, "c": None, "d": 3},
            frozenset({"a", "c", "d"}),
        )
        assert result == {"a": 1, "d": 3}

    def test_empty_input(self):
        assert _filter_kwargs({}, frozenset({"a"})) == {}

    def test_empty_allowed(self):
        assert _filter_kwargs({"a": 1}, frozenset()) == {}


class TestMakeFilteredFactory:
    def test_wraps_correctly(self):
        def my_factory(x=0, y=0):
            return x + y

        wrapped = _make_filtered_factory(my_factory, frozenset({"x"}))
        # y should be filtered out
        assert wrapped(x=5, y=10) == 5

    def test_none_values_filtered(self):
        def my_factory(x=0):
            return x

        wrapped = _make_filtered_factory(my_factory, frozenset({"x"}))
        assert wrapped(x=None) == 0  # None filtered out, default used


class TestGlobalRegistries:
    def test_model_classes_loads(self):
        assert len(MODEL_CLASSES) > 0
        # Should contain known game IDs
        assert "prisoners_dilemma" in MODEL_CLASSES

    def test_brain_factories_loads(self):
        assert len(BRAIN_FACTORIES) > 0
        assert "prisoners_dilemma" in BRAIN_FACTORIES
        # Each game should have at least one brain factory
        for game_id, factories in BRAIN_FACTORIES.items():
            assert len(factories) > 0, f"{game_id} has no brain factories"
