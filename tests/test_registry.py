import pytest

from hackstack import registry


@pytest.fixture
def clean_kind():
    """A throwaway kind so tests never collide with real registrations."""
    kind = "test_widget"
    yield kind
    registry.unregister_kind(kind)


def test_registered_component_is_found_by_its_config_name(clean_kind):
    @registry.register(clean_kind, "shiny")
    class Shiny:
        pass

    assert registry.get(clean_kind, "shiny") is Shiny


def test_register_returns_the_class_unchanged(clean_kind):
    class Plain:
        pass

    assert registry.register(clean_kind, "plain")(Plain) is Plain


def test_unknown_name_error_lists_what_is_available(clean_kind):
    @registry.register(clean_kind, "alpha")
    class Alpha:
        pass

    with pytest.raises(KeyError, match=r"'beta'.*alpha"):
        registry.get(clean_kind, "beta")


def test_registering_the_same_name_twice_is_refused(clean_kind):
    @registry.register(clean_kind, "dup")
    class First:
        pass

    with pytest.raises(ValueError, match="already registered"):

        @registry.register(clean_kind, "dup")
        class Second:
            pass


def test_available_lists_names_for_a_kind(clean_kind):
    @registry.register(clean_kind, "b")
    class B:
        pass

    @registry.register(clean_kind, "a")
    class A:
        pass

    assert registry.available(clean_kind) == ["a", "b"]
