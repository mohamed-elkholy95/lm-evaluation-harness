from __future__ import annotations

import logging
from functools import partial

from lm_eval.api.filter import FilterEnsemble
from lm_eval.api.registry import filter_registry, get_filter
from lm_eval.tasks._yaml_loader import _import_fun_from_str

from . import custom, extraction, selection, transformation

eval_logger = logging.getLogger(__name__)


def build_filter_ensemble(
        filter_name: str,
        components: list[tuple[str, dict[str, str | int | float] | None]],
) -> FilterEnsemble:
        """
            Create a filtering pipeline.

                Handles filter functions specified as:
                    - A registered filter name (string looked up in the filter registry)
                        - A callable (class or function resolved via !function YAML tag)
                            - An absolute file path string from an unresolved !function tag
                                  (e.g. '/path/to/utils.py/MultiChoiceRegexFilter'), which is
                                        dynamically imported as a fallback.
                                            """
        filters = []
        for func, kwargs in components:
                    resolved = _resolve_filter(func)
                    filters.append(partial(resolved, **(kwargs or {})))

        return FilterEnsemble(
            name=filter_name,
            filters=filters,
        )


def _resolve_filter(func):
        """Resolve a filter function from a name, callable, or file path.

            This function handles three cases:
                1. ``func`` is already callable (e.g. a class resolved by the
                       ``!function`` YAML tag) -- returned as-is.
                           2. ``func`` is a string registered in the filter registry --
                                  looked up and returned.
                                      3. ``func`` is an absolute file path string produced when
                                             ``!function`` tags are loaded with ``resolve_func=False``
                                                    (e.g. ``'/abs/path/to/utils.MultiChoiceRegexFilter'``) --
                                                           dynamically imported via :func:`_import_fun_from_str`.

                                                               Raises :class:`KeyError` if none of the strategies succeed.
                                                                   """
        # Case 1: already a callable (resolved !function or Python class)
        if callable(func):
                    return func

        # Case 2: registered filter name (e.g. "take_first", "regex")
        try:
                    return get_filter(func)
except KeyError:
        pass

    # Case 3: unresolved !function path -- try dynamic import
    if isinstance(func, str) and "/" in func:
                try:
                                return _import_fun_from_str(func)
except (ImportError, AttributeError, ValueError) as exc:
            eval_logger.debug(
                                "Failed to dynamically import filter from path '%s': %s",
                                func,
                                exc,
            )

    raise KeyError(
                f"Filter '{func}' could not be resolved. It is not a registered "
                f"filter name, a callable, or a valid file path to a filter class."
    )


__all__ = [
        "custom",
        "extraction",
        "selection",
        "transformation",
        "build_filter_ensemble",
]
