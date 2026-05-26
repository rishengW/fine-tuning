"""Structured exception classes used by the schema validation layer.

The design's *Error Handling* table assigns dedicated error categories to each
component (``EmptyCandidateList``, ``ConfigLoadError``, ``LoRAConfigInvalid``,
``DecodingParamsInvalid`` and others). All of those errors share a single
shape requirement: when they are raised because of a schema/field violation
they MUST surface the offending field name(s) so the operator can locate the
problem in their configuration file (Requirements 2.3, 4.9, 7.7).

This module provides:

* :class:`MathLoRAError` -- common ancestor for every error this package
  raises, so callers can do a single ``except MathLoRAError`` at top-level
  entry points.
* :class:`SchemaValidationError` -- the base for field-level validation
  errors. It accepts a ``pydantic.ValidationError`` and exposes the offending
  field paths via :attr:`SchemaValidationError.fields`.
* :class:`LoRAConfigInvalid`, :class:`ConfigLoadError`,
  :class:`DecodingParamsInvalid` -- the three error categories explicitly
  named by task 1.2 of the implementation plan and by the *Error Handling*
  table in the design document.

The error subclasses are deliberately thin: they carry the same payload as
:class:`SchemaValidationError`, plus a stable category name, and they leave
component-level fields (e.g. checkpoint path for ``CheckpointInvalid``) to be
added by their producing components in later tasks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pydantic import ValidationError


class MathLoRAError(Exception):
    """Common ancestor for every error raised by this package.

    Components are encouraged to derive their domain-specific errors from
    this class (directly or indirectly) so that callers at process entry
    points can install a single ``except MathLoRAError`` handler.
    """


# ---------------------------------------------------------------------------
# Schema validation error base
# ---------------------------------------------------------------------------


class SchemaValidationError(MathLoRAError):
    """Base class for errors raised when a pydantic schema rejects an input.

    The class is intentionally usable both as a wrapper around a
    :class:`pydantic.ValidationError` (the common case for the generated
    ``parse`` classmethods) and as a stand-alone error constructed with an
    explicit field list and message (used by callers that detect a violation
    outside of pydantic, e.g. ``LoRA_Trainer`` checking that every entry in
    ``target_modules`` matches a real base-model module name).

    Attributes:
        fields: Ordered tuple of dotted field paths that failed validation.
            Empty only when the error is constructed from a non-field
            condition (rare; prefer naming a field whenever possible because
            the design explicitly requires it).
        details: Per-field structured details, when available. The shape
            mirrors :meth:`pydantic.ValidationError.errors` so existing
            tooling (logs, error reporters) can consume it directly.
        cause: The originating :class:`pydantic.ValidationError`, if the
            error was constructed via :meth:`from_validation_error`.
    """

    #: Stable string used by reporting code and tests to identify the error
    #: category. Subclasses override this with the names declared in the
    #: design's *Error Handling* table.
    category: str = "SchemaValidationError"

    def __init__(
        self,
        message: str,
        *,
        fields: "tuple[str, ...] | list[str] | None" = None,
        details: "tuple[dict[str, Any], ...] | list[dict[str, Any]] | None" = None,
        cause: "ValidationError | None" = None,
    ) -> None:
        self.fields: tuple[str, ...] = tuple(fields) if fields else ()
        self.details: tuple[dict[str, Any], ...] = (
            tuple(details) if details else ()
        )
        self.cause: ValidationError | None = cause
        super().__init__(self._format_message(message))

    # ---- alternate constructors -------------------------------------------------

    @classmethod
    def from_validation_error(
        cls,
        exc: "ValidationError",
        *,
        message: str | None = None,
    ) -> "SchemaValidationError":
        """Wrap a :class:`pydantic.ValidationError` preserving field paths.

        The pydantic error's ``loc`` tuples are flattened into dotted paths
        (``("foo", 0, "bar")`` becomes ``"foo.0.bar"``) so they read
        naturally in error messages and remain stable for log parsing.
        """

        details = list(exc.errors())
        fields = tuple(_loc_to_dotted_path(d.get("loc", ())) for d in details)
        msg = message or _build_default_message(cls.category, fields)
        return cls(msg, fields=fields, details=details, cause=exc)

    # ---- internals --------------------------------------------------------------

    def _format_message(self, message: str) -> str:
        """Return ``message`` with the field list appended when known.

        Keeping the formatted form on the exception args makes the field
        list visible in default Python tracebacks without forcing callers
        to install a custom logger.
        """

        if not self.fields:
            return message
        return f"{message} [fields: {', '.join(self.fields)}]"

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"{type(self).__name__}(category={self.category!r}, "
            f"fields={self.fields!r})"
        )


# ---------------------------------------------------------------------------
# Concrete schema errors named in task 1.2 / Error Handling table
# ---------------------------------------------------------------------------


class LoRAConfigInvalid(SchemaValidationError):
    """LoRA hyperparameters violated their declared ranges or vocabularies.

    Triggered by ``LoRA_Trainer`` (Requirement 4.9) when ``r`` is outside
    ``[4, 128]``, ``dropout`` is outside ``[0.0, 1.0]``, ``bias`` is not in
    ``{"none", "all", "lora_only"}``, or any ``target_modules`` entry does
    not match a real base-model module name.

    The class is also raised by :meth:`math_lora.types.models.LoRAConfig.parse`
    so that an upstream config loader sees the same exception type whether
    the rejection happens at parse time or at module-resolution time later
    in the pipeline.
    """

    category: str = "LoRAConfigInvalid"


class ConfigLoadError(SchemaValidationError):
    """A YAML/JSON profile failed to load or was missing required fields.

    Triggered by ``Hardware_Budget_Planner`` (Requirement 2.3) when the
    ``Hardware_Profile`` or ``Budget_Profile`` config file is malformed or
    is missing any field declared in Requirement 2.1 / 2.2. The intent is
    that the surfaced message identifies *both* the file and the field; the
    file path is recorded on :attr:`source_path` while the field paths live
    on :attr:`SchemaValidationError.fields`.
    """

    category: str = "ConfigLoadError"

    def __init__(
        self,
        message: str,
        *,
        source_path: str | None = None,
        fields: "tuple[str, ...] | list[str] | None" = None,
        details: "tuple[dict[str, Any], ...] | list[dict[str, Any]] | None" = None,
        cause: "ValidationError | None" = None,
    ) -> None:
        self.source_path: str | None = source_path
        suffix = f" (source: {source_path})" if source_path else ""
        super().__init__(
            f"{message}{suffix}",
            fields=fields,
            details=details,
            cause=cause,
        )

    @classmethod
    def from_validation_error(
        cls,
        exc: "ValidationError",
        *,
        message: str | None = None,
        source_path: str | None = None,
    ) -> "ConfigLoadError":
        """Wrap a :class:`pydantic.ValidationError`, recording the source file."""

        details = list(exc.errors())
        fields = tuple(_loc_to_dotted_path(d.get("loc", ())) for d in details)
        msg = message or _build_default_message(cls.category, fields)
        return cls(
            msg,
            source_path=source_path,
            fields=fields,
            details=details,
            cause=exc,
        )


class DecodingParamsInvalid(SchemaValidationError):
    """Decoding parameters were out of their declared ranges.

    Triggered by ``Inference_Server`` (Requirement 7.7) and ``Evaluator``
    (Requirement 6.3, via the same validator) when ``temperature`` is
    outside ``[0.0, 2.0]``, ``top_p`` is outside ``[0.0, 1.0]``, ``top_k``
    is negative, or ``max_new_tokens`` is non-positive.

    The error names exactly the mutated field per Property 35 in the design
    document.
    """

    category: str = "DecodingParamsInvalid"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _loc_to_dotted_path(loc: "tuple[Any, ...] | list[Any]") -> str:
    """Convert a pydantic ``loc`` tuple into a stable dotted path.

    Pydantic uses integers in ``loc`` to represent list indices; we render
    them with the same dotted notation so that downstream callers can use
    a single string-comparison form.
    """

    return ".".join(str(part) for part in loc) if loc else ""


def _build_default_message(category: str, fields: tuple[str, ...]) -> str:
    """Build a default error message that names the offending fields."""

    if not fields:
        return f"{category}: schema validation failed"
    if len(fields) == 1:
        return f"{category}: invalid value for field '{fields[0]}'"
    joined = ", ".join(f"'{f}'" for f in fields)
    return f"{category}: invalid values for fields {joined}"


__all__ = [
    "MathLoRAError",
    "SchemaValidationError",
    "LoRAConfigInvalid",
    "ConfigLoadError",
    "DecodingParamsInvalid",
]
