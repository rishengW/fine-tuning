"""Profile config loaders for the ``Hardware_Budget_Planner`` (Requirement 2).

This module is the entry point through which ``Hardware_Profile`` and
``Budget_Profile`` configuration files (YAML or JSON) reach the rest of the
pipeline. It is the *file* half of the contract declared by Requirement 2.3:
when loading fails, the surfaced :class:`~math_lora.types.ConfigLoadError`
must identify *both* the file and the offending field.

Two pure functions are exposed:

* :func:`load_hardware_profile` -- returns a validated
  :class:`~math_lora.types.HardwareProfile` (Requirement 2.1).
* :func:`load_budget_profile` -- returns a validated
  :class:`~math_lora.types.BudgetProfile` (Requirement 2.2).

Both functions share an identical loading skeleton; the only differences are
the target schema and the error-message wording. They are deliberately
implemented on top of a single private helper (:func:`_load_profile`) so the
behavior contract -- file-not-found, parse error, schema error, unsupported
extension -- is captured exactly once.

Error-handling contract
-----------------------

Every failure path raises :class:`~math_lora.types.ConfigLoadError` with a
populated ``source_path`` so the operator can locate the offending file.

* **Missing file.** ``ConfigLoadError("profile file not found", source_path=p)``.
* **Parse failure.** ``ConfigLoadError("profile file failed to parse: <cause>",
  source_path=p)`` -- the underlying YAML/JSON exception message is appended
  so the operator sees the line/column without needing the chained traceback.
* **Schema failure.** Pydantic's ``ValidationError`` is rewrapped via
  :meth:`ConfigLoadError.from_validation_error` which preserves the
  per-field ``loc`` paths in :attr:`ConfigLoadError.fields`.
* **Unsupported extension.** ``ConfigLoadError("unsupported config file
  extension '<ext>'", source_path=p)`` -- the extension is included in the
  message so the operator sees exactly which suffix the loader rejected.
* **Non-mapping top level.** A YAML/JSON file whose root is not a mapping
  (e.g. a list or a bare scalar) raises a ``ConfigLoadError`` rather than
  producing a confusing pydantic error, since pydantic would only report
  ``input should be a valid dictionary``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Final

import yaml
from pydantic import ValidationError

from math_lora.types.errors import ConfigLoadError
from math_lora.types.models import BudgetProfile, HardwareProfile


# Tuple of accepted file extensions, lowercased and including the leading dot.
# YAML's two conventional extensions (``.yaml`` and ``.yml``) are both
# accepted because real-world configs in the wild use either one.
_YAML_EXTENSIONS: Final[tuple[str, ...]] = (".yaml", ".yml")
_JSON_EXTENSIONS: Final[tuple[str, ...]] = (".json",)
_SUPPORTED_EXTENSIONS: Final[tuple[str, ...]] = _YAML_EXTENSIONS + _JSON_EXTENSIONS


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def load_hardware_profile(path: "str | os.PathLike[str]") -> HardwareProfile:
    """Load a :class:`HardwareProfile` from a YAML or JSON file.

    Args:
        path: Filesystem path to a ``.yaml``, ``.yml``, or ``.json`` file
            whose top-level mapping declares every field required by
            Requirement 2.1 (``gpu_model``, ``gpu_count``,
            ``vram_per_gpu_gb``, ``system_ram_gb``, ``disk_space_gb``,
            ``accelerator_family``, ``deployment``).

    Returns:
        A frozen :class:`HardwareProfile` instance with every field
        populated and validated.

    Raises:
        ConfigLoadError: If the file does not exist, fails to parse, has an
            unsupported extension, or fails schema validation. The error's
            :attr:`~ConfigLoadError.source_path` always names the file, and
            for schema failures :attr:`~ConfigLoadError.fields` lists the
            offending field paths.
    """

    return _load_profile(path, HardwareProfile)


def load_budget_profile(path: "str | os.PathLike[str]") -> BudgetProfile:
    """Load a :class:`BudgetProfile` from a YAML or JSON file.

    Args:
        path: Filesystem path to a ``.yaml``, ``.yml``, or ``.json`` file
            whose top-level mapping declares every field required by
            Requirement 2.2 (``max_cost``, ``currency``,
            ``max_wallclock_hours``, ``cost_rate_per_gpu_hour``).

    Returns:
        A frozen :class:`BudgetProfile` instance with every field populated
        and validated.

    Raises:
        ConfigLoadError: Same contract as :func:`load_hardware_profile`.
    """

    return _load_profile(path, BudgetProfile)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _load_profile(
    path: "str | os.PathLike[str]",
    schema: "type[HardwareProfile] | type[BudgetProfile]",
) -> "HardwareProfile | BudgetProfile":
    """Shared loader skeleton parameterized over the target schema.

    The function is split into three stages -- read, parse, validate -- so
    that each error category surfaces a distinct ``ConfigLoadError`` with a
    message tailored to the failure cause. Splitting also keeps the public
    functions thin wrappers, which is what task 3.1 requires.

    Note on type narrowing: the return type is the union of the two schema
    types because mypy cannot specialize a ``type[T]`` parameter in this
    branch position; callers see the precise return type via the
    schema-specific public functions :func:`load_hardware_profile` and
    :func:`load_budget_profile`.
    """

    # Normalize once; ``Path`` accepts ``str`` and ``os.PathLike`` uniformly,
    # and ``source_path_str`` is what we attach to every raised error.
    file_path = Path(os.fspath(path))
    source_path_str = str(file_path)

    # --- Stage 1: existence check -----------------------------------------
    # ``Path.is_file`` returns False both when the path is missing and when
    # it points at a directory; the second case is also a load failure for
    # our purposes (we cannot read a directory as YAML/JSON), so a single
    # branch handles both.
    if not file_path.is_file():
        raise ConfigLoadError(
            "profile file not found",
            source_path=source_path_str,
        )

    # --- Stage 2: extension dispatch --------------------------------------
    # ``suffix`` returns the empty string when there is no extension. We
    # lowercase to normalize ``.YAML`` / ``.Json`` etc. before comparison.
    suffix = file_path.suffix.lower()
    if suffix not in _SUPPORTED_EXTENSIONS:
        raise ConfigLoadError(
            f"unsupported config file extension '{suffix}'",
            source_path=source_path_str,
        )

    # --- Stage 3: read + parse --------------------------------------------
    # We read the file as text so the YAML and JSON branches receive a
    # ``str``. Reading uses UTF-8 explicitly so that configs authored on
    # systems with non-UTF-8 default encodings still parse identically
    # everywhere -- determinism matters because dataset content hashes and
    # Run_Manifest reproducibility (Requirement 8.10) depend on it.
    try:
        text = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        # Defensive branch: ``is_file()`` returned True moments ago but a
        # concurrent unlink/rename could still cause the read to fail. We
        # surface it with the same shape as a missing-file error so callers
        # do not need a separate handler.
        raise ConfigLoadError(
            f"profile file failed to read: {exc}",
            source_path=source_path_str,
        ) from exc

    raw_data = _parse_text(text, suffix=suffix, source_path=source_path_str)

    # The schema models require a mapping at the top level; a YAML file
    # whose root is a list (``- foo``) or a bare scalar would otherwise
    # produce a less-helpful pydantic message. Surface it ourselves so the
    # operator sees the actual cause.
    if not isinstance(raw_data, dict):
        raise ConfigLoadError(
            "profile file must contain a top-level mapping, "
            f"got {type(raw_data).__name__}",
            source_path=source_path_str,
        )

    # --- Stage 4: schema validation ---------------------------------------
    # We bypass the schema's ``parse`` classmethod (which would also raise
    # ``ConfigLoadError`` because both profiles set
    # ``_schema_error_cls = ConfigLoadError``) and call ``model_validate``
    # directly so we can inject the ``source_path`` on the error. The
    # ``parse`` classmethod cannot do that because it has no knowledge of
    # the file the data came from.
    try:
        return schema.model_validate(raw_data)
    except ValidationError as exc:
        raise ConfigLoadError.from_validation_error(
            exc,
            source_path=source_path_str,
        ) from exc


def _parse_text(text: str, *, suffix: str, source_path: str) -> Any:
    """Parse ``text`` as YAML or JSON based on the file extension.

    The branch is chosen by the (already-validated) ``suffix`` rather than
    by content sniffing so that operator intent is honored: a file named
    ``hardware.json`` is always parsed as JSON, never as YAML, even though
    a strict-JSON document is also valid YAML.
    """

    # ``yaml.safe_load`` is used (not ``yaml.load``) because we never want
    # to deserialize arbitrary Python objects from a config file -- that
    # would be a remote-code-execution vector (yaml.load on untrusted input
    # is a known PyYAML footgun).
    if suffix in _YAML_EXTENSIONS:
        try:
            return yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ConfigLoadError(
                f"profile file failed to parse: {exc}",
                source_path=source_path,
            ) from exc

    # ``suffix`` was already validated against ``_SUPPORTED_EXTENSIONS`` by
    # the caller, so the only remaining branch is JSON. We still phrase the
    # condition explicitly for readability rather than using ``else``.
    if suffix in _JSON_EXTENSIONS:
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ConfigLoadError(
                f"profile file failed to parse: {exc}",
                source_path=source_path,
            ) from exc

    # Defensive: if a future extension is added to ``_SUPPORTED_EXTENSIONS``
    # without a corresponding parse branch, we surface a clear error rather
    # than silently returning ``None``.
    raise ConfigLoadError(  # pragma: no cover - guard for future maintainers
        f"no parser registered for extension '{suffix}'",
        source_path=source_path,
    )


__all__ = [
    "load_hardware_profile",
    "load_budget_profile",
]
