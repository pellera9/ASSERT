"""SpanCollector Protocol — decouples P2M from any specific trace backend.

P2M's OTel integration depends on this Protocol, not on Phoenix.
Phoenix is one implementation. Developers can inject any backend.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

# OpenInference column contract — what P2M's extraction layer depends on.
REQUIRED_COLUMNS = frozenset({
    "context.trace_id",
    "context.span_id",
    "parent_id",
    "name",
    "start_time",
    "end_time",
    "attributes.openinference.span.kind",
    "attributes.input.value",
    "attributes.output.value",
})

TRAJECTORY_COLUMNS = frozenset({
    "attributes.llm.input_messages",
    "attributes.llm.output_messages",
    "attributes.llm.tools",
})

SESSION_COLUMNS = frozenset({
    "attributes.session.id",
})


@runtime_checkable
class SpanCollector(Protocol):
    """Minimal interface P2M depends on for trace collection.

    Any object implementing get_spans() satisfies this — no inheritance needed.
    Phoenix is one implementation. Jaeger/Datadog/file export are others.
    """

    def get_spans(
        self,
        project_name: str,
        *,
        start_time: str | None = None,
        end_time: str | None = None,
        trace_ids: list[str] | None = None,
    ) -> Any:
        """Return spans. The return type is intentionally Any to avoid
        a hard pandas dependency at the Protocol level. Implementations
        return pd.DataFrame with OpenInference column names."""
        ...

    def validate(self, spans: Any) -> list[str]:
        """Return warnings for missing/malformed columns. Empty = OK."""
        ...


class DataFrameCollector:
    """Wraps a pre-loaded DataFrame as a SpanCollector.

    Use when you already have spans from any source — Arize cloud export,
    Parquet file, Jaeger export translated to OpenInference columns.
    """

    def __init__(self, df: Any) -> None:
        self._df = df

    def get_spans(self, project_name: str | None = None, **kwargs: Any) -> Any:
        return self._df

    def validate(self, spans: Any) -> list[str]:
        warnings: list[str] = []
        if hasattr(spans, "columns"):
            missing = REQUIRED_COLUMNS - set(spans.columns)
            if missing:
                warnings.append(f"Missing required columns: {sorted(missing)}")
        return warnings


class PhoenixCollector:
    """SpanCollector backed by a local Phoenix instance.

    Phoenix is an OPTIONAL dependency — only imported when instantiated.
    Install: pip install 'p2m-policy[phoenix]'
    """

    def __init__(
        self,
        endpoint: str = "http://localhost:6006",
        *,
        project_name: str | None = None,
    ) -> None:
        try:
            import phoenix as px

            self._client = px.Client(endpoint=endpoint)
        except ImportError as e:
            raise ImportError(
                "PhoenixCollector requires arize-phoenix. "
                "Install with: pip install 'p2m-policy[phoenix]'"
            ) from e
        self._default_project = project_name

    def get_spans(
        self,
        project_name: str | None = None,
        *,
        start_time: str | None = None,
        end_time: str | None = None,
        trace_ids: list[str] | None = None,
    ) -> Any:
        import pandas as pd

        name = project_name or self._default_project
        if name is None:
            raise ValueError("project_name required")

        df: pd.DataFrame = self._client.get_spans_dataframe(
            project_name=name,
            start_time=start_time,
            end_time=end_time,
        )
        if trace_ids:
            df = df[df["context.trace_id"].isin(trace_ids)]
        return df.reset_index(drop=True)

    def validate(self, spans: Any) -> list[str]:
        warnings: list[str] = []
        if not hasattr(spans, "columns"):
            return ["spans is not a DataFrame"]
        missing = REQUIRED_COLUMNS - set(spans.columns)
        if missing:
            warnings.append(f"Missing required columns: {sorted(missing)}")

        # Check LLM spans have output messages
        kind_col = "attributes.openinference.span.kind"
        output_col = "attributes.llm.output_messages"
        if kind_col in spans.columns and output_col in spans.columns:
            llm_mask = spans[kind_col] == "LLM"
            if llm_mask.any():
                no_output = spans.loc[llm_mask, output_col].isna().sum()
                if no_output > 0:
                    warnings.append(
                        f"{no_output} LLM span(s) missing output_messages. "
                        "Trajectory evaluation will be incomplete."
                    )

        if "attributes.session.id" not in spans.columns:
            warnings.append(
                "No session.id column. Session-level evaluation requires this."
            )
        return warnings
