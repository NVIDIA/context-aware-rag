---
name: vlm-structured-prompts
description: >-
  Configure aggregation_prompt and description_merge_prompt for VLM structured
  summarization, including LVS_ENABLE_LLM_MERGING gating for description merge.
  Use when customizing VLM structured summary prompts, enabling LLM event
  description merging, or changing LVS summarization prompt params.
---

# VLM Structured Summarization Prompts

Use this skill when changing prompts for `vlm_structured_summarization` or
`vlm_structured_summarization_online` in `vlm_structured_base.py`.

## Parameters

| Param | Purpose | When applied |
| --- | --- | --- |
| `aggregation_prompt` | System prompt for final narrative aggregation | Always (falls back to built-in default when unset/empty) |
| `description_merge_prompt` | System prompt for merging adjacent same-type descriptions | Only when LLM merging is enabled |
| `enable_llm_merging` | Config flag to enable LLM description merging | Also enabled by env `LVS_ENABLE_LLM_MERGING` |

## Enable LLM description merging

`description_merge_prompt` is only used when LLM merging is enabled via either:

- function param: `enable_llm_merging: true`
- environment variable: `LVS_ENABLE_LLM_MERGING=true` (also `1` / `yes`)

When LLM merging is disabled, adjacent same-type descriptions are concatenated with ` | ` and `description_merge_prompt` is ignored (the merge pipeline is not built).

## Config example

```yaml
functions:
  summarization:
    type: vlm_structured_summarization
    params:
      enable_llm_merging: !ENV ${LVS_ENABLE_LLM_MERGING:false}
      aggregation_prompt: |
        Write a concise chronological summary of the events.
      description_merge_prompt: |
        Merge the provided descriptions into one coherent description.
    tools:
      llm: summarization_llm
      db: elasticsearch_db
```

## Implementation notes

- Code lives in `src/vss_ctx_rag/functions/summarization/vlm_structured_base.py`.
- Defaults: `DEFAULT_AGGREGATION_PROMPT` and `DEFAULT_DESCRIPTION_MERGE_PROMPT`.
- User message templates stay fixed:
  - aggregation: events via `{input}`
  - description merge: `{event_type}` and `{descriptions}`
- Shared params schema: `VlmStructuredParamsBase`.

## Docs to keep in sync

When changing these params, update:

- `README.md` (VLM Structured Summarization Prompts)
- `docs/source/overview/configuration.md` (VLM Structured Summarization section)
- `docs/source/overview/features.md` (Summarization bullet)
- `config/config.yaml` / `config/config_lvs.yaml` comments or examples
