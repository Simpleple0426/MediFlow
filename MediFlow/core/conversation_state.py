"""Only patient scoping survives between turns; clinical details stay in dialogue."""


def merge_state(previous, entities, intent_group, message, request_id):
    # Keep the signature for callers; discard legacy structured clinical slots.
    previous = previous if isinstance(previous, dict) else {}
    result = {"domain": "healthcare"}
    if isinstance(previous.get("memory_scope"), dict):
        result["memory_scope"] = dict(previous["memory_scope"])
    return result
