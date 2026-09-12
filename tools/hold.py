"""A no-op action: do nothing this cycle.

Lets an agent pass without being forced into a real trade or task when
nothing looks worth doing.
"""


def hold() -> str:
    return "held position, no action taken"
