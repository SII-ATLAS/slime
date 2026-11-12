"""Map slime's exhausted-budget reply to Ava's native agent termination."""

from ava_core.core.client import ContextLengthExceededError, OpenAIClient
from ava_core.core.model import kind


@kind("ava_budgeted_openai")
class BudgetedOpenAIClient(OpenAIClient):
    def __init__(self, *args, context_limit: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.context_limit = context_limit

    async def step(self, trace, **kwargs):
        result = await super().step(trace, **kwargs)
        metadata = result.last_assistant().metadata
        if (
            metadata.get("finish_reason") == "length"
            and metadata.get("completion_tokens") == 0
            and metadata.get("prompt_tokens", 0) >= self.context_limit
        ):
            raise ContextLengthExceededError(metadata["prompt_tokens"], self.context_limit)
        return result
