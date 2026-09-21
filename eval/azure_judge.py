"""Azure gpt-5-nano as a DeepEval judge. DeepEval's built-in Azure model sends temperature=0 and
omits reasoning_effort, both fatal to gpt-5-nano (rejects the first, returns empty on the second).
"""
import random
import re
import time
from pathlib import Path

from deepeval.models.base_model import DeepEvalBaseLLM
from openai import (APIConnectionError, APITimeoutError, AsyncAzureOpenAI,
                    AzureOpenAI, RateLimitError)
from pydantic import BaseModel

_CRED_DIR = Path(__file__).resolve().parents[2]


def _load_credentials() -> dict:
    """Prefers the GraphRAG .env, falls back to azure*.txt."""
    env = Path(__file__).resolve().parents[1] / "graphrag" / ".env"
    if env.exists():
        vals = dict(
            line.split("=", 1)
            for line in env.read_text().splitlines()
            if "=" in line and not line.startswith("#")
        )
        return {
            "api_key": vals["GRAPHRAG_API_KEY"],
            "endpoint": vals["GRAPHRAG_API_BASE"],
            "api_version": vals["GRAPHRAG_API_VERSION"],
            "deployment": vals["GRAPHRAG_DEPLOYMENT"],
        }

    uri = (_CRED_DIR / "azure_chat_uri.txt").read_text().strip()
    m = re.match(r"(https://[^/]+)/openai/deployments/([^/]+)/", uri)
    return {
        "api_key": (_CRED_DIR / "azure_chat_key.txt").read_text().strip(),
        "endpoint": m.group(1),
        "deployment": m.group(2),
        "api_version": re.search(r"api-version=([^&\s]+)", uri).group(1),
    }


class AzureGPT5Nano(DeepEvalBaseLLM):
    def __init__(self):
        self.cred = _load_credentials()
        self._sync = AzureOpenAI(
            api_key=self.cred["api_key"],
            azure_endpoint=self.cred["endpoint"],
            api_version=self.cred["api_version"],
        )
        self._async = AsyncAzureOpenAI(
            api_key=self.cred["api_key"],
            azure_endpoint=self.cred["endpoint"],
            api_version=self.cred["api_version"],
        )
        super().__init__(self.cred["deployment"])

    def load_model(self):
        return self._sync

    def get_model_name(self) -> str:
        return f"azure/{self.cred['deployment']}"

    def _kwargs(self, prompt: str, schema) -> dict:
        kw = {
            "model": self.cred["deployment"],
            "messages": [{"role": "user", "content": prompt}],
            # Mandatory -- without it the model burns the cap on reasoning and returns nothing.
            "reasoning_effort": "minimal",
            # gpt-5 accepts only default temperature; budget is completion tokens, not max_tokens.
            "max_completion_tokens": 4000,
        }
        if schema is not None:
            kw["response_format"] = schema
        return kw

    @staticmethod
    def _unwrap(resp, schema):
        choice = resp.choices[0]
        if schema is not None:
            parsed = getattr(choice.message, "parsed", None)
            if parsed is not None:
                return parsed
        content = choice.message.content or ""
        if not content.strip():
            raise RuntimeError(
                f"empty judge response (finish_reason={choice.finish_reason}); "
                "reasoning_effort is probably not being applied"
            )
        return schema.model_validate_json(content) if schema else content

    def _retry(self, call, *, attempts: int = 6):
        """Judging runs many threads against the same deployment, often while an
        indexing or query run is using it too, so 429s are routine. Without this
        they surfaced as dropped samples and a `None` metric — which reads in the
        results exactly like a genuine scoring failure."""
        delay = 4.0
        for i in range(attempts):
            try:
                return call()
            except RateLimitError:
                if i == attempts - 1:
                    raise
                time.sleep(delay + random.uniform(0, delay / 2))
                delay = min(delay * 2, 90)
            except (APIConnectionError, APITimeoutError):
                if i == attempts - 1:
                    raise
                time.sleep(delay)
        return None

    def generate(self, prompt: str, schema: type[BaseModel] | None = None):
        kw = self._kwargs(prompt, schema)
        if schema is not None:
            return self._unwrap(
                self._retry(lambda: self._sync.beta.chat.completions.parse(**kw)), schema)
        return self._unwrap(
            self._retry(lambda: self._sync.chat.completions.create(**kw)), None)

    async def a_generate(self, prompt: str, schema: type[BaseModel] | None = None):
        # Sync path carries the retry logic (async_mode=False on every metric); this delegates.
        return self.generate(prompt, schema)


if __name__ == "__main__":
    j = AzureGPT5Nano()
    print("model:", j.get_model_name())
    print("plain:", repr(j.generate("Reply with exactly: JUDGE-READY")))

    class Score(BaseModel):
        score: float
        reason: str

    print("schema:", j.generate(
        "Rate how well 'Paris' answers 'What is the capital of France?' "
        "Return a score between 0 and 1 and a one-line reason.", Score))
