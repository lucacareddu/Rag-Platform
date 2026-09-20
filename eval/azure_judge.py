"""Azure gpt-5-nano as a DeepEval judge model.

DeepEval's built-in Azure model cannot be used here. It sends `temperature=0`
and omits `reasoning_effort`, and gpt-5-nano rejects the first and needs the
second: at its default reasoning effort it spends the whole completion budget
on hidden reasoning tokens and returns an empty string. A judge that silently
returns "" scores every test case as a parse failure, which is indistinguishable
from a genuinely bad answer — so this wrapper is what keeps the evaluation
honest rather than merely working.

Credentials are read from the azure*.txt files in the repo parent, the same
source the GraphRAG workspace uses, so there is one place to rotate them.
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
    """Prefers the GraphRAG workspace .env, falls back to the raw azure*.txt."""
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
            # Mandatory. Without it the model emits reasoning tokens until it
            # hits the cap and returns nothing at all.
            "reasoning_effort": "minimal",
            # gpt-5 accepts only the default temperature; and the budget is
            # counted as completion tokens, not `max_tokens`.
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
        # DeepEval is driven synchronously here (async_mode=False on every
        # metric), so the sync path carries the retry logic and this delegates.
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
