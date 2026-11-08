"""JSON completion, validation, and cache for the two judge providers."""

import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv


class ProviderError(RuntimeError):
    """The configured model could not return a completion."""


def configuration():
    load_dotenv(Path(__file__).with_name(".env"))
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    model = os.getenv("LLM_MODEL", "").strip()
    if provider not in ("mistral", "ollama") or not model:
        raise ProviderError("Set LLM_PROVIDER (mistral or ollama) and LLM_MODEL in .env.")
    return provider, model


def request_completion(provider, model, prompt):
    headers = {"Content-Type": "application/json"}
    body = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    if provider == "mistral":
        key = os.getenv("MISTRAL_API_KEY", "").strip()
        if not key:
            raise ProviderError("Set MISTRAL_API_KEY in .env.")
        endpoint = "https://api.mistral.ai/v1/chat/completions"
        headers["Authorization"] = f"Bearer {key}"
        body.update(response_format={"type": "json_object"}, temperature=0)
    else:
        endpoint = "http://127.0.0.1:11434/api/chat"
        body.update(format="json", stream=False, options={"temperature": 0, "num_ctx": 8192})

    request = Request(endpoint, data=json.dumps(body).encode("utf-8"), headers=headers)
    try:
        with urlopen(request, timeout=180) as response:
            result = json.load(response)
        if provider == "mistral":
            return result["choices"][0]["message"]["content"]
        return result["message"]["content"]
    except HTTPError as error:
        raise ProviderError(
            f"{provider} returned HTTP {error.code}. Check API access, quota, and model name."
        ) from error
    except (URLError, TimeoutError) as error:
        raise ProviderError(f"Could not reach {provider}: {error}") from error
    except (ValueError, KeyError, IndexError, TypeError) as error:
        raise ProviderError(f"Unexpected response from {provider}: {error}") from error


def complete(prompt, validate=None, log_attempt=None):
    """Return a JSON object, retrying once after a JSON or validation error."""
    provider, model = configuration()
    key_data = json.dumps([provider, model, prompt], ensure_ascii=False).encode("utf-8")
    path = Path("cache") / "judge" / (hashlib.sha256(key_data).hexdigest() + ".json")

    def parse(raw):
        result = json.loads(raw)
        if not isinstance(result, dict):
            raise ValueError("Expected a JSON object.")
        if validate:
            validate(result)
        return result

    if path.exists():
        try:
            raw = path.read_text(encoding="utf-8")
            result = parse(raw)
        except ValueError:
            pass
        else:
            if log_attempt:
                log_attempt("cache", raw, None)
            return result

    request_prompt = prompt
    for attempt in (1, 2):
        raw = request_completion(provider, model, request_prompt)
        try:
            result = parse(raw)
        except (ValueError, TypeError) as error:
            if log_attempt:
                log_attempt(attempt, raw, str(error))
            if attempt == 2:
                raise ValueError(f"Invalid judge response after two attempts: {error}") from error
            request_prompt = f"{prompt}\nYour previous response was invalid: {error}\nReturn a corrected JSON object."
            continue

        if log_attempt:
            log_attempt(attempt, raw, None)
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as temporary:
            temporary.write(raw)
        Path(temporary.name).replace(path)
        return result
