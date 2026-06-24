"""
Minimal Ollama bridge -- mirrors your local_llm_bridge.

Calls the local Ollama HTTP API (/api/generate) with a stop sequence so the
model emits one Thought or one Action line at a time, exactly like the
Graph-CoT reference implementation (which used ``stop="\\n"``).
"""
import json
import requests
import config


class OllamaBridge:
    def __init__(self, model=None):
        self.model = model or config.OLLAMA_MODEL
        self.url = config.OLLAMA_URL.rstrip("/") + "/api/generate"

    def generate(self, prompt, stop=("\n",), temperature=None, max_tokens=None):
        """Single-shot completion. Returns the generated string (stripped)."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": config.TEMPERATURE if temperature is None else temperature,
                "num_ctx": config.NUM_CTX,
                "num_predict": max_tokens or config.MAX_NEW_TOKENS,
                "stop": list(stop),
                "seed": config.SEED,
            },
        }
        try:
            r = requests.post(self.url, json=payload, timeout=config.REQUEST_TIMEOUT)
            r.raise_for_status()
            data = r.json()
            return (data.get("response") or "").strip()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama call failed ({self.model}): {e}")

    def health_check(self):
        try:
            r = requests.get(config.OLLAMA_URL.rstrip("/") + "/api/tags", timeout=10)
            tags = [m["name"] for m in r.json().get("models", [])]
            ok = any(self.model.split(":")[0] in t for t in tags)
            return ok, tags
        except Exception as e:
            return False, [f"(could not reach Ollama: {e})"]


if __name__ == "__main__":
    b = OllamaBridge()
    ok, tags = b.health_check()
    print(f"model={b.model}  available={ok}")
    print("installed:", tags)
    if ok:
        print("test:", b.generate("Reply with the single word OK.", stop=("\n",)))
