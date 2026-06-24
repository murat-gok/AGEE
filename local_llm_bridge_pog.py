#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
AGEE Revizyon — PoG için Lokal LLM Köprüsü (Ollama)
=============================================================================
PoG'un run_llm'i ToG'dan FARKLI:
  - İmza: run_llm(prompt, temp, max_tokens, keys, engine, print_in, print_out)
  - Dönüş: (result, token_num)  -- tuple!  token_num = {total,input,output}
  - 'gpt' in engine kontrolü yapıyor -> lokal modelde bunu baypas etmeliyiz.

Bu köprü PoG'un run_llm'ini birebir uyumlu imzayla değiştirir.
=============================================================================
"""
import time
from openai import OpenAI

_CLIENT = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

# Hangi backbone (deney başına değiştir):
ACTIVE_BACKBONE = "qwen2.5:7b"   # veya "llama3.1:8b"

# ANSI renk kodları (PoG orijinalindeki print'ler için zararsız placeholder)
color_green = ""; color_yellow = ""; color_end = ""


def run_llm(prompt, temperature, max_tokens, opeani_api_keys,
            engine="gpt-3.5-turbo", print_in=True, print_out=True):
    """PoG run_llm yerine geçen lokal sürüm. (result, token_num) döndürür."""
    model = ACTIVE_BACKBONE
    messages = [
        {"role": "system",
         "content": "You are an AI assistant that helps people find "
                    "information over a knowledge graph. Follow the requested "
                    "output format exactly."},
        {"role": "user", "content": prompt},
    ]
    last_err = None
    for attempt in range(3):
        try:
            resp = _CLIENT.chat.completions.create(
                model=model, messages=messages,
                temperature=float(temperature),
                max_tokens=int(max_tokens),
                extra_body={"options": {"num_ctx": 2048}},
            )
            content = resp.choices[0].message.content or ""
            usage = resp.usage
            token_num = {
                "total": getattr(usage, "total_tokens", 0) or 0,
                "input": getattr(usage, "prompt_tokens", 0) or 0,
                "output": getattr(usage, "completion_tokens", 0) or 0,
            }
            if content.strip():
                return content, token_num
        except Exception as e:                       # noqa
            last_err = e
            time.sleep(2 * (attempt + 1))
    print(f"  [uyarı] PoG LLM çağrısı başarısız (model={model}): {last_err}")
    return "", {"total": 0, "input": 0, "output": 0}


if __name__ == "__main__":
    print(f"PoG lokal köprü testi (backbone={ACTIVE_BACKBONE})")
    out, tok = run_llm("Reply with exactly one word: OK", 0.0, 10, "",
                       "qwen2.5:7b", False, False)
    print("  Yanıt:", repr(out), "| token:", tok)
    print("  [OK]" if out else "  [UYARI] boş yanıt")
