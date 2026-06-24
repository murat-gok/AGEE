#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
AGEE Revizyon — ToG/PoG için Lokal LLM Yama (Ollama Köprüsü)
=============================================================================
ToG ve PoG, varsayılan olarak OpenAI API'sine (GPT-3.5/GPT-4) çağrı yapar.
Bu fonksiyon, o çağrıyı LOKAL Ollama'ya (Llama-3.1-8B / Qwen-2.5-7B)
yönlendirir. RTX 3050 6GB üzerinde çalışır.

KURULUM (repo tarafında):
  1) ToG/utils.py içinde 'run_llm' fonksiyonunu bulun.
     PoG/utils.py içinde 'run_LLM' fonksiyonunu bulun.
  2) O fonksiyonun GÖVDESİNİ aşağıdaki run_llm_local ile değiştirin
     (imzayı koruyun: ToG -> run_llm(prompt, temperature, max_tokens,
      opeani_api_keys, engine); PoG benzer).
  3) openai paketini kurun (Ollama OpenAI-uyumlu uç sağlar):
       pip install openai
  4) Ollama servisini başlatın ve modeli çekin:
       ollama serve            # ayrı terminal
       ollama pull llama3.1:8b
       ollama pull qwen2.5:7b
  5) Düşük VRAM için context'i 2048'e sabitleyin (Modelfile veya options).

ÖNEMLİ (6GB VRAM):
  - num_ctx=2048 kullanın. Daha büyük context VRAM'i taşırır,
    katmanlar CPU'ya offload olur ve hız 5-10x düşer.
  - Çalışırken AYRI terminalde izleyin:  watch -n1 nvidia-smi
    GPU-Util düşük + RAM yüksekse offload var demektir.
=============================================================================
"""
import time
from openai import OpenAI

# Ollama'nın OpenAI-uyumlu yerel ucu
_CLIENT = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # Ollama anahtarı yok sayar; boş bırakılamaz
)

# ToG/PoG'daki 'engine' argümanını Ollama model adına eşleyen tablo.
# Repo, engine olarak 'gpt-3.5-turbo' gönderse bile lokal modele çevrilir.
_MODEL_MAP = {
    "gpt-3.5-turbo": "qwen2.5:7b",   # varsayılan lokal backbone
    "gpt-4": "qwen2.5:7b",
    "llama": "llama3.1:8b",
    "llama3.1": "llama3.1:8b",
    "qwen": "qwen2.5:7b",
    "qwen2.5": "qwen2.5:7b",
}

# Hangi backbone'la koştuğunuzu BURADAN sabitleyin (deney başına değiştirin):
ACTIVE_BACKBONE = "llama3.1:8b"   # veya "llama3.1:8b"


def _resolve_model(engine: str) -> str:
    if engine in _MODEL_MAP:
        # engine eşlemesi yerine aktif backbone'u tercih et (deney kontrolü)
        return ACTIVE_BACKBONE
    return ACTIVE_BACKBONE


def run_llm_local(prompt, temperature=0.0, max_tokens=256,
                  opeani_api_keys=None, engine="gpt-3.5-turbo"):
    """ToG/PoG run_llm/run_LLM yerine geçen lokal sürüm.
    İmza ToG ile uyumludur; PoG çağrısı da aynı argümanları verir."""
    model = _resolve_model(engine)
    messages = [
        {"role": "system",
         "content": "You are an AI assistant that helps reason over a "
                    "knowledge graph. Follow the requested output format "
                    "exactly."},
        {"role": "user", "content": prompt},
    ]
    # Küçük modeller bazen boş/bozuk döndürür -> 3 deneme
    last_err = None
    for attempt in range(3):
        try:
            resp = _CLIENT.chat.completions.create(
                model=model,
                messages=messages,
                temperature=float(temperature),
                max_tokens=int(max_tokens),
                # Ollama'ya context sınırını ilet (6GB için kritik):
                extra_body={"options": {"num_ctx": 2048}},
            )
            content = resp.choices[0].message.content
            if content and content.strip():
                return content
        except Exception as e:           # noqa
            last_err = e
            time.sleep(2 * (attempt + 1))
    # 3 deneme de başarısızsa boş döndür (repo'nun parse fallback'i devreye girer)
    print(f"  [uyarı] LLM çağrısı başarısız (model={model}): {last_err}")
    return ""


# ToG kodu büyük olasılıkla 'run_llm' adını import eder; alias bırakıyoruz:
run_llm = run_llm_local
run_LLM = run_llm_local


if __name__ == "__main__":
    print(f"Lokal LLM köprüsü testi (backbone={ACTIVE_BACKBONE})")
    print("Ollama servisinin açık olması gerekir (ollama serve).")
    try:
        out = run_llm_local("Reply with exactly one word: OK", max_tokens=10)
        print("  Yanıt:", repr(out))
        print("  [OK]" if out else "  [UYARI] boş yanıt — servisi/modeli kontrol edin")
    except Exception as e:
        print("  [HATA]", e)
