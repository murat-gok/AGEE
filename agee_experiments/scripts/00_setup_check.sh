#!/usr/bin/env bash
# =============================================================================
# AGEE Revizyon — Adım 0: Kurulum ve Ortam Doğrulama
# Hedef makine: NVIDIA RTX 3050 6GB VRAM, 16GB RAM, Windows + WSL Ubuntu
# Çalıştırma:  bash 00_setup_check.sh
# =============================================================================
set -u  # tanımsız değişkende dur (set -e KOYMUYORUZ; her kontrolü görmek istiyoruz)

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[UYARI]${NC} $1"; }
err()  { echo -e "${RED}[HATA]${NC} $1"; }

echo "============================================================"
echo " AGEE Deney Ortamı Doğrulama (RTX 3050 6GB)"
echo "============================================================"

# --- 1. Python ve temel paketler --------------------------------------------
echo; echo ">>> 1. Python ortamı"
if command -v python3 &>/dev/null; then
    ok "python3: $(python3 --version 2>&1)"
else
    err "python3 bulunamadı."
fi

for pkg in networkx numpy scipy pandas datasets; do
    if python3 -c "import $pkg" 2>/dev/null; then
        ver=$(python3 -c "import $pkg; print(getattr($pkg,'__version__','?'))" 2>/dev/null)
        ok "$pkg ($ver)"
    else
        warn "$pkg yüklü değil  ->  pip install $pkg"
    fi
done
# python-louvain (community) ayrı isimle import edilir
if python3 -c "import community" 2>/dev/null; then
    ok "python-louvain (community)"
else
    warn "python-louvain yok  ->  pip install python-louvain"
fi

# --- 2. GPU ve sürücü --------------------------------------------------------
echo; echo ">>> 2. GPU durumu"
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version \
               --format=csv,noheader 2>/dev/null | while read -r line; do
        ok "GPU: $line"
    done
    TOTAL_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [ -n "${TOTAL_MB:-}" ] && [ "$TOTAL_MB" -lt 7000 ]; then
        warn "VRAM ${TOTAL_MB}MB (<7GB). num_ctx=2048 ŞART, ekran yükü GPU'da olmamalı."
    fi
else
    warn "nvidia-smi yok (WSL'de GPU passthrough kapalı olabilir). CPU modunda 3-6 tok/s beklenir."
fi

# --- 3. Ollama ---------------------------------------------------------------
echo; echo ">>> 3. Ollama servisi"
if command -v ollama &>/dev/null; then
    ok "ollama kurulu: $(ollama --version 2>&1 | head -1)"
    if curl -s http://localhost:11434/api/tags &>/dev/null; then
        ok "Ollama servisi çalışıyor (localhost:11434)"
        echo "    Yüklü modeller:"
        curl -s http://localhost:11434/api/tags 2>/dev/null \
            | python3 -c "import sys,json; [print('     -',m['name']) for m in json.load(sys.stdin).get('models',[])]" 2>/dev/null \
            || echo "     (liste alınamadı)"
    else
        warn "Ollama servisi kapalı  ->  yeni terminalde:  ollama serve"
    fi
else
    err "ollama yok. Kur:  curl -fsSL https://ollama.com/install.sh | sh"
fi

# --- 4. Gerekli modeller -----------------------------------------------------
echo; echo ">>> 4. Backbone modeller (Q4_K_M)"
NEED_MODELS=("qwen2.5:7b" "llama3.1:8b")
if curl -s http://localhost:11434/api/tags &>/dev/null; then
    INSTALLED=$(curl -s http://localhost:11434/api/tags | python3 -c "import sys,json;print(' '.join(m['name'] for m in json.load(sys.stdin).get('models',[])))" 2>/dev/null)
    for m in "${NEED_MODELS[@]}"; do
        if echo "$INSTALLED" | grep -q "${m%%:*}"; then
            ok "$m mevcut"
        else
            warn "$m yok  ->  ollama pull $m"
        fi
    done
else
    warn "Servis kapalı olduğu için model kontrolü atlandı."
fi

# --- 5. RoG subgraph veri setleri --------------------------------------------
echo; echo ">>> 5. RoG ön-işlenmiş subgraph veri setleri (HuggingFace)"
echo "    İndirme (Stage 2'de yapılacak):"
echo "      python3 -c \"from datasets import load_dataset; load_dataset('rmanluo/RoG-webqsp')\""
echo "      python3 -c \"from datasets import load_dataset; load_dataset('rmanluo/RoG-cwq')\""
CACHE="${HF_HOME:-$HOME/.cache/huggingface}/datasets"
for ds in RoG-webqsp RoG-cwq; do
    if find "$CACHE" -iname "*${ds#RoG-}*" 2>/dev/null | grep -qi "$ds" 2>/dev/null \
       || ls -d "$CACHE"/*"${ds,,}"* &>/dev/null; then
        ok "$ds önbellekte görünüyor"
    else
        warn "$ds henüz indirilmemiş (Stage 2'de inecek)"
    fi
done

echo; echo "============================================================"
echo " Doğrulama bitti. [UYARI] satırlarını sırayla giderin."
echo "============================================================"
