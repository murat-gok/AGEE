# AGEE KBS Revizyonu — Deney Paketi Çalıştırma Kılavuzu

Bu paket, hakem taleplerinden **deney gerektirenleri** (İP-1 WebQSP/CWQ, İP-2 ToG/PoG,
İP-3 ikinci LLM) makinenizde uçtan uca koşturmanız için hazırlandı.
Hedef donanım: **RTX 3050 6GB VRAM, 16GB RAM, WSL Ubuntu**.

> **Temel mimari kararı:** ToG ve PoG tam Freebase'i Virtuoso'da bekler (~100 GB RAM).
> 16 GB makinede bu çalışmaz. Bu yüzden Virtuoso'yu **baypas edip** RoG'un
> soru-bazlı ön-işlenmiş subgraph'larını (HuggingFace) kullanıyoruz.

---

## Dosya yapısı

```
agee_experiments/
├── scripts/
│   ├── 00_setup_check.sh                  # Ortam doğrulama (önce bunu çalıştır)
│   └── 10_run_agee_on_trajectories.py     # Trajectory -> AGEE -> sonuç tabloları
├── adapters/
│   ├── rog_subgraph_adapter.py            # Virtuoso baypas (RoG subgraph KG)
│   └── local_llm_bridge.py                # ToG/PoG -> lokal Ollama köprüsü
├── parsers/
│   └── tog_pog_parser.py                  # reasoning_chains -> trajectory
└── README.md                              # bu dosya
```

---

## Adım 0 — Ortam kurulumu (~yarım gün)

```bash
# Ollama kurulumu
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &                     # arka planda
ollama pull qwen2.5:7b             # birincil backbone (mevcut deneyinizle uyumlu)
ollama pull llama3.1:8b            # ikinci backbone (İP-3, R1.4 cevabı)

# Python bağımlılıkları
pip install networkx python-louvain numpy scipy pandas datasets openai

# Doğrulama
bash scripts/00_setup_check.sh
```

`[UYARI]` satırlarını sırayla giderin. **6GB için kritik:** `num_ctx=2048`,
GPU'da ekran yükü olmamalı. Çalışırken ayrı terminalde `watch -n1 nvidia-smi`
ile izleyin; GPU-Util düşük + RAM yüksekse katmanlar CPU'ya kaçıyordur
(hız 5-10x düşer) → context'i 1536'ya indirin.

---

## Adım 1 — ToG ve PoG repolarını hazırla

```bash
cd /mnt/c/Users/user/metricAGEE/agee_experiments
git clone https://github.com/IDEA-FinAI/ToG.git
git clone https://github.com/liyichen-cly/PoG.git
```

### Otomatik yama (ToG)

Elle düzenleme YOK — patch scripti üç dosyayı (`freebase_func.py`, `utils.py`,
`main_freebase.py`) otomatik yamalar ve her birinin `.bak` yedeğini alır:

```bash
# 1) Backbone'u seç (deney başına değiştir)
#    adapters/local_llm_bridge.py içinde ACTIVE_BACKBONE = "qwen2.5:7b" veya "llama3.1:8b"

# 2) ToG'u yamala
python scripts/02_patch_tog.py --tog-root ToG --dataset webqsp

# Geri almak istersen:
# python scripts/02_patch_tog.py --tog-root ToG --restore
```

Yama ne yapar:
- `freebase_func.py`: `relation_search_prune`, `entity_search`,
  `id2entity_name_or_type` -> RoG subgraph shim'i (Virtuoso baypas)
- `utils.py`: `run_llm` -> lokal Ollama köprüsü
- `main_freebase.py`: her soruda doğru subgraph'ı aktif eden çağrı

> **Virtuoso KURMUYORUZ.** Shim, KG erişimini bellekteki RoG subgraph'a
> yönlendirir. Entity adları doğrudan kimlik olarak kullanılır (MID yok).

PoG da ToG forku olduğu için aynı yapıdadır; PoG için patch scriptindeki
`--tog-root PoG` ile deneyin, ya da PoG'un `our_results/` örnek çıktısıyla
şemayı doğrulayıp aynı shim'i elle bağlayın.

---

## Adım 2 — RoG subgraph'larını indir + ToG formatına çevir

```bash
# RoG subgraph'larını indir (ilk çağrıda otomatik iner)
python -c "from datasets import load_dataset; load_dataset('rmanluo/RoG-webqsp')"
python -c "from datasets import load_dataset; load_dataset('rmanluo/RoG-cwq')"

# RoG -> ToG veri formatı + shim subgraph indeksi üret
python scripts/01_prepare_rog_for_tog.py --dataset webqsp --split test --limit 200
python scripts/01_prepare_rog_for_tog.py --dataset cwq --split test --limit 200
```

Bu script şunları üretir:
- `ToG/data/WebQSP.json` (ToG'un `prepare_dataset`'inin okuyacağı)
- `rog_subgraphs_webqsp.jsonl` (shim'in subgraph indeksi)

> Tam Freebase Virtuoso'yu **KURMAYIN** — 16 GB RAM'e sığmaz. Her soru subgraph'ı
> ~1300-1950 düğüm, belleğe rahat sığar.

---

## Adım 3 — Pilot koşu (~1 gün)

Önce küçük örneklemle (Adım 2'de `--limit 200` ile hazırladınız) her şeyin
uçtan uca çalıştığını doğrulayın.

```bash
# Köprüyü test et (Ollama açık olmalı)
python adapters/local_llm_bridge.py     # "OK" benzeri yanıt dönmeli
```

### ToG'u çalıştır — crash-safe (gece boyu koşu için ÖNERİLEN)

ToG'un normal `main_freebase.py`'si bir soruda hata alırsa (örn. IndexError)
TÜM koşu çöker. Gece boyu koşacak iş için crash-safe runner kullanın:

```bash
cd /mnt/c/Users/user/metricAGEE/agee_experiments/ToG/ToG

python /mnt/c/Users/user/metricAGEE/agee_experiments/scripts/03_run_tog_safe.py \
    --dataset webqsp --width 3 --depth 3 --LLM_type qwen2.5:7b --prune_tools llm
```

Bu runner:
- Bir soruda hata olursa o soruyu ATLAR, koşu DEVAM EDER (hatayı `failed_webqsp.jsonl`'e loglar)
- Her işlenen soruyu `checkpoint_webqsp.txt`'e yazar — kesinti olursa **kaldığı yerden devam** eder (tekrar çalıştırmanız yeterli)
- Çıktı yine `ToG_webqsp.jsonl` (parser bunu okur)

> **Arka planda koşturma (terminal kapansa bile sürer):**
> ```bash
> nohup python /mnt/c/Users/user/metricAGEE/agee_experiments/scripts/03_run_tog_safe.py \
>     --dataset webqsp --width 3 --depth 3 --LLM_type qwen2.5:7b \
>     > tog_run.log 2>&1 &
> ```
> İlerlemeyi izlemek: `tail -f tog_run.log`. Baştan başlamak için `--no-resume`
> ekleyip eski `checkpoint_webqsp.txt` + `ToG_webqsp.jsonl`'i silin.

CWQ için: önce `02_patch_tog.py --tog-root ToG --dataset cwq` ile yamayı cwq
indeksine yönlendirin, `01_prepare_rog_for_tog.py --dataset cwq --shim-index-dir ToG/ToG`
çalıştırın, sonra runner'ı `--dataset cwq --depth 4` ile koşun.

> **Backbone değiştirme:** `adapters/local_llm_bridge.py` içinde
> `ACTIVE_BACKBONE` değişkenini `"llama3.1:8b"` yapıp aynı koşuyu tekrarlayın
> (İP-3 / R1.4 için ikinci LLM). `--LLM_type` argümanı ne olursa olsun köprü
> `ACTIVE_BACKBONE`'u kullanır.

### Trajectory çıkar (ÖNCE şemayı doğrula!)
```bash
cd /mnt/c/Users/user/metricAGEE/agee_experiments

# İlk iş: gerçek çıktının şemasını gör
python parsers/tog_pog_parser.py --inspect ToG/ToG/ToG_webqsp.jsonl

# Şema doğruysa parse et
python parsers/tog_pog_parser.py ToG/ToG/ToG_webqsp.jsonl \
    --out ToG_webqsp_agee_ready.json
```

> ToG'un `save_2_jsonl` çıktısı `{"question","results","reasoning_chains"}`
> şemasında — parser bunu doğrudan okur. PoG çıktısı için `our_results/`
> içindeki örnek satırla `--inspect` sonucunu karşılaştırın; anahtar adı
> farklıysa parser'daki `_CHAIN_KEYS` listesine ekleyin.

### AGEE hesapla
```bash
python scripts/10_run_agee_on_trajectories.py \
    --dataset webqsp --rog-split test \
    --agee-ready ToG_webqsp_agee_ready.json \
    --agent ToG \
    --out results_ToG_webqsp.csv \
    --limit 200          # pilot için; tam koşuda kaldırın
```

Script `metricAGEE_v3` paketinizi otomatik arar (`METRIC_AGEE_PATH` ortam
değişkeniyle yolu verebilirsiniz). Bulursa makaledeki v3 formülüyle birebir
skor üretir; bulamazsa referans implementasyonla pipeline'ı test eder.

> **metricAGEE_v3 API uyarlaması:** Scriptteki `compute_agee` içindeki
> `calc.compute(trajectory, G)` çağrısını kendi `AGEECalculator` arayüzünüze
> göre düzenleyin (metodun adı/imzası farklıysa).

---

## Adım 4 — Tam koşu ve matris

Pilot sorunsuzsa tam test setlerine ölçekleyin. Hedef deney matrisi:

| Dataset | Ajanlar                              | Backbone'lar          |
|---------|--------------------------------------|------------------------|
| WebQSP  | ToG, PoG, BFS, Greedy, RandomWalk    | Qwen2.5-7B, Llama3.1-8B |
| CWQ     | ToG, PoG, BFS, Greedy, RandomWalk    | Qwen2.5-7B, Llama3.1-8B |
| MetaQA  | (mevcut sonuçlar — yeniden koşmaya gerek yok) | Qwen2.5-7B       |

Her hücre için ToG/PoG'u koş → parse et → AGEE hesapla. BFS/Greedy/RandomWalk
zaten mevcut AGEE kodunuzda var; onları aynı RoG subgraph'ları üzerinde
koşturmak için `SubgraphKG.to_networkx()` grafını ajanlarınıza verin.

> **Kapsam eşiği:** Tam CWQ (3,531 soru × çok sayıda sıralı LLM çağrısı × ~15 tok/s)
> compute bütçenizi aşarsa, hop-sayısına göre **tabakalı örneklem** (örn. her
> hop sınıfından 300 soru) raporlayın ve makalede açıkça belirtin. ToG'un orijinal
> ablasyonları da 1000'er soruluk alt-küme kullanmıştı — bu kabul edilebilir.

---

## Beklenen çıktılar (makale tabloları için)

Her koşu şunları verir:
- `results_<agent>_<dataset>.csv` — soru başına Hits@1, AGEE, S/I/E, subgraph boyutu
- Özet: ajan başına ortalama Hits@1 & AGEE
- Bileşen-Hits@1 nokta-biserial korelasyonları (Tablo 5'in WebQSP/CWQ versiyonu)

CSV'leri bana geri verin; **revizyon tablolarını ve metni** (Sonuçlar,
Interpretability, response-to-reviewers) ben dolduracağım.

---

## Sorun giderme

| Belirti | Olası neden / çözüm |
|---------|---------------------|
| Çok yavaş (<5 tok/s) | Katmanlar CPU'da. `num_ctx`'i düşür, `ollama ps` ile kontrol et |
| Boş/bozuk LLM yanıtı | Küçük model formatı bozuyor. `local_llm_bridge` 3 kez dener; oran yüksekse prompt'u sadeleştir |
| `qid eşleşmedi` uyarısı | parser qid'i ile RoG id'si farklı. Script soru metniyle de dener; yine olmazsa eşleme fonksiyonunu uyarla |
| `reasoning_chains` boş | Ajan o soruda hiç keşif yapmamış (erken cevap). Normaldir; trajectory boş kayıtları AGEE'de ayrı işaretle |
| Louvain hatası | `pip install python-louvain` (import adı `community`) |

---

## Hatırlatma: bu paketin kapsamı

Bu paket **yalnızca deney kısmıdır** (İP-1/2/3). Deney gerektirmeyen işleri
(İP-4 complexity, İP-5 related work + 3 referans, İP-6 interpretability metni,
iç tutarsızlık düzeltmeleri, response-to-reviewers, DOCX/figür format işleri)
sonuçlar gelince ben yazacağım.
