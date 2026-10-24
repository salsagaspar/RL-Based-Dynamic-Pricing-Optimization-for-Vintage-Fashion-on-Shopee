# RL-Based Dynamic Pricing Optimization for Vintage Fashion on Shopee

Repository ini merupakan arsitektur end-to-end proyek **Data Science & Reinforcement Learning (RL)** untuk optimasi penetapan harga dinamis (*dynamic pricing*) pada katalog pakaian vintage era 90-an (*grunge, washed boxy, retro tees, denim jackets*) yang dijual melalui platform e-commerce **Shopee**.

---

## 1. Latar Belakang & Masalah Bisnis

Penjualan pakaian *vintage* memiliki karakteristik unik yang membedakannya dari ritel fesyen konvensional:
1. **Kelangkaan & Keunikan Produk (*Scarcity*)**: Sebagian besar stok pakaian vintage 90-an bersifat terbatas (*one-of-a-kind*, *deadstock*, atau *curated thrift*). Jika dijual terlalu murah (*underpricing*), penjual kehilangan potensi margin yang signifikan (*left money on the table*).
2. **Biaya Penyimpanan & Risiko Deadstock (*Holding Cost*)**: Jika barang dibanderol terlalu mahal (*overpricing*), barang akan menumpuk di gudang, menyerap modal kerja, dan membebani margin akibat biaya simpan (*inventory holding cost*).
3. **Struktur Biaya Marketplace Shopee**:
   - Biaya Administrasi Star Seller (Fashion): ~4.5%
   - Biaya Partisipasi Program Gratis Ongkir Xtra: ~3.5%
   - Biaya Pemrosesan Transaksi: ~1.0%
   - *Total fee beban platform*: **~9.0%** dari harga transaksi.
4. **Lonjakan Trafik Promosi Platform**: Tanggal gajian (*Payday Sale*, tanggal 25–27) dan festival belanja *Double Date* (9.9, 10.10, 11.11, 12.12) meningkatkan intensitas niat beli (*demand shock*), membuka peluang penyesuaian margin tanpa menurunkan volume transaksi.

---

## 2. Formulasi Masalah: Markov Decision Process (MDP)

Optimasi harga dinamis dirumuskan sebagai proses keputusan Markov berhorizon terbatas $T = 60$ hari:

### A. Ruang Keadaan (*State Space* $S_t \in \mathbb{R}^6$)
Setiap langkah waktu harian $t$, agen mengamati vektor keadaan yang dinormalisasi:
1. **$s_{\text{stock}}$**: Rasio sisa stok terhadap stok awal ($\text{Stock}_t / \text{Stock}_0$).
2. **$s_{\text{time}}$**: Rasio sisa durasi musim/periode ($(T - t) / T$).
3. **$s_{\text{cost}}$**: Rasio modal unit terhadap harga katalog dasar ($\text{Cost} / P_{\text{base}}$).
4. **$s_{\text{price}}$**: Rasio harga berlaku saat ini terhadap harga katalog dasar ($P_t / P_{\text{base}}$).
5. **$s_{\text{campaign}}$**: Indikator biner hari promo Shopee ($1.0$ jika Payday/Double Date, $0.0$ hari biasa).
6. **$s_{\text{velocity}}$**: Rata-rata kecepatan penjualan (unit/hari) dalam 3 hari terakhir.

### B. Ruang Aksi (*Action Space* $A_t$)
Agen memilih penyesuaian harga relatif terhadap harga katalog referensi:
$$A = \{-20\%, -10\%, -5\%, 0\%, +5\%, +10\%, +20\%\}$$

### C. Kurva Permintaan Pasar (*Demand Response Simulator*)
Respon pasar dimodelkan dengan kurva elastisitas harga iso-elastis dengan proses stokastik Poisson:
$$\mathbb{E}[Q_t(P_t)] = Q_{\text{base}} \cdot \left(\frac{P_t}{P_{\text{base}}}\right)^{\epsilon} \cdot \beta_{\text{campaign}}$$
$$Q_t \sim \text{Poisson}(\mathbb{E}[Q_t(P_t)])$$
*di mana $\epsilon < 0$ adalah elastisitas harga kategori fesyen vintage dan $\beta_{\text{campaign}} \approx 1.6$ saat promo.*

### D. Fungsi Imbalan (*Reward Function* $R_t$)
Imbalan harian memperhitungkan laba bersih riil setelah potongan biaya Shopee, dikurangi biaya simpan dan penalti kehabisan stok:
$$R_t = \text{Sold}_t \times \left[ P_t \times (1 - \text{Fee}_{\text{Shopee}}) - \text{Cost} \right] - \left(\text{Stock}_t \times C_{\text{holding}}\right) - \left(\text{Unmet}_t \times C_{\text{stockout}}\right)$$

---

## 3. Struktur Direktori Proyek

```text
├── configs/
│   └── config.yaml               # Parameter pasar Shopee, fee, biaya holding, & hyperparameter RL
├── data/
│   ├── raw/                      # Salinan dataset asli Excel (6 sheet supply chain & pesanan)
│   └── processed/                # Data terproses & tabel agregasi SKU
├── models/                       # Checkpoint model & Q-table tersimpan (.pkl)
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb          # Eksplorasi 6 sheet & rantai pasok vintage
│   ├── 02_demand_elasticity_and_simulator.ipynb    # Validasi elastisitas & simulator permintaan
│   └── 03_rl_pricing_agent_training.ipynb          # Pelatihan RL, kurva konvergensi & komparasi
├── reports/
│   └── figures/                  # Visualisasi evaluasi kebijakan harga
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── loader.py             # Parser 6 lembar kerja Excel (Produk, Pemasok, Pesanan, Retur, dll)
│   ├── features/
│   │   ├── __init__.py
│   │   └── feature_pipeline.py   # Rekayasa fitur ekonomi, margin neto, & elastisitas
│   ├── environment/
│   │   ├── __init__.py
│   │   ├── demand_model.py       # Simulator respon elastisitas permintaan Shopee
│   │   └── shopee_pricing_env.py # Gymnasium-compatible environment simulasi retail dinamis
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py               # Interface kelas abstrak BasePricingAgent
│   │   ├── baselines.py          # FixedCatalog, RuleBasedMarkdown, & CampaignAware
│   │   ├── q_learning.py         # Tabular Q-Learning dengan diskritisasi state
│   │   └── bandit_agents.py      # LinUCB Contextual Bandit & Epsilon-Greedy
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── backtester.py         # Engine backtesting Monte Carlo multi-episode
│   │   └── metrics.py            # Kalkulasi GMV, Laba Bersih, ASP, Sell-Through, Holding Cost
│   └── utils/
│       ├── __init__.py
│       └── helpers.py            # Logger terstruktur, pemuat konfigurasi YAML, & format Rupiah
├── tests/
│   ├── test_data_loader.py       # Pengujian integritas pemuatan data Excel
│   ├── test_environment.py       # Pengujian reset, step, reward, & state space
│   └── test_agents.py            # Pengujian seleksi aksi & update kebijakan agen
├── dataset_supply_chain_pakaian_90an_shopee.xlsx   # File dataset kerja
├── app.py                        # Dashboard Web Interaktif (Streamlit App)
├── main.py                       # Pipeline utama pelatihan & benchmark via CLI
├── requirements.txt              # Daftar dependensi pustaka Python
├── .gitignore                    # Konfigurasi pengecualian git
└── README.md                     # Dokumentasi arsitektur proyek
```

---

## 4. Panduan Menjalankan Proyek

### A. Instalasi Dependensi
```bash
pip install -r requirements.txt
```

### B. Menjalankan Unit Tests
Pengujian otomatis memastikan parser data, environment simulasi, dan seluruh algoritma pricing berjalan valid:
```bash
python -m pytest tests/
```

### C. Melatih & Membandingkan Agen (*Training & Benchmark*)
Eksekusi pipeline lengkap untuk melatih agen Q-Learning dan membandingkannya terhadap agen aturan heuristik (*Fixed Price, Clearance Markdown, Campaign Rule*) dan Contextual Bandit (*LinUCB*):
```bash
python main.py --sku V90-TEE-001 --episodes 300 --eval 50
```

### D. Menjalankan Dashboard Web Interaktif (Streamlit)
Jalankan aplikasi web visual untuk menguji simulator, melihat trajektori harga dinamis, dan kalkulator fee Shopee:
```bash
streamlit run app.py
```

### E. Eksplorasi Interaktif melalui Jupyter Notebook
Buka direktori `notebooks/` menggunakan Jupyter Lab / VS Code untuk visualisasi mendalam:
```bash
jupyter notebook notebooks/01_exploratory_data_analysis.ipynb
```

---

## 5. Ringkasan Evaluasi Benchmark Strategi

Hasil perbandingan simulasi Monte Carlo 5 strategi penetapan harga pada SKU `V90-TEE-001` (*Kaos Graphic Grunge*):

| Agen / Kebijakan | Rata-rata Laba Bersih | Rata-rata GMV | Sell-Through Rate | Rata-rata Biaya Holding |
| :--- | :---: | :---: | :---: | :---: |
| **Rule-Based Markdown** | **Rp 129.838** | Rp 704.666 | 83.7% | Rp 118.848 |
| **Fixed Catalog Pricing** | Rp 123.412 | Rp 689.233 | 76.7% | Rp 122.990 |
| **LinUCB Contextual Bandit**| Rp 115.069 | Rp 678.445 | **94.3%** | **Rp 90.837** |
| **Q-Learning Agent (RL)** | Rp 110.036 | Rp 674.250 | 76.3% | Rp 122.792 |
| **Campaign-Aware Pricing** | Rp 108.391 | Rp 670.804 | 76.0% | Rp 122.360 |

### Wawasan Bisnis (*Actionable Insights*):
- **LinUCB Contextual Bandit** mencapai tingkat perputaran inventori tertinggi (**94.3% Sell-Through**) dan memangkas biaya simpan gudang hingga terendah (**Rp 90.837**), sangat ideal untuk barang vintage yang memiliki risiko susut nilai tinggi jika tersimpan terlalu lama.
- **Rule-Based Markdown** mempertahankan laba kotor tertinggi saat musim reguler, tetapi berisiko menumpuk stok mati jika tren vintage 90-an bergeser lebih cepat dari durasi siklus stok.
- **RL (Q-Learning)** mengadaptasi harga secara dinamis berdasarkan sisa stok harian dan hari kampanye Shopee, memberikan dasar bagi pengembangan model Deep Q-Network (DQN) atau PPO skala penuh.
