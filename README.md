# DHM_QAOA

QAOA（Quantum Approximate Optimization Algorithm）を用いて **チーム分け問題** を解くための実験コードです。
標準的な **XY-QAOA（一様ミキサー）** と、プレイヤー間の実力差に応じてミキサーの重みを層ごとに変化させる **Adaptive SAHM-QAOA（ASA）** を、同一条件で比較します。

量子回路は [Qiskit](https://www.ibm.com/quantum/qiskit) で構築し、[Amazon Braket](https://aws.amazon.com/braket/) の状態ベクトルシミュレータ **SV1** 上で実行します。

## 問題設定

`P` 人のプレイヤーを、それぞれ `N` 人ずつの `G` チームに分け、各プレイヤーに `R` 種類のロール（役割）のいずれかを割り当てます。

- 変数 `x[p, g, r] ∈ {0, 1}`：プレイヤー `p` をチーム `g` のロール `r` に割り当てるとき 1
- 必要な量子ビット数：`P × G × R`

| 設定 | P | G | R | N | 量子ビット数 |
|------|---|---|---|---|-------------|
| P4   | 4 | 2 | 2 | 2 | 16 |
| P6   | 6 | 2 | 2 | 3 | 24 |

### QUBO の構成

目的関数は以下 5 項の重み付き和です（重みは `QUBO_LAMBDAS = (λ1, λ2, λ3, λ4, λ5)`）。

| 項 | 内容 |
|----|------|
| λ1 | チーム戦力の均衡（各チームの戦力の二乗和を罰則化し、強者が偏らないようにする） |
| λ2 | 同一チーム内でのロールの重複に対する罰則 |
| λ3 | 得意ロール（expertise）への割り当てに対する報酬 |
| λ4 | One-hot 制約：各プレイヤーはちょうど 1 つの (チーム, ロール) に属する |
| λ5 | チーム人数制約：各チームはちょうど `N` 人 |

戦力値は `strength_ratings × player_expertise` を最大値で正規化したものを使います。QUBO はイジング形式（Z / ZZ 項）に変換して位相分離（コスト）層に使用します。

### データ条件

| 条件 | 説明 | 戦力値 (strength) | 得意度 (expertise, ロールA/B) |
|------|------|------------------|------------------------------|
| **Kakusa（格差）** – P4 | 強者と弱者の差がはっきりしている | `[3.0, 1.0, 3.0, 1.0]` | `[5,1], [1,5], [5,1], [1,5]` |
| **Kinsa（僅差）** – P6 | 実力差が小さく拮抗している | `[2.0, 1.2, 1.8, 1.4, 2.2, 0.9]` | `[5,1], [1,5], [5,1], [1,5], [4,2], [1,1]` |

## 手法

どちらの手法も、制約を満たす決定論的な初期状態（実行可能解）から出発し、XX + YY 相互作用（`rxx` / `ryy`）によるミキサーで **同じチーム・同じロールのスロット間でプレイヤーを入れ替え** ます。パラメータは COBYLA で最適化します。

### Standard XY-QAOA（`XY_*.py`）

全プレイヤーペアに対して一様な重み `w = 1.0` のミキサーを適用するベースラインです。

### Adaptive SAHM-QAOA（`ASA_*.py`）

QAOA の層（ステップ）が進むにつれてミキサーの重みを「マクロ → ミクロ」に変化させます（カリキュラム的な混合）。

```
w_macro = (1 + |s_p1 - s_p2|)^3     # 実力差が大きいペアほど強く交換（大きく動かす）
w_micro = 1.0                       # 一様な探索（微調整）
alpha   = layer_idx / (total_layers - 1)
w       = (1 - alpha) * w_macro + alpha * w_micro
```

- 前半の層：実力差の大きいプレイヤー同士の交換を強く促し、戦力バランスを大まかに整える
- 後半の層：一様ミキサーに近づけ、局所的な微調整を可能にする
- 重みの大きい順にゲートを適用し、`w < 0.1` のペアは回路深さ削減のためスキップ

## リポジトリ構成

```
DHM_QAOA/
└── XY_and_ASA/
    ├── XY_P4_Kakusa_10times.py    # 標準 XY-QAOA, P=4, 格差データ
    ├── XY_P6_Kinsa_10times.py     # 標準 XY-QAOA, P=6, 僅差データ
    ├── ASA_P4_Kakusa_10times.py   # Adaptive SAHM-QAOA, P=4, 格差データ
    └── ASA_P6_Kinsa_10times.py    # Adaptive SAHM-QAOA, P=6, 僅差データ
```

各スクリプトはシード `BASE_SEED`（=314）から 1 ずつ変えて `NUM_EXPERIMENTS`（=10）回の実験を行い、結果を CSV に保存します。

## 必要環境

- Python 3.9 以上
- AWS アカウントと Amazon Braket の利用権限（SV1 を使用）

```bash
pip install numpy qiskit qiskit-algorithms qiskit-braket-provider
```

AWS 認証情報を設定しておきます（例：`aws configure`）。SV1 はリージョン `us-west-1` を想定しています。

> **注意**：SV1 はタスク実行時間に応じて課金されます。各実験では COBYLA の反復ごとに回路を実行するため、`P_STEPS` や `MAXITER` を大きくするとコストと実行時間が増えます。

## 使い方

```bash
cd XY_and_ASA
python XY_P4_Kakusa_10times.py
python ASA_P4_Kakusa_10times.py
```

### 主なパラメータ（各スクリプト冒頭）

| 変数 | 説明 |
|------|------|
| `PROBLEM_CONFIG` | 問題サイズ `{'P', 'G', 'R', 'N'}` |
| `QPU_BACKEND_NAME` | Braket のバックエンド名（既定 `"SV1"`） |
| `P_STEPS` | QAOA の層数 p |
| `MAXITER` | COBYLA の最大反復回数 |
| `SHOTS_EVAL` / `SHOTS_FINAL` | 最適化中 / 最終評価時のショット数 |
| `QUBO_LAMBDAS` | QUBO 各項の重み (λ1〜λ5) |
| `NUM_EXPERIMENTS` / `BASE_SEED` | 実験回数と開始シード |

## 出力

実行するとコンソールに各試行の結果と平均が表示され、CSV ファイルが作成されます。

| 列 | 説明 |
|----|------|
| `Run` / `Seed` | 試行番号と使用シード |
| `Best Team` | 得られた最良の実行可能解のチーム分け（例：P4 では `03\|12`、P6 では `025/134`） |
| `Valid Ratio` | 最終測定のうち制約（One-hot・人数）を満たすショットの割合 |
| `Success Prob` | 最良解のビット列が観測された確率 |
| `Best Energy` | 最良解の QUBO エネルギー |
| `Time(s)` | 最適化にかかった時間 |
| `Iterations` | 目的関数の評価回数 |

末尾に `AVERAGE`（平均）と `STD_DEV`（標準偏差）の行が追加されます。
