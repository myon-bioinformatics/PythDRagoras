# PythDRagoras

<p align="center">
  <img src="docs/brand/icon.png" alt="PythDRagoras icon" width="160" />
</p>

[Icon](docs/brand/icon.png) · [Favicon](docs/brand/favicon.png)

> Propositions, precedents, objections.
>
> Tests explanations as propositions with counterexamples, rule search and sensitivity analysis for NPB baseball outcomes. Split out from Aoi's pythdragoras/ (git subtree).

![PythDRagoras](https://github.com/myon-bioinformatics/Aoi/raw/main/docs/images/pythdragoras.webp)

*イメージ図（構想を共有するための図。実際のデータ・HTML構造ではない）*

## Role in Aoi

**Propositions, proofs and coordinate decomposition.**

| Challenges | Approach | Outcomes & value |
|---|---|---|
| Explanations are often ambiguous | 1. State propositions and draw diagrams | Understanding through figures and text |
| Diagrams and concepts are hard to grasp | 2. Search for conditions | Try conditions by changing them |
| The verification procedure is unclear | 3. Run sensitivity analysis | Prompts for further thinking |

Topics: mathematics, visualization, education support

## 概要

PythDRagoras は、説明を「もし A ならば B」の命題として書き、反例と異議を必ず並べて検証するツールです。答えを出すことより、説明が足りなくなる場所を見つけることを目的にしています。

- **命題と「異議あり」**: 元の命題・対偶・逆・裏の4つの形で判定します。判定の基準は結果を見る前に宣言し、反例が1件でもあれば「支持」でも判例を並べます（下の「命題と「異議あり」」）。
- **条件の探索**: 候補条件の組み合わせを総当たりで探し、目的の状態をどれだけ言い当てるかで並べます（`rule_search.py`）。
- **感度分析**: 指標・重み・境界や、試合ごとの得失点を動かしたときの変化を調べます（`season_sensitivity.py`・`score_sensitivity.py`）。
- 除外には理由を必須にし、何を・なぜ除いたかを結果に残します。外部の主張は観測ではなく claim として登録します。

思想は Aoi の [docs/PythDRagoras.md](https://github.com/myon-bioinformatics/Aoi/blob/main/docs/PythDRagoras.md)、命題の仕様は [docs/propositions.md](https://github.com/myon-bioinformatics/Aoi/blob/main/docs/propositions.md)。
[Aoi](https://github.com/myon-bioinformatics/Aoi) のサブプロジェクト [`pythdragoras/`](https://github.com/myon-bioinformatics/Aoi/tree/main/pythdragoras) から独立したリポジトリです（下の「Aoi との関係」）。

## インストールと使い方

Python 3.11 以上と [uv](https://docs.astral.sh/uv/) を使います。実行時の依存は polars と numpy だけです（`pyproject.toml`。ortools は使っていません）。

```bash
git clone https://github.com/myon-bioinformatics/PythDRagoras.git
cd PythDRagoras
uv sync
uv run python pythdragoras.py --help
```

分析の入力（サイクルデータ `cycles/` など）は Aoi に置いています。パスは呼び出し側（Aoi の `aoi_paths.py`）が解決して渡します。littlebit の CLI は `--cycle-dir`、または `--root` / `AOI_ROOT` と `AOI_RESEARCH_CYCLE` を受け取り、配置は探しません。実際のコマンド例は後半の「詳しい仕様と研究での使い方」にあります。

## 構成

| ファイル | 中身 |
|---|---|
| `pythdragoras.py` | シーズン表から残差・順位の偏り・命題を検証する CLI（`apply_exclusions`・`cumulative_test`・`rank_test`・`summary_markdown`） |
| `propositions.py` | 命題・判例・異議あり（仕様は [docs/propositions.md](https://github.com/myon-bioinformatics/Aoi/blob/main/docs/propositions.md)） |
| `sets.py` | 登録した命題の条件を集合として名前で呼び、∩・∪・¬ の式を命題として判定する |
| `rule_search.py` | 条件の組み合わせ（「または」でつないだ「かつ」の組）を総当たりで探す |
| `season_sensitivity.py` | 年成績の指標・重み・境界を動かす感度探索（標準ライブラリだけ） |
| `score_sensitivity.py` | 試合ごとの得失点を1軸だけ変える感度分析（差分更新） |
| `littlebit_approx.py`・`littlebit_prune.py`・`littlebit_holdout.py` | 候補の 0/1 行列から潜在パターンを条件候補にする探索と、その検証（実験） |
| `littlebit_survey.py` | 乗算・浮動小数なしの画像の領域・重心推定（実験、単一ファイル） |
| `registration_audit.py` | 来歴ラベル（事前登録か posthoc か）の監査 |
| `next_seeds.py` | 元の命題と逆の判定を並べ、次の命題の種を表示する（判定ではない） |

`pythdragoras.py` などの主な関数:

| 関数 | 中身 |
|---|---|
| `apply_exclusions` | 分析から除くシーズンを適用する。理由のない除外は受け付けず、何を・なぜ除いたかを結果に残す |
| `cumulative_test` | 期待勝率からのずれの累積を z 値にする（固定指数・可変指数の両方） |
| `rank_test` | 順位が下位に偏る珍しさ。帰無仮説（各シーズン独立に一様）と、全球団の実際の記録との比較を並べる |
| `summary_markdown` | 検証記録。原因は書かない。競合仮説の状態を並べる |
| `propositions.py` | 命題・判例・異議あり（仕様は [docs/propositions.md](https://github.com/myon-bioinformatics/Aoi/blob/main/docs/propositions.md)） |
| `littlebit_approx.py` | 候補の 0/1 行列から潜在パターンを条件候補にする（探索。判定は `rule_search.py`） |

## テスト

サイクルデータ（`cycles/`）、`scripts/`、SakAnalytics は Aoi に残します。コピーはせず、CI は実行時に Aoi の main を解決して checkout します。手元では main を使います。

```bash
git clone https://github.com/myon-bioinformatics/Aoi.git .test-deps/aoi
git -C .test-deps/aoi checkout main
AOI_ROOT=$PWD/.test-deps/aoi PYTHONPATH=$PWD/.test-deps/aoi/sakanalytics \
  uv run pytest -q --junitxml=reports/junit.xml
```

CI（`.github/workflows/ci.yml`）は Python 3.11–3.15 で同じことをします。`workflow_dispatch` の `aoi_ref` で ref を指定できます。解決した SHA はジョブの summary に残します。

- テストは `AOI_ROOT` だけで Aoi を受け取り、データの場所は Aoi の `aoi_paths.py` に聞きます。このリポジトリに `aoi_paths.py` は置きません。
- `AOI_ROOT` が Aoi を指していないと、pytest は最初に止まって理由を出します。
- 実行時の依存は polars と numpy だけです（ortools は使っていません）。
- テストは、これに加えて `AOI_ROOT` の Aoi を使います。
  - SakAnalytics: `tests/test_propositions.py:195`（`PYTHONPATH` の `sakanalytics`）。
  - Aoi の `scripts/`: `tests/test_c001_replay_compare.py:7`（`compare_c001_replay`）と `tests/test_c001_exploratory_candidates.py:14`（`check_c001_exploratory_candidates.py`）。
  - これ以外に Aoi の兄弟サブプロジェクトと `scripts/` を使わないことは、`tests/test_no_sibling_imports.py` で確かめます（上の3つは既知の例外）。

## Aoi との関係

このリポジトリは [Aoi](https://github.com/myon-bioinformatics/Aoi) のサブプロジェクト [`pythdragoras/`](https://github.com/myon-bioinformatics/Aoi/tree/main/pythdragoras) から独立したものです。
親は Aoi（Aoi の親は [myon-bioinformatics](https://github.com/myon-bioinformatics/myon-bioinformatics)）。Aoi の [`child-heads.json`](https://github.com/myon-bioinformatics/Aoi/blob/main/child-heads.json) に登録され、tree は登録した時点の Aoi の `pythdragoras/` と同じです（`.github/`、`pyproject.toml`、`uv.lock`、`conftest.py`、`.gitignore`、この README も含む）。`git subtree` で同期します（下の「Aoi との同期」。手順の全体は Aoi の [docs/child-repositories.md](https://github.com/myon-bioinformatics/Aoi/blob/main/docs/child-repositories.md)）。
この README は Aoi の `pythdragoras/README.md` と同じファイルです。Aoi の中の文書・データへのリンクは、どちらで読んでも開けるように GitHub の URL で書いています。

### サイクルのパス

`aoi_paths.py` は Aoi 直下の 1 か所だけに置きます（このリポジトリには置きません）。`littlebit_approx.py` と `littlebit_holdout.py` は配置を推定しません。サイクルのフォルダは `--cycle-dir` で渡します。省略するときは `--root` または環境変数 `AOI_ROOT` と、環境変数 `AOI_RESEARCH_CYCLE`（`<root>/cycles/<名前>`）です。

この 2 ファイルの SHA-256 は、Aoi の `cycles/c001-chunichi/outputs/exploratory/littlebit-holdout/holdout.provenance.json` で固定しています。子の CI は checkout した Aoi の `source_sha256` と `source_divergence` をそのまま読みます。pull request ではソースのバイトが違うとき警告だけにし、main への push と schedule では落ちます。フラグが無いときも落ちます。証跡の `evidence_sha256`、入力、`holdout.json` / `holdout.md` は動かしません。

### Aoi との同期

このリポジトリは Aoi の `pythdragoras/` の subtree です。除外するファイルはなく、Aoi の `pythdragoras/` の tree とこのリポジトリの tree は同じです。Aoi の中では `pythdragoras/.github/` は動かず（GitHub が読むのはリポジトリ直下の `.github/` だけ）、`pythdragoras/conftest.py` は Aoi の pytest でも同じように `AOI_ROOT` を確かめます。

- 履歴は Aoi `85243d6d1285e3bf4667e1447308543c7d648ab7`（main、#51 のマージコミット）の `pythdragoras/` から `git filter-branch --subdirectory-filter pythdragoras` で切り出しました（先頭 `12522c890e64313005e2bb3cad07d90365d4cf60`、tree `873bd0d947a99da043e314cb11762189540c6b98`）。git 2.47.3 の `git subtree split --prefix=pythdragoras` は Aoi の 585 コミット中 506 で止まり、10/4 の古い先頭を返したためです（[gh_identity#74](https://github.com/myon-bioinformatics/gh_identity/issues/74)）。
- この履歴は Aoi の履歴とつながっていないので、Aoi に一度だけ join を入れます（Aoi PR #52）。`git-subtree-split:` にこのリポジトリの main のコミットを書いた squash コミット（tree はこのリポジトリと同じ）を、Aoi の tree を変えないマージの2つ目の親にします。join 以降は `git subtree` が join を起点にするので、#74 の止まる範囲をたどりません。
- join は2つ目の親として残す必要があります。join を含む Aoi の PR は squash せず、merge commit で入れます（squash すると2つ目の親が消え、`git subtree pull` と `split` が止まります）。
- `git-subtree-split:` のコミットはこのリポジトリの main にある必要があります。このリポジトリの PR を squash マージして main の sha が変わったときは、Aoi で同じ tree の join を入れ直します。

子 → Aoi（Aoi の main から作ったブランチで）:

```bash
git subtree pull --prefix=pythdragoras https://github.com/myon-bioinformatics/PythDRagoras.git main --squash
test "$(git rev-parse HEAD:pythdragoras)" = "$(git rev-parse FETCH_HEAD^{tree})"   # Aoi 側に未同期の変更がないとき
```

Aoi → 子（Aoi の main で）:

```bash
git fetch https://github.com/myon-bioinformatics/PythDRagoras.git main   # split の起点のコミットを手元に置く
git subtree split --prefix=pythdragoras -b sync/pythdragoras
test "$(git rev-parse sync/pythdragoras^{tree})" = "$(git rev-parse HEAD:pythdragoras)"
git push https://github.com/myon-bioinformatics/PythDRagoras.git sync/pythdragoras:refs/heads/<新しいブランチ>
```

- push したブランチからこのリポジトリに PR を作ります。マージしたら、Aoi で「子 → Aoi」の pull をして、Aoi 側の起点をこのリポジトリの main に付け直します（squash マージでは split のコミットが main に残らないため）。
- gh_identity#74 が直るまで、Aoi に対して gh_identity の `subtree-sync --write` は使いません。`--write` なしの計画だけにし、tree の一致は上の `test` で確かめます。
- Aoi の `scripts/check_child_heads.py --fail-on-diff` が、Aoi の `child-heads.json` に登録した sha の tree と Aoi の `pythdragoras/` を比べます。

## 関連サブプロジェクト

親の [Aoi](https://github.com/myon-bioinformatics/Aoi) のサブプロジェクトは、それぞれ独立したリポジトリになっています。Aoi の中の場所（subtree）とあわせて載せます。一覧と登録の詳細は Aoi の [docs/child-repositories.md](https://github.com/myon-bioinformatics/Aoi/blob/main/docs/child-repositories.md) です。

| リポジトリ | Aoi の中の場所 | 役割 |
|---|---|---|
| [BlueProbe](https://github.com/myon-bioinformatics/BlueProbe) | [`blueprobe/`](https://github.com/myon-bioinformatics/Aoi/tree/main/blueprobe) | 観測と抽出（保存した HTML/CSS/DOM を読み、何があるかを記録する） |
| [QueRyu](https://github.com/myon-bioinformatics/QueRyu) | [`queryu/`](https://github.com/myon-bioinformatics/Aoi/tree/main/queryu) | 取得と観測データセットの作成（来歴つきのキャッシュ） |
| [SakAnalytics](https://github.com/myon-bioinformatics/SakAnalytics) | [`sakanalytics/`](https://github.com/myon-bioinformatics/Aoi/tree/main/sakanalytics) | 観測データからの指標と統計（チーム×シーズンの指標、残り試合の見通し） |
| **PythDRagoras**（このリポジトリ） | [`pythdragoras/`](https://github.com/myon-bioinformatics/Aoi/tree/main/pythdragoras) | 説明の検証（命題・反例・異議あり） |
| [NagoyAction](https://github.com/myon-bioinformatics/NagoyAction) | [`nagoyaction/`](https://github.com/myon-bioinformatics/Aoi/tree/main/nagoyaction) | パイプラインの実行と自動化（同じ定義をローカルと GitHub Actions で） |
| [DRAgoWing](https://github.com/myon-bioinformatics/DRAgoWing) | [`dragowing/`](https://github.com/myon-bioinformatics/Aoi/tree/main/dragowing) | 図とレポート（Plotly.js の HTML、四つの形の表） |

## ライセンス

MIT License（[LICENSE](LICENSE)）。Aoi の `LICENSE` と同じ内容です。

---

# 詳しい仕様と研究での使い方（NPB）

以下は Aoi の研究サイクル（中日・NPB のデータ）での使い方と、各機能の詳しい仕様です。コマンド例のパス（`pythdragoras/…`・`cycles/…`）は Aoi のルートからの場所です。

## 命題と「異議あり」

説明を「もし A ならば B」の命題として書き、元の命題・対偶・逆・裏の4つの形で判定する。
判定の基準は、結果を見る前に言葉で宣言する（必ず / ほとんど 0.90 / 概ね 0.75 / 多くの場合 0.50）。

```text
判定（上から順に最初に当てはまるもの）
  n < min_n                      → 判断保留
  必ず かつ 反例なし / あり        → 支持 / 棄却
  区間の下限 ≥ 基準                → 支持
  区間が基準をまたぐ              → 判断保留
  区間の上限 < 基準 かつ 関係あり   → 修正（主張が強すぎる）
  区間の上限 < 基準 かつ 関係なし   → 棄却
```

反例が1件でもあれば、判定が「支持」でも **異議あり** として判例を並べる。支持された説明の中の反例こそが、説明が足りなくなる場所だからである。
除外したシーズンの判例は、統計からは外しても「除外中の判例」として必ず表示する。

外部の主張（記事・レポート・他のAI）は、観測ではなく claim として登録し、パイプラインで再現できるかを確かめる。

## 潜在パターンを条件候補にする

`littlebit_approx.py` は、人が書いた候補条件のマスクを単位 × 条件の 0/1 行列にして、LittleBit 式（`littlebit_prune.littlebit`）で分解する。各潜在軸を「この単位はパターンに乗るか」という 1 本の条件にして `search` に渡す。成立と反例の数え方は `search` のビット演算のままである。目的の列は因子に渡さない。足した条件には `source=littlebit-approx`、rank、seed、元の条件と重み、`posthoc=true` を付ける。登録区分は `registration_audit.classify` に従い、この作り方では posthoc になる。

手順と、落とす軸・残差の手がかりの規則は [docs/littlebit-approx.md](docs/littlebit-approx.md)。c001 での並びは `cycles/c001-chunichi/outputs/exploratory/littlebit-approx/` に置く。確かめの記録は `confirmation.md` に置く。

帰無仮説は各シーズンを独立とみなすため、前年からの戦力の持ち越しを無視し、珍しさを過大に見積もる。
そのため、同じ期間の全球団との比較（経験的基準）を必ず並べる。

```bash
uv run python pythdragoras/pythdragoras.py --season cycles/c001-chunichi/outputs/season.jsonl \
  --config cycles/c001-chunichi/analysis.toml --hypotheses cycles/c001-chunichi/hypotheses.toml \
  --propositions cycles/c001-chunichi/propositions.toml --claims cycles/c001-chunichi/claims.toml \
  --outdir cycles/c001-chunichi/outputs
```

## 年単位から始める差分探索（season_sensitivity.py）

先に年成績の指標・重み・境界を動かし、変化と安定性を調べる。
有望な候補や説明の破れる年が見えてから、次節の試合単位へ進む。
このCLIは標準ライブラリだけで動作し、SciPyを必要としない。

```bash
python -S pythdragoras/season_sensitivity.py \
  --season cycles/c001-chunichi/outputs/season.jsonl \
  --additional-season cycles/c001-chunichi/references/season-2026-central-summary.jsonl \
  --config cycles/c001-chunichi/season_sensitivity.toml \
  --out cycles/c001-chunichi/outputs/season_sensitivity.json \
  --markdown cycles/c001-chunichi/outputs/season_sensitivity.md
```

`x + weight × y >= threshold` を設定し、片方を固定した感度曲線を計算する。
条件付き確率・MCC・基準との差、有限差分の傾き・変化・格子上の極値・最大値の区間・台形面積を出す。
閾値の移動では、境界を越えたチーム×年だけ混同行列の件数を更新する。
滑らかな関数の微分や極値を推定したとは扱わない。計算式と候補条件はTOMLに保存する。

年をまたいで同じ球団が含まれるため、独立標本の有意差検定は行わない。
1年の全球団を外して選び直す検証と、2026を選択から分離した照合を残す。
欠損した列・検証年・指標が外れるチーム×年も列挙し、中日の年次推移は欠年をつながない。
期待する6球団の不足、重複した球団×年、順位同率の未定義な扱いはエラーにする。
候補は探索的な関連であり、原因・CS出場確率・命題の支持を自動判定しない。

初回の結果・定義・読み方は [年単位の探索記録](https://github.com/myon-bioinformatics/Aoi/blob/main/cycles/c001-chunichi/research/R27-season-sensitivity.md)、
年別表は [実測結果](https://github.com/myon-bioinformatics/Aoi/blob/main/cycles/c001-chunichi/outputs/season_sensitivity.md)。
2026は既知の結果を含む回顧的照合であり、完全な盲検ではない。
通常の年表に2026を取り込んだ後は補助入力を外す。

## 次段階: 試合ごとの得失点変更（score_sensitivity.py）

NNUEの「変更箇所だけ更新する」考え方を参考に、試合の寄与を引いて新しい寄与を足す。
ニューラルネットや学習済み重みは使わない。対象は SakAnalytics と同じ試合原票。
RF・RA・W・L・T を差分更新し、派生値だけを再評価するため、各変更の更新量は試合数によらない。
探索全体は、対象試合数 × ステップ数 × 2軸に比例する。

```bash
uv run python pythdragoras/score_sensitivity.py \
  --games data/observations/npb_calendar/games.jsonl \
  --config cycles/c001-chunichi/analysis.toml \
  --max-step 3 --out cycles/c001-chunichi/outputs/score_sensitivity.json
```

`--team` で任意の設定済みチームを選べる。既定は `[focus].team`。

| 出力 | 意味 |
|---|---|
| `baseline` | 変更前の得失点、勝敗、勝率、指数1.83の期待勝率・期待勝利数 |
| `curve` | 各試合の得点を0〜3点増やす、または失点を0〜3点減らした結果。常に一方の軸だけ変更 |
| `delta` | 同じシーズンの基準値との差 |
| `timeline` | 日付・key順に1試合ずつ適用した途中経過、当該変更の差分、基準との差分 |
| `changed_runs` | 実際に変更した得失点の合計。失点は0で止める |
| `candidates` | 探索年の各試合1点改善で、変更1点あたりの勝ち越し増加が大きい順の命題候補 |

候補条件は既定で「探索年が3年以上あり、すべての探索年で勝ち越しが増加」。
`--min-seasons` で指定し、選択条件とともに結果に残す。2点・3点の結果を見て選択条件を変えない。
`[[exclude]]` は理由付きで適用し、`[confirm].seasons` は曲線を出すが候補の選択・順位には使わない。
候補は `propositions.toml` に自動登録しない。独立した年での検証と、人による命題の具体化が必要。

たとえば0対1の敗戦は、RFを1点増やすと引き分け、2点増やすと勝利となる。
増えた点数と勝利数の関係には段差があるため、期待勝率の滑らかな変化と実際の勝敗変化を両方残す。
全試合が引き分けなら勝率は未定義（null）、双方の総得点が0なら期待勝率もnullとする。

これは**最終スコアを変更する感度分析**であり、得点改善を実現する戦術・選手能力の因果効果ではない。
相手の同一試合の得失点も対になるが、全球団の順位は再計算しない。
イニング・延長・サヨナラ・その後の采配は再現せず、Aクラス確率も推定しない。
`timeline` は変更の適用順であり、将来の成績予測や実際のシーズン途中成績ではない。
入力範囲はユーザーが用意した原票そのもの。シーズン完結性は自動保証しないため、未完了年を含めるときは明示的に除外または検証用年に設定する。

検証: ランダムな80試合について、RF/RA両軸・4ステップの**各編集時点**を独立した全件再計算と照合。
引き分け、0失点、チームの向き、重複・不正原票、除外、検証用年からの選択漏洩、CLIも検証する。

## LittleBit 3-way survey（実験、`littlebit_survey.py`）

乗算と浮動小数を使わずに、画像の領域と重心をどこまで推定できるかを測る実験です。標準ライブラリだけの単一ファイルで、`pythdragoras.py` などの既存の処理は変えていません。

- **Way1**: 8x8 パッチ（4 画素ずつずらして重ねる）の行と列に、8 点 FWHT（加算・減算だけ）をかけます。AC 成分の L1 エネルギーを測り、パッチ平均（DC >> 6）との差を三値スピン（+1 / 0 / -1、閾値 24）にします。エネルギーが 512 未満、または有効スピンが 0 のパッチは飛ばします。
- **Way2**: 有効スピンの画素を 8 ビット回転と明暗反転で揺らし、そのパッチのエネルギー変化 ΔE（2 つの和）が 64 以上の画素を候補にします。刻み 1/2/4 の右・下の近傍に同じ符号の候補があればリンクを張ります。リンクが 2 本未満の候補は孤立ノイズとして捨てます。
- **Way3**: Way2 のリンク履歴だけから領域を再構成します（画像は見ない）。行と列を走査し、+ のノードで内側に入り、- のノードで外側に出ます。行走査と列走査の両方で内側になった画素を領域とし、重心は整数の (分子, 分母) で返します。
- **コアの約束**: Way1〜3 のコア関数（`CORE_FUNCTIONS`）は `*`・`/`・`**` と浮動小数を使いません。添字は行オフセットを足し合わせて作ります。テストで AST を検査します。比を小数にするのは表示（`--report`）のときだけです。乱数の seed は固定です。

```bash
python littlebit_survey.py --report
```

### 合格線とその根拠

64x64 の合成画像 6 種類で測りました。テストは固定 seed（11〜16）で行い、合格線は seed 1〜30 で測った最小値から余裕を取って決めました。

| シナリオ | 内容 | seed 1〜30 の IoU 最小 | 重心誤差の最大 (px) | 合格線（IoU / skip 率 / 重心誤差） |
|---|---|---:|---:|---|
| Dense Object | 塗りつぶした楕円 | 0.986 | 0.18 | ≥ 0.95 / ≥ 0.70 / ≤ 0.5 |
| Thin Skeleton | 2 px 幅の線 3 本 | 0.935 | 0.43 | ≥ 0.90 / ≥ 0.65 / ≤ 1.0 |
| Hollow Ring | 中空の輪（穴は空のまま） | 0.996 | 0.03 | ≥ 0.95 / ≥ 0.35 / ≤ 0.5 |
| High Noise Dark | 暗い背景、ゆらぎ ±8、3% の白点ノイズ | 0.961 | 0.60 | ≥ 0.90 / — / ≤ 1.0 |
| Low Contrast | 明暗差 60 の楕円 | 0.996 | 0.04 | ≥ 0.95 / ≥ 0.70 / ≤ 0.5 |
| Multi Blob | 離れた楕円 2 個 | 0.986 | 0.36 | ≥ 0.95 / ≥ 0.65 / ≤ 1.0 |

- skip 率は内部が一様なパッチほど高くなります。輪は縁が長いので低く（0.44）、ノイズの多い画像では白点のせいでほとんど飛ばせません（0.06〜0.18）。そのため High Noise Dark には skip 率の合格線を設けていません。
- High Noise Dark では、孤立ノイズを捨てる処理（リンク 2 本以上）をやめると IoU が 0.987 から 0.840 に下がります。テストでも、捨てる処理があるほうが IoU が 0.05 以上高いことを確認しています。
- 再構成した領域を (3, 2) ずらすと、どのシナリオでも IoU は下がります（線は 0.07 前後まで下がり、塗りつぶした形は 0.6〜0.74 程度）。
- 6 シナリオとも固定の合成画像による測定で、実際の画像での性能は保証しません。
