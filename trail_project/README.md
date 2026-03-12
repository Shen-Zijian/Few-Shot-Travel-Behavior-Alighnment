# TRAIL

**Travel Behavior Retrieval-Augmented Iterative Alignment with LLMs**

TRAIL 是一套基于少量样本（few-shot）的出行行为校准框架，利用大语言模型（LLM）融合历史行为先验（TCS 2011）与少量当期调查样本（TCS 2022），实现跨时间的交通方式预测与出行行为分布对齐。本项目当前聚焦 **主要交通方式预测（mode choice prediction）**。

---

## 任务定义

当前代码使用的是 **跨年共享的 9 类机械化交通方式任务**，不是之前 README 中写的 5 类任务。

依据：

- `TCS2011/TCS2011 Raw Data & Operational Mannual/DOM-TCS2011.pdf` 说明 `TP24` 只保留 9 类主要机械化方式，并提供 `Mode_Hier`
- `TCS 2022 Database 2/TCS 2022 DATABASE(burn to CD)/SubModeLookup.xlsx` 说明 `Main_mode <= 9` 对应同一套 9 类层级

统一后的共享标签空间如下：

| 编码 | 2011 `Mode_Hier` | 2022 `Main_mode` | 标签 |
|------|------------------|------------------|------|
| 1 | 1 | 1 | Rail |
| 2 | 2 | 2 | LRT |
| 3 | 3 | 3 | Tram |
| 4 | 4 | 4 | Ferry |
| 5 | 5 | 5 | PLB |
| 6 | 6 | 6 | Bus |
| 7 | 7 | 7 | Private Vehicle |
| 8 | 8 | 8 | Taxi |
| 9 | 9 | 9 | SPB |

2022 中以下编码不进入当前共享任务：

- `10 = Other`
- `99 = Walk`
- `999 = Unknown`

---

## 数据说明

### TCS 2011（历史先验）

核心数据来自 Access 数据库 `TCS2011 database.accdb`，当前主任务使用 4 张表：

| 表 / 文件 | 描述 |
|-----------|------|
| `HH` | household 数据 |
| `HM` | household member 数据 |
| `TP24` | mechanised trip 数据，只保留 9 类主要机械化方式 |
| `11TPUSB` | TPUSB 到 `DB26` 的空间映射 |

`DOM-TCS2011.pdf` 明确说明：

- `TP24` 是从 `TP` 中抽出的 mechanised trip 子表
- 只保留 9 类主要机械化方式
- `Mode_Hier` 的层级为 `Rail, LRT, Tram, Ferry, PLB, Bus, Private Vehicle, Taxi, SPB`

当前 2011 预处理链路：

1. 从 Access 读取 `HH / HM / TP24 / 11TPUSB`
2. 用 `Q_NO + MEM` 联表 `TP24` 和 `HM`
3. 用 `Q_NO` 联表 `HH`
4. 用 `D5 / D6 -> 11TPUSB -> DB26` 把起终点统一到 26 区
5. 用 `Mode_Hier` 直接映射到统一 9 类标签

### TCS 2022（目标年份）

需联表 3 个 Excel 文件：

| 文件 | 描述 | 行数 |
|------|------|------|
| `HH.xlsx` | household 层级 | 35,325 |
| `HM.xlsx` | member 层级 | 89,471 |
| `TP.xlsx` | trip 层级 | 125,237 |

当前 2022 预处理链路：

1. 读取 `HH / HM / TP`
2. 先过滤 `Main_mode <= 9`
3. 联表 household 和 member 特征
4. 将 `Main_mode` 直接映射到统一 9 类标签

过滤结果：

- 原始 `TP`: 125,237
- 过滤后 `Main_mode <= 9`: 81,728

---

## 当前模式分布

以下分布来自当前代码重新预处理后的产物：

- `data/interim/harmonized/tcs2011_harmonized.parquet`
- `data/interim/harmonized/tcs2022_harmonized.parquet`

### TCS 2011

| 编码 | 标签 | 数量 | 占比 |
|------|------|------|------|
| 1 | Rail | 46,173 | 38.1% |
| 2 | LRT | 4,180 | 3.4% |
| 3 | Tram | 1,527 | 1.3% |
| 4 | Ferry | 543 | 0.4% |
| 5 | PLB | 14,167 | 11.7% |
| 6 | Bus | 36,318 | 30.0% |
| 7 | Private Vehicle | 8,870 | 7.3% |
| 8 | Taxi | 2,487 | 2.1% |
| 9 | SPB | 6,939 | 5.7% |

### TCS 2022

| 编码 | 标签 | 数量 | 占比 |
|------|------|------|------|
| 1 | Rail | 33,709 | 41.2% |
| 2 | LRT | 3,071 | 3.8% |
| 3 | Tram | 409 | 0.5% |
| 4 | Ferry | 353 | 0.4% |
| 5 | PLB | 7,310 | 8.9% |
| 6 | Bus | 24,211 | 29.6% |
| 7 | Private Vehicle | 7,841 | 9.6% |
| 8 | Taxi | 1,407 | 1.7% |
| 9 | SPB | 3,417 | 4.2% |

---

## 变量协调

两年数据均映射到统一 schema `UnifiedTripRecord`，规则定义在 `configs/data/harmonization.yaml`。

### 核心字段对照

| 统一字段 | 2011 来源 | 2022 来源 | 备注 |
|----------|----------|----------|------|
| `main_mode` | `TP24.Mode_Hier` | `TP.Main_mode` | 当前任务统一为共享 9 类机械化模式 |
| `trip_purpose` | `TP24.Pur` | `TP.T_Pur` | 统一为 `1=Work / 2=Education / 7=Other` |
| `age_group` | `HM.B2` | `HM.B2` | 分组：`0-14 / 15-24 / 25-44 / 45-64 / 65+` |
| `sex` | `HM.B1` | `HM.B1` | `1=Male, 2=Female` |
| `employment_status` | `HM.E_Status` | `HM.E_status` | 统一为 `1..7` |
| `income_group` | `HH.B15` | 暂缺 | 2022 目前统一填 `0` |
| `car_availability` | `HH.C2A1/C2A2/C2B1/C2B2/C2C1/C2C2` | `HH.C1A1..C1A5` | 统一为 `0/1/2+` |
| `departure_period` | `TP24.TiPer` | `TP.4_Pks` | 两年都直接使用官方时段编码 |
| `journey_time` | `TP24.Joutm` | `TP.Journey Time` | 分钟 |
| `origin_zone` | `TP24.D5 -> 11TPUSB.DB26` | `TP.O_26PDD` | 统一到 26 区 |
| `destination_zone` | `TP24.D6 -> 11TPUSB.DB26` | `TP.D_26PDD` | 统一到 26 区 |
| `trip_weight` | `TP24.WT_TRIP` | `TP.WT_TRIP` | 扩展权重 |

---

## 代码结构

```text
trail_project/
├── configs/
│   ├── data/
│   │   ├── tcs2011.yaml
│   │   ├── tcs2022.yaml
│   │   └── harmonization.yaml
│   ├── model/
│   ├── experiment/
│   └── eval/
├── data/
│   ├── interim/harmonized/
│   └── processed/
├── scripts/
│   ├── preprocess_2011.py
│   ├── preprocess_2022.py
│   ├── build_harmonized_dataset.py
│   ├── build_prototypes.py
│   ├── run_baseline.py
│   ├── run_trail.py
│   ├── evaluate.py
│   └── export_tables_figures.py
└── src/trail/
    ├── data/
    │   ├── loader.py
    │   ├── harmonizer.py
    │   ├── filters.py
    │   ├── schema.py
    │   └── splitter.py
    ├── features/
    ├── prototypes/
    ├── retrieval/
    ├── llm/
    ├── baselines/
    ├── evaluation/
    └── visualization/
```

当前关键实现：

- `src/trail/data/loader.py`
  - 2011 从 Access 读取 `HH/HM/TP24/11TPUSB`
  - 2022 从 `HH/HM/TP.xlsx` 读取并联表
- `src/trail/data/harmonizer.py`
  - 将两年数据统一为共享 schema 和共享 9 类标签
- `scripts/preprocess_2011.py`
  - 产出 `tcs2011_harmonized.parquet`
- `scripts/preprocess_2022.py`
  - 产出 `tcs2022_harmonized.parquet`

---

## 快速开始

### 1. 安装依赖

```bash
cd trail_project
pip install -r requirements.txt
```

额外前提：

- Windows 需要已安装 `Microsoft Access Driver (*.mdb, *.accdb)`
- 2011 预处理依赖 `pyodbc`

### 2. 配置 API Key

```bash
# 编辑 trail_project/.env
OPENAI_API_KEY=...
```

当前 `src/trail/llm/client.py` 使用 OpenAI SDK，但默认请求 `https://api.poe.com/v1` 兼容端点；如果改为官方 OpenAI，需要同步修改该文件中的 `base_url`。

### 3. 预处理数据

```bash
python scripts/preprocess_2011.py
python scripts/preprocess_2022.py
python scripts/build_harmonized_dataset.py
```

### 4. 构建原型与历史检索库

```bash
python scripts/build_prototypes.py --n_clusters 10
```

### 5. 跑 baseline

```bash
python scripts/run_baseline.py --model all
python scripts/run_baseline.py --model xgboost --fewshot_ratio 0.05 --seed 42
```

### 6. 跑 TRAIL

```bash
python scripts/run_trail.py --fewshot_ratio 0.01 --dry_run --n_samples 50
python scripts/run_trail.py --fewshot_ratio 0.01
python scripts/run_trail.py --fewshot_ratio 0.05
python scripts/run_trail.py --fewshot_ratio 0.10
```

### 7. 评估与导出

```bash
python scripts/evaluate.py
python scripts/export_tables_figures.py
```

---

## 结果状态

此前仓库中的实验结果和 README 表述基于旧的 5 类口径，与当前代码不一致。现在已经修正为共享 9 类机械化模式口径，因此以下内容需要重新生成：

- `outputs/predictions/`
- `outputs/metrics/`
- `outputs/tables/`
- `outputs/figures/`
- README 中的模型性能表

也就是说，当前数据预处理和标签定义已经校正，但 baseline / TRAIL 指标需要在新口径下重跑后再解释。

---

## 已知限制

| 限制 | 说明 |
|------|------|
| 2011 依赖 Access 驱动 | 无 ODBC driver 时无法读取 `.accdb` |
| 2022 收入字段暂缺 | 当前统一填 `income_group = 0` |
| 旧实验结果已失效 | 模式口径从旧 5 类改为共享 9 类后需要全部重跑 |
| LLM 费用未重估 | 9 类任务下需重新评估 full run 成本 |

---

## 当前结论

当前代码已经按官方数据结构整理出 TCS 2011 和 TCS 2022 的共同交通模式，并完成统一编码对应：

- 不再使用 `WALKO24` 作为主任务的 2011 方式标签来源
- 2011 改为使用 `TP24.Mode_Hier`
- 2022 改为使用 `Main_mode <= 9` 的官方 9 类主方式层级
- 两年现在处于同一套 `1..9` 编码空间，可直接用于跨年 mode choice 建模
