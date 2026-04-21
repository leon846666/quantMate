# quantMate

> 多因子 A 股量化研究框架 — PostgreSQL + Parquet + Tushare + LightGBM

---

## 1. 快速开始

### 1.1 准备环境
```bash
# 推荐 Python 3.10+，在项目根目录创建虚拟环境
python -m venv .venv
.venv\Scripts\activate         # Windows
# source .venv/bin/activate    # macOS/Linux

pip install -r requirements.txt
```

### 1.2 准备 PostgreSQL
1. 安装 PostgreSQL 14+，记下 `host / port / user / password`
2. 打开 `config/settings.yaml`，填入 `database` 部分的真实连接信息
3. 初始化数据库：
   ```bash
   python data/db/init_db.py
   ```
   该脚本会自动创建 `quantmate` 库以及所有表（幂等，`IF NOT EXISTS`）。

### 1.3 填入 Tushare Token
打开 `config/settings.yaml`，把 `data_source.tushare.token` 改成你自己的 token；
也可以直接用环境变量 `TUSHARE_TOKEN` 覆盖它。

### 1.4 一键跑 Demo
不依赖真实 tushare / postgres，用内置 mock 数据跑一遍完整管道（造数据→算因子→训 LightGBM→回测→出报告）：
```bash
python main.py --demo
```
结果会输出到 `reports/` 下，包含净值曲线图 + 绩效表。

### 1.5 跑真实管道
```bash
# 从 tushare 拉数据写入 PostgreSQL + Parquet
python main.py --fetch --start 2018-01-01 --end 2022-12-31

# 训练 LightGBM 多因子模型
python main.py --train

# 回测 + 出报告
python main.py --backtest
```

---

## 2. 模块职责

| 模块            | 目录               | 核心职责                                                              |
| --------------- | ------------------ | --------------------------------------------------------------------- |
| 数据模块        | `data/`            | Tushare 拉取, CSV 缓存, PostgreSQL 写入, OHLCV 存 Parquet             |
| 股票分析模块    | `analysis/`        | 选股过滤、因子计算、特征工程                                          |
| 策略模块        | `strategy/`        | 每策略一个类，继承 `BaseStrategy`                                     |
| 回测模块        | `backtest/`        | 独立回测引擎，模拟成交、仓位变化                                      |
| 评价模块        | `evaluation/`      | 收益率、最大回撤、夏普、胜率、IC、IR                                  |
| 可视化模块      | `visualization/`   | 净值曲线、分组回测、报表                                              |
| 实盘模块        | `trading/`         | 券商 API 接入占位                                                     |
| Utils           | `utils/`           | 日志、日期、配置加载                                                  |

模块之间只通过数据结构 (`data/models.py`) 和接口耦合，互不影响增删。

---

## 3. VS Code 调试

打开项目根目录后，按 `F5` 可以选择以下调试配置（见 `.vscode/launch.json`）：

- `main.py --demo (mock data)` — 零依赖跑通
- `main.py --fetch (real tushare)` — 真实拉数据
- `main.py --train` — 训模型
- `init_db.py` — 初始化数据库
- `pytest: current file` — 当前打开的测试文件

扩展推荐：`Python`, `Debugpy`, `Jupyter`, `YAML`, `SQLTools + PostgreSQL driver`。

---

## 4. 状态存储

| 数据 | 存放位置 |
| --- | --- |
| 日线 OHLCV | `data/market/<ts_code>.parquet` |
| 结构化数据 | PostgreSQL `quantmate` 库 |
| Tushare 原始缓存 | `data/cache/*.csv` |
| 日志 | `logs/quantmate.log` |
| 回测报告 | `reports/` |

---

## 5. 开发约定

- 一切配置走 `config/settings.yaml`，禁止硬编码
- 新加策略：在 `strategy/` 下新建 `xxx.py`，继承 `BaseStrategy`，然后在 `strategy/registry.py` 注册
- 新加因子：在 `analysis/factors.py` 里加一个函数，用 `@register_factor` 装饰器自动接入
- 所有日志走 `utils/logger.get_logger(__name__)`

---

## 6. 许可

MIT (自用研究项目)
