-- ============================================================
-- quantMate PostgreSQL schema — idempotent (IF NOT EXISTS)
-- ============================================================

-- stock_basic: 股票基本信息
CREATE TABLE IF NOT EXISTS stock_basic (
    ts_code     VARCHAR(16)  PRIMARY KEY,
    symbol      VARCHAR(16)  NOT NULL,
    name        VARCHAR(64)  NOT NULL,
    industry    VARCHAR(64),
    market      VARCHAR(8),
    list_date   DATE,
    delist_date DATE,
    updated_at  TIMESTAMP    DEFAULT CURRENT_TIMESTAMP
);

-- daily_basic: 每日指标 (PE, PB, 换手率等)
CREATE TABLE IF NOT EXISTS daily_basic (
    ts_code       VARCHAR(16),
    trade_date    DATE,
    close         DOUBLE PRECISION,
    turnover_rate DOUBLE PRECISION,
    pe            DOUBLE PRECISION,
    pe_ttm        DOUBLE PRECISION,
    pb            DOUBLE PRECISION,
    ps            DOUBLE PRECISION,
    ps_ttm        DOUBLE PRECISION,
    dv_ratio      DOUBLE PRECISION,
    total_share   DOUBLE PRECISION,
    float_share   DOUBLE PRECISION,
    total_mv      DOUBLE PRECISION,
    circ_mv       DOUBLE PRECISION,
    PRIMARY KEY (ts_code, trade_date)
);
CREATE INDEX IF NOT EXISTS idx_daily_basic_date ON daily_basic(trade_date);

-- adj_factor: 复权因子
CREATE TABLE IF NOT EXISTS adj_factor (
    ts_code    VARCHAR(16),
    trade_date DATE,
    adj_factor DOUBLE PRECISION,
    PRIMARY KEY (ts_code, trade_date)
);

-- suspend: 停牌
CREATE TABLE IF NOT EXISTS suspend (
    ts_code       VARCHAR(16),
    trade_date    DATE,
    suspend_type  VARCHAR(16),
    suspend_reason TEXT,
    PRIMARY KEY (ts_code, trade_date)
);

-- fina_indicator: 财务指标
CREATE TABLE IF NOT EXISTS fina_indicator (
    ts_code        VARCHAR(16),
    ann_date       DATE,
    end_date       DATE,
    roe            DOUBLE PRECISION,
    roa            DOUBLE PRECISION,
    gross_margin   DOUBLE PRECISION,
    netprofit_yoy  DOUBLE PRECISION,
    or_yoy         DOUBLE PRECISION,
    current_ratio  DOUBLE PRECISION,
    quick_ratio    DOUBLE PRECISION,
    debt_to_assets DOUBLE PRECISION,
    PRIMARY KEY (ts_code, end_date)
);
CREATE INDEX IF NOT EXISTS idx_fina_end_date ON fina_indicator(end_date);

-- moneyflow: 资金流向
CREATE TABLE IF NOT EXISTS moneyflow (
    ts_code      VARCHAR(16),
    trade_date   DATE,
    buy_sm_vol   DOUBLE PRECISION,
    buy_md_vol   DOUBLE PRECISION,
    buy_lg_vol   DOUBLE PRECISION,
    buy_elg_vol  DOUBLE PRECISION,
    sell_sm_vol  DOUBLE PRECISION,
    sell_md_vol  DOUBLE PRECISION,
    sell_lg_vol  DOUBLE PRECISION,
    sell_elg_vol DOUBLE PRECISION,
    net_mf_vol   DOUBLE PRECISION,
    net_mf_amount DOUBLE PRECISION,
    PRIMARY KEY (ts_code, trade_date)
);

-- daily_ohlcv: 日线行情 (前复权)，同时也以 Parquet 存储在 data/market/
CREATE TABLE IF NOT EXISTS daily_ohlcv (
    ts_code    VARCHAR(16),
    trade_date DATE,
    open       DOUBLE PRECISION,
    high       DOUBLE PRECISION,
    low        DOUBLE PRECISION,
    close      DOUBLE PRECISION,
    vol        DOUBLE PRECISION,
    amount     DOUBLE PRECISION,
    adj        VARCHAR(8)   DEFAULT 'qfq',
    currency   VARCHAR(8)   DEFAULT 'CNY',
    PRIMARY KEY (ts_code, trade_date, adj)
);
CREATE INDEX IF NOT EXISTS idx_daily_ohlcv_date   ON daily_ohlcv(trade_date);
CREATE INDEX IF NOT EXISTS idx_daily_ohlcv_code   ON daily_ohlcv(ts_code);

-- factor_values: 计算好的因子快照 (便于回测时直接读)
CREATE TABLE IF NOT EXISTS factor_values (
    trade_date  DATE,
    ts_code     VARCHAR(16),
    factor_name VARCHAR(64),
    value       DOUBLE PRECISION,
    PRIMARY KEY (trade_date, ts_code, factor_name)
);
CREATE INDEX IF NOT EXISTS idx_factor_name_date ON factor_values(factor_name, trade_date);
