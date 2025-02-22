# Dataset Documentation

## Stock Prices
| Column | Type   | Description   | Source       |
|--------|--------|---------------|--------------|
| date   | Date   | Trading date  | Alpha Vantage|
| open   | Float  | Opening price | Yahoo Finance|
| volume | BIGINT | Shares traded | Yahoo Finance|


## Economic Indicators
| Column | Type  | Description            | Frequency |
|--------|-------|------------------------|-----------|
| GDP    | Float | Gross Domestic Product | Quaterly  |
|UNRATE  | Float | Unemployment Rate      | Monthly   |