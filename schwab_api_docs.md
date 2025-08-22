# Charles Schwab API - Historical Price Data

## Overview
This endpoint retrieves historical price data for equity symbols from the Charles Schwab API.

## Parameters

### Required Parameters

#### `symbol` (string, query)
- **Description**: The equity symbol used to look up price history
- **Example**: `AAPL`
- **Required**: Yes

### Optional Parameters

#### `periodType` (string, query)
- **Description**: The chart period being requested
- **Available values**: `day`, `month`, `year`, `ytd`
- **Default**: Varies by context

#### `period` (integer, query)
- **Description**: The number of chart period types
- **Valid values by periodType**:
  - `day`: 1, 2, 3, 4, 5, 10
  - `month`: 1, 2, 3, 6
  - `year`: 1, 2, 3, 5, 10, 15, 20
  - `ytd`: 1
- **Defaults**:
  - `day`: 10
  - `month`: 1
  - `year`: 1
  - `ytd`: 1

#### `frequencyType` (string, query)
- **Description**: The time frequency type for data points
- **Available values**: `minute`, `daily`, `weekly`, `monthly`
- **Valid values by periodType**:
  - `day`: `minute`
  - `month`: `daily`, `weekly`
  - `year`: `daily`, `weekly`, `monthly`
  - `ytd`: `daily`, `weekly`
- **Defaults**:
  - `day`: `minute`
  - `month`: `weekly`
  - `year`: `monthly`
  - `ytd`: `weekly`

#### `frequency` (integer, query)
- **Description**: The time frequency duration
- **Valid values by frequencyType**:
  - `minute`: 1, 5, 10, 15, 30
  - `daily`: 1
  - `weekly`: 1
  - `monthly`: 1
- **Default**: 1

#### `startDate` (integer, query)
- **Description**: Start date in milliseconds since UNIX epoch
- **Example**: `1451624400000`
- **Default**: Calculated as (endDate - period), excluding weekends and holidays

#### `endDate` (integer, query)
- **Description**: End date in milliseconds since UNIX epoch
- **Example**: `1451624400000`
- **Default**: Market close of previous business day

#### `needExtendedHoursData` (boolean, query)
- **Description**: Include extended hours trading data
- **Values**: `true`, `false`

#### `needPreviousClose` (boolean, query)
- **Description**: Include previous close price and date
- **Values**: `true`, `false`

## Usage Examples

### Basic Daily Data
```
GET /pricehistory?symbol=AAPL&periodType=month&period=1&frequencyType=daily
```

### Intraday Minute Data
```
GET /pricehistory?symbol=TSLA&periodType=day&period=1&frequencyType=minute&frequency=5
```

### Year-to-Date Weekly Data
```
GET /pricehistory?symbol=MSFT&periodType=ytd&frequencyType=weekly
```

### Custom Date Range
```
GET /pricehistory?symbol=GOOGL&startDate=1609459200000&endDate=1640995200000&frequencyType=daily
```

## Notes

- All date parameters use milliseconds since UNIX epoch
- Weekend and holiday data is automatically excluded from default date calculations
- Extended hours data is optional and separate from regular trading hours
- Default values are applied when parameters are not specified, following the rules above