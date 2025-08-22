# Charles Schwab API - Order Management Documentation

## Overview
The Schwab Trader API supports order placement, modification, and cancellation for equity and option trades through POST and PUT endpoints.

## Rate Limits
- **Order Requests**: 0-120 requests per minute per account (throttled)
- **GET Requests**: Unlimited (unthrottled)
- Limits are set during application registration
- Contact: TraderAPI@schwab.com

## Supported Asset Types
- **EQUITY**: Stocks and ETFs
- **OPTION**: Options contracts

## Options Symbology
Options symbols follow this format:
```
[Underlying Symbol (6 chars)] + [Expiration (6 chars)] + [Call/Put (1 char)] + [Strike Price (8 chars)]
```

### Examples
- `XYZ   210115C00050000` = XYZ $50 Call expiring 2021-01-15
- `XYZ   210115C00055000` = XYZ $55 Call expiring 2021-01-15
- `XYZ   210115C00062500` = XYZ $62.50 Call expiring 2021-01-15

## Order Instructions by Asset Type

| Instruction     | EQUITY | OPTION |
|----------------|--------|--------|
| BUY            | ✓      | ✗      |
| SELL           | ✓      | ✗      |
| BUY_TO_OPEN    | ✗      | ✓      |
| BUY_TO_COVER   | ✓      | ✗      |
| BUY_TO_CLOSE   | ✗      | ✓      |
| SELL_TO_OPEN   | ✗      | ✓      |
| SELL_SHORT     | ✓      | ✗      |
| SELL_TO_CLOSE  | ✗      | ✓      |

## Order Examples

### 1. Buy Market Order (Stock)
Buy 15 shares of XYZ at market price, good for day.

```json
{
  "orderType": "MARKET",
  "session": "NORMAL",
  "duration": "DAY",
  "orderStrategyType": "SINGLE",
  "orderLegCollection": [
    {
      "instruction": "BUY",
      "quantity": 15,
      "instrument": {
        "symbol": "XYZ",
        "assetType": "EQUITY"
      }
    }
  ]
}
```

### 2. Buy Limit Order (Single Option)
Buy to open 10 contracts of XYZ March 15, 2024 $50 Call at $6.45 limit.

```json
{
  "complexOrderStrategyType": "NONE",
  "orderType": "LIMIT",
  "session": "NORMAL",
  "price": "6.45",
  "duration": "DAY",
  "orderStrategyType": "SINGLE",
  "orderLegCollection": [
    {
      "instruction": "BUY_TO_OPEN",
      "quantity": 10,
      "instrument": {
        "symbol": "XYZ   240315C00500000",
        "assetType": "OPTION"
      }
    }
  ]
}
```

### 3. Vertical Put Spread
Buy 2 contracts of XYZ $45 Put and sell 2 contracts of XYZ $43 Put at $0.10 net debit.

```json
{
  "orderType": "NET_DEBIT",
  "session": "NORMAL",
  "price": "0.10",
  "duration": "DAY",
  "orderStrategyType": "SINGLE",
  "orderLegCollection": [
    {
      "instruction": "BUY_TO_OPEN",
      "quantity": 2,
      "instrument": {
        "symbol": "XYZ   240315P00045000",
        "assetType": "OPTION"
      }
    },
    {
      "instruction": "SELL_TO_OPEN",
      "quantity": 2,
      "instrument": {
        "symbol": "XYZ   240315P00043000",
        "assetType": "OPTION"
      }
    }
  ]
}
```

### 4. One Triggers Another (OTA)
Buy 10 shares at $34.97 limit. If filled, sell 10 shares at $42.03 limit.

```json
{
  "orderType": "LIMIT",
  "session": "NORMAL",
  "price": "34.97",
  "duration": "DAY",
  "orderStrategyType": "TRIGGER",
  "orderLegCollection": [
    {
      "instruction": "BUY",
      "quantity": 10,
      "instrument": {
        "symbol": "XYZ",
        "assetType": "EQUITY"
      }
    }
  ],
  "childOrderStrategies": [
    {
      "orderType": "LIMIT",
      "session": "NORMAL",
      "price": "42.03",
      "duration": "DAY",
      "orderStrategyType": "SINGLE",
      "orderLegCollection": [
        {
          "instruction": "SELL",
          "quantity": 10,
          "instrument": {
            "symbol": "XYZ",
            "assetType": "EQUITY"
          }
        }
      ]
    }
  ]
}
```

### 5. One Cancels Other (OCO)
Sell 2 shares with limit at $45.97 OR stop-limit at $37.03/$37.00. One fills, other cancels.

```json
{
  "orderStrategyType": "OCO",
  "childOrderStrategies": [
    {
      "orderType": "LIMIT",
      "session": "NORMAL",
      "price": "45.97",
      "duration": "DAY",
      "orderStrategyType": "SINGLE",
      "orderLegCollection": [
        {
          "instruction": "SELL",
          "quantity": 2,
          "instrument": {
            "symbol": "XYZ",
            "assetType": "EQUITY"
          }
        }
      ]
    },
    {
      "orderType": "STOP_LIMIT",
      "session": "NORMAL",
      "price": "37.00",
      "stopPrice": "37.03",
      "duration": "DAY",
      "orderStrategyType": "SINGLE",
      "orderLegCollection": [
        {
          "instruction": "SELL",
          "quantity": 2,
          "instrument": {
            "symbol": "XYZ",
            "assetType": "EQUITY"
          }
        }
      ]
    }
  ]
}
```

### 6. One Triggers OCO
Buy 5 shares at $14.97. If filled, trigger OCO: sell at $15.27 limit OR $11.27 stop.

```json
{
  "orderStrategyType": "TRIGGER",
  "session": "NORMAL",
  "duration": "DAY",
  "orderType": "LIMIT",
  "price": 14.97,
  "orderLegCollection": [
    {
      "instruction": "BUY",
      "quantity": 5,
      "instrument": {
        "assetType": "EQUITY",
        "symbol": "XYZ"
      }
    }
  ],
  "childOrderStrategies": [
    {
      "orderStrategyType": "OCO",
      "childOrderStrategies": [
        {
          "orderStrategyType": "SINGLE",
          "session": "NORMAL",
          "duration": "GOOD_TILL_CANCEL",
          "orderType": "LIMIT",
          "price": 15.27,
          "orderLegCollection": [
            {
              "instruction": "SELL",
              "quantity": 5,
              "instrument": {
                "assetType": "EQUITY",
                "symbol": "XYZ"
              }
            }
          ]
        },
        {
          "orderStrategyType": "SINGLE",
          "session": "NORMAL",
          "duration": "GOOD_TILL_CANCEL",
          "orderType": "STOP",
          "stopPrice": 11.27,
          "orderLegCollection": [
            {
              "instruction": "SELL",
              "quantity": 5,
              "instrument": {
                "assetType": "EQUITY",
                "symbol": "XYZ"
              }
            }
          ]
        }
      ]
    }
  ]
}
```

### 7. Trailing Stop Order
Sell 10 shares with $10 trailing stop. Trail adjusts up with price, triggers market order on decline.

```json
{
  "complexOrderStrategyType": "NONE",
  "orderType": "TRAILING_STOP",
  "session": "NORMAL",
  "stopPriceLinkBasis": "BID",
  "stopPriceLinkType": "VALUE",
  "stopPriceOffset": 10,
  "duration": "DAY",
  "orderStrategyType": "SINGLE",
  "orderLegCollection": [
    {
      "instruction": "SELL",
      "quantity": 10,
      "instrument": {
        "symbol": "XYZ",
        "assetType": "EQUITY"
      }
    }
  ]
}
```

## Common Parameters

### Order Types
- `MARKET`: Execute at current market price
- `LIMIT`: Execute at specified price or better
- `STOP`: Execute market order when stop price reached
- `STOP_LIMIT`: Execute limit order when stop price reached
- `TRAILING_STOP`: Stop order that trails price movement
- `NET_DEBIT`: Multi-leg order with net debit price
- `NET_CREDIT`: Multi-leg order with net credit price

### Duration Types
- `DAY`: Order expires at end of trading day
- `GOOD_TILL_CANCEL`: Order remains active until filled or cancelled
- `IMMEDIATE_OR_CANCEL`: Fill immediately or cancel
- `FILL_OR_KILL`: Fill entire order immediately or cancel

### Session Types
- `NORMAL`: Regular trading hours
- `AM`: Pre-market session
- `PM`: After-market session
- `SEAMLESS`: All available sessions

### Order Strategy Types
- `SINGLE`: Single order
- `OCO`: One Cancels Other
- `TRIGGER`: One Triggers Another (OTA)

## Notes

- All price values should be strings in JSON
- Quantities are integers
- Options symbols must include proper formatting with spaces
- Complex strategies use `childOrderStrategies` arrays
- Trailing stops use `stopPriceOffset` for dollar amounts or percentages