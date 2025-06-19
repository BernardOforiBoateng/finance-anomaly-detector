# Required CSV Format

Your bank CSV should have these columns (common names):

## Option 1: Basic Format
```
Date,Description,Amount,Balance
2024-01-15,AMAZON MARKETPLACE,-52.99,1547.23
2024-01-14,STARBUCKS #1234,-4.75,1600.22
```

## Option 2: Detailed Format (Better)
```
Date,Description,Amount,Category,Account,Type
2024-01-15,AMAZON MARKETPLACE,-52.99,Shopping,Checking,Debit
2024-01-14,SALARY DEPOSIT,3500.00,Income,Checking,Credit
```

## Most Banks Provide:
- **Chase**: Date, Description, Amount, Type, Balance
- **Bank of America**: Date, Description, Amount, Running Balance
- **Wells Fargo**: Date, Amount, Description, Ending Daily Balance
- **Capital One**: Transaction Date, Posted Date, Description, Category, Debit, Credit

## To Export Your Data:
1. Log into your online banking
2. Go to transaction history
3. Select date range (3-6 months is ideal)
4. Export as CSV
5. Save as `transactions.csv`