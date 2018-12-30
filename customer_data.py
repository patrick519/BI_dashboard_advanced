# Max van Delft, Patrick t Jong
# April 2018
# run by 'python customer_dataV2.py'. execution takes about 3 minutes

# Costumer data before 2014:
# 1. Last transaction date                            V
# 2. First transaction date                           V
# 3. Average time between transactions                V
# 4. Lifespan                                         V
# 5. Number of transactions                           V
# 6. Spending behaviour (total spending price)        V
# 7. Maximal transaction price                        V
# 8. Average transaction price                        V
# 9. Countrycode                                      V
# 10-15 spending     last month, 2nd last month, etc  V
# 16-21 transactions last month, 2nd last month, etc  V
# 22-26 spending     last week , last 2 weeks  , etc  V
# 27-31 transactions last week , last 2 weeks  , etc  V

import datetime
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn import preprocessing
import time

print ("Started")
#Global variables
df = None
df1 = None
df2 = None
dfCustomer = None
df1ByAccountName = None

# Read data from sales.csv
def readdata():
    global df
    df = pd.read_csv('sales.csv', names=['saleId','saleDateTime','accountName','coins','currency','priceInCurrency','priceInEUR','methodId','ip','ipCountry'])
    df['saleDateTime'] = pd.to_datetime(df['saleDateTime'])
    return;


# Split dataset in data before and after 1 jan 2014
# We collect feature data for customers that joined before this time
def splitdata():
    global df, df1, df2
    df1 = df[df['saleDateTime'] <  pd.Timestamp('20140101')]
    df2 = df[df['saleDateTime'] >= pd.Timestamp('20140101')]
    return;

# Print dataset to the output CSV file
def printdata():
    global dfCustomer
    print ("printing sheet to csv")
    dfCustomer.to_csv(path_or_buf="customer_data.csv", index=False)
    return;

# Create features
def setfeatures():
    global df1, dfCustomer, df1ByAccountName
    print ("busy setting features")
    dfCustomer = pd.DataFrame()
    dfCustomer['accountName'] = df1['accountName']
    df1ByAccountName  = df1.groupby('accountName')
    # Source: https://stackoverflow.com/questions/22072943/faster-way-to-transform-group-with-mean-value-in-pandas/22073449
    dfCustomer['lastTransactionDate' ] = df1ByAccountName['saleDateTime'].transform('max') #1
    dfCustomer['firstTransactionDate'] = df1ByAccountName['saleDateTime'].transform('min') #2
    dfCustomer['lifeSpan'            ] = dfCustomer['lastTransactionDate' ] - dfCustomer['firstTransactionDate'] #3
    dfCustomer['numberOfTransactions'] = df1ByAccountName['accountName' ].transform('count') #4
    dfCustomer['avgTimeBetweenTransactions'] = dfCustomer.apply(lambda row: row.lifeSpan/row.numberOfTransactions, axis=1) #5
    dfCustomer['totalSpending'       ] = df1ByAccountName['priceInEUR'  ].transform('sum') #6

    dfCustomer['maxTransactionPrice' ] = df1ByAccountName['priceInEUR'  ].transform('max') #7
    dfCustomer['avgTransactionPrice' ] = df1ByAccountName['priceInEUR'  ].transform(np.mean) #8
    dfCustomer['country'             ] = df1['ipCountry']
    # Convert date to number of days:mins:secs (timedelta obtject) until 1 jan 2014
    dfCustomer['firstTransactionDate'] = pd.Timestamp('20140101') - dfCustomer['firstTransactionDate']
    dfCustomer['lastTransactionDate' ] = pd.Timestamp('20140101') - dfCustomer['lastTransactionDate' ]
    # convert timedelta to days
    dfCustomer['firstTransactionDate'] = dfCustomer['firstTransactionDate'].astype('timedelta64[D]') # extract days from timedelta
    dfCustomer['lastTransactionDate' ] = dfCustomer['lastTransactionDate' ].astype('timedelta64[D]')
    dfCustomer['lifeSpan'            ] = dfCustomer['lifeSpan'            ].astype('timedelta64[D]')
    dfCustomer['avgTimeBetweenTransactions'] = dfCustomer.apply(lambda row: row.lifeSpan/row.numberOfTransactions, axis=1)
    # Check for duplicates
    dfCustomer.drop_duplicates(subset=["accountName"],inplace=True)
    dfCustomer = dfCustomer.reset_index(drop=True)
    # Convert countries to country codes, i.e. the rank of a country in terms of the number of transactions. 0 => most transactions
    dfCountries = pd.DataFrame()
    dfCountries["country"] = df1["ipCountry"]
    dfCountries["numberOfTransactions"] = df1.groupby('ipCountry')['ipCountry'].transform('count')
    dfCountries.drop_duplicates(subset=["country"],inplace=True)
    dfCountries = dfCountries.sort_values(by=['numberOfTransactions'], ascending=False)
    dfCountries = dfCountries[pd.notnull(dfCountries["country"])]
    dfCountries = dfCountries.reset_index(drop=True)
    dfCountries["rank"] = range(0,len(dfCountries.index))
    #dfCountries = dfCountries.drop('numberOfTransactions', 1)
    #dfCustomer['countryCode'] = len(dfCountries.index)
    d = dict((name,None) for name in dfCountries["country"])
    d = defaultdict(lambda: len(dfCountries.index)+1, d)
    for index, row in dfCountries.iterrows():
        d[row["country"]] = row["rank"]

    # to set values in a whole column of a dataframe we set it equal to a list of values. The list must have the right length. This method is up to 50 times faster than using loc or ix
    # https://stackoverflow.com/questions/45795274/fast-way-of-populating-a-very-large-dataframe-with-values
    #print("Time setting the countrycodes directly into the dataframe:")
    #start = time.time()
    #for i in range(0,len(dfCustomer.index)) :
    #    dfCustomer.loc[i,'countryCode'] = d[dfCustomer.iloc[i]["country"]]
    #end = time.time()
    #print(end - start)

    #print("Time setting country of customers in a list and filling a list with countrycode:")
    listCountries = dfCustomer["country"]
    listCountryCode = []
    start = time.time()
    for i in range(0,len(dfCustomer.index)) :
        listCountryCode.append(d[listCountries[i]])
    dfCustomer['countryCode'] = listCountryCode #9
    end = time.time()
    #print(end - start)

    # Clean up old country column
    del dfCustomer['country']

    # used for identification of accounts having purchased in the last month, 2nd last month, etc
    dfLastMonth    = df1[df1['saleDateTime'] >= pd.Timestamp('20131201')]
    dfLastMonth   ['totalSpending'       ] = dfLastMonth   .groupby('accountName')['priceInEUR' ].transform('sum')
    dfLastMonth   ['numberOfTransactions'] = dfLastMonth   .groupby('accountName')['accountName'].transform('count')
    df2ndLastMonth = df1[df1['saleDateTime'] >= pd.Timestamp('20131101')]
    df2ndLastMonth = df2ndLastMonth[df2ndLastMonth['saleDateTime'] < pd.Timestamp('20131201')]
    df2ndLastMonth['totalSpending'       ] = df2ndLastMonth.groupby('accountName')['priceInEUR' ].transform('sum')
    df2ndLastMonth['numberOfTransactions'] = df2ndLastMonth.groupby('accountName')['accountName'].transform('count')
    df3rdLastMonth = df1[df1['saleDateTime'] >= pd.Timestamp('20131001')]
    df3rdLastMonth = df3rdLastMonth[df3rdLastMonth['saleDateTime'] < pd.Timestamp('20131101')]
    df3rdLastMonth['totalSpending'       ] = df3rdLastMonth.groupby('accountName')['priceInEUR' ].transform('sum')
    df3rdLastMonth['numberOfTransactions'] = df3rdLastMonth.groupby('accountName')['accountName'].transform('count')
    df4thLastMonth = df1[df1['saleDateTime'] >= pd.Timestamp('20130901')]
    df4thLastMonth = df4thLastMonth[df4thLastMonth['saleDateTime'] < pd.Timestamp('20131001')]
    df4thLastMonth['totalSpending'       ] = df4thLastMonth.groupby('accountName')['priceInEUR' ].transform('sum')
    df4thLastMonth['numberOfTransactions'] = df4thLastMonth.groupby('accountName')['accountName'].transform('count')
    df5thLastMonth = df1[df1['saleDateTime'] >= pd.Timestamp('20130801')]
    df5thLastMonth = df5thLastMonth[df5thLastMonth['saleDateTime'] < pd.Timestamp('20130901')]
    df5thLastMonth['totalSpending'       ] = df5thLastMonth.groupby('accountName')['priceInEUR' ].transform('sum')
    df5thLastMonth['numberOfTransactions'] = df5thLastMonth.groupby('accountName')['accountName'].transform('count')
    df6thLastMonth = df1[df1['saleDateTime'] >= pd.Timestamp('20130701')]
    df6thLastMonth = df6thLastMonth[df6thLastMonth['saleDateTime'] < pd.Timestamp('20130801')]
    df6thLastMonth['totalSpending'       ] = df6thLastMonth.groupby('accountName')['priceInEUR' ].transform('sum')
    df6thLastMonth['numberOfTransactions'] = df6thLastMonth.groupby('accountName')['accountName'].transform('count')
    dfLastMonth   .drop_duplicates(subset=["accountName"],inplace=True)
    df2ndLastMonth.drop_duplicates(subset=["accountName"],inplace=True)
    df3rdLastMonth.drop_duplicates(subset=["accountName"],inplace=True)
    df4thLastMonth.drop_duplicates(subset=["accountName"],inplace=True)
    df5thLastMonth.drop_duplicates(subset=["accountName"],inplace=True)
    df6thLastMonth.drop_duplicates(subset=["accountName"],inplace=True)

    dfCustomer["spendingLastMonth"   ] = 0 #10
    dfCustomer["spending2ndLastMonth"] = 0 #11
    dfCustomer["spending3rdLastMonth"] = 0 #12
    dfCustomer["spending4thLastMonth"] = 0 #13
    dfCustomer["spending5thLastMonth"] = 0 #14
    dfCustomer["spending6thLastMonth"] = 0 #15

    dfCustomer["transactionsLastMonth"   ] = 0 #16
    dfCustomer["transactions2ndLastMonth"] = 0 #17
    dfCustomer["transactions3rdLastMonth"] = 0 #18
    dfCustomer["transactions4thLastMonth"] = 0 #19
    dfCustomer["transactions5thLastMonth"] = 0 #20
    dfCustomer["transactions6thLastMonth"] = 0 #21

    d1 = dict((name,0) for name in dfLastMonth   ["accountName"])
    d2 = dict((name,0) for name in df2ndLastMonth["accountName"])
    d3 = dict((name,0) for name in df3rdLastMonth["accountName"])
    d4 = dict((name,0) for name in df4thLastMonth["accountName"])
    d5 = dict((name,0) for name in df5thLastMonth["accountName"])
    d6 = dict((name,0) for name in df6thLastMonth["accountName"])
    d1 = defaultdict(lambda: 0, d1)
    d2 = defaultdict(lambda: 0, d2)
    d3 = defaultdict(lambda: 0, d3)
    d4 = defaultdict(lambda: 0, d4)
    d5 = defaultdict(lambda: 0, d5)
    d6 = defaultdict(lambda: 0, d6)

    for index, row in dfLastMonth.iterrows():
        d1[row["accountName"]] = row["totalSpending"]
    for index, row in df2ndLastMonth.iterrows():
        d2[row["accountName"]] = row["totalSpending"]
    for index, row in df3rdLastMonth.iterrows():
        d3[row["accountName"]] = row["totalSpending"]
    for index, row in df4thLastMonth.iterrows():
        d4[row["accountName"]] = row["totalSpending"]
    for index, row in df5thLastMonth.iterrows():
        d5[row["accountName"]] = row["totalSpending"]
    for index, row in df6thLastMonth.iterrows():
        d6[row["accountName"]] = row["totalSpending"]


    listCustomerAccountName = dfCustomer["accountName"]

    listSpendingLastMonth    = [0] * len(dfCustomer)
    listSpending2ndLastMonth = [0] * len(dfCustomer)
    listSpending3rdLastMonth = [0] * len(dfCustomer)
    listSpending4thLastMonth = [0] * len(dfCustomer)
    listSpending5thLastMonth = [0] * len(dfCustomer)
    listSpending6thLastMonth = [0] * len(dfCustomer)
    for i in range(0,len(dfCustomer.index)) :
        listSpendingLastMonth[i]    = d1[listCustomerAccountName[i]]
        listSpending2ndLastMonth[i] = d2[listCustomerAccountName[i]]
        listSpending3rdLastMonth[i] = d3[listCustomerAccountName[i]]
        listSpending4thLastMonth[i] = d4[listCustomerAccountName[i]]
        listSpending5thLastMonth[i] = d5[listCustomerAccountName[i]]
        listSpending6thLastMonth[i] = d6[listCustomerAccountName[i]]
    dfCustomer['spendingLastMonth'   ] = listSpendingLastMonth
    dfCustomer['spending2ndLastMonth'] = listSpending2ndLastMonth
    dfCustomer['spending3rdLastMonth'] = listSpending3rdLastMonth
    dfCustomer['spending4thLastMonth'] = listSpending4thLastMonth
    dfCustomer['spending5thLastMonth'] = listSpending5thLastMonth
    dfCustomer['spending6thLastMonth'] = listSpending6thLastMonth


    for index, row in dfLastMonth.iterrows():
        d1[row["accountName"]] = row["numberOfTransactions"]
    for index, row in df2ndLastMonth.iterrows():
        d2[row["accountName"]] = row["numberOfTransactions"]
    for index, row in df3rdLastMonth.iterrows():
        d3[row["accountName"]] = row["numberOfTransactions"]
    for index, row in df4thLastMonth.iterrows():
        d4[row["accountName"]] = row["numberOfTransactions"]
    for index, row in df5thLastMonth.iterrows():
        d5[row["accountName"]] = row["numberOfTransactions"]
    for index, row in df6thLastMonth.iterrows():
        d6[row["accountName"]] = row["numberOfTransactions"]

    listTransactionsLastMonth    = [0] * len(dfCustomer) # list of number of transaction over the last month per customer
    listTransactions2ndLastMonth = [0] * len(dfCustomer)
    listTransactions3rdLastMonth = [0] * len(dfCustomer)
    listTransactions4thLastMonth = [0] * len(dfCustomer)
    listTransactions5thLastMonth = [0] * len(dfCustomer)
    listTransactions6thLastMonth = [0] * len(dfCustomer)
    for i in range(0,len(dfCustomer.index)) :
        listTransactionsLastMonth[i]    = d1[listCustomerAccountName[i]]
        listTransactions2ndLastMonth[i] = d2[listCustomerAccountName[i]]
        listTransactions3rdLastMonth[i] = d3[listCustomerAccountName[i]]
        listTransactions4thLastMonth[i] = d4[listCustomerAccountName[i]]
        listTransactions5thLastMonth[i] = d5[listCustomerAccountName[i]]
        listTransactions6thLastMonth[i] = d6[listCustomerAccountName[i]]
    dfCustomer['transactionsLastMonth'   ] = listTransactionsLastMonth
    dfCustomer['transactions2ndLastMonth'] = listTransactions2ndLastMonth
    dfCustomer['transactions3rdLastMonth'] = listTransactions3rdLastMonth
    dfCustomer['transactions4thLastMonth'] = listTransactions4thLastMonth
    dfCustomer['transactions5thLastMonth'] = listTransactions5thLastMonth
    dfCustomer['transactions6thLastMonth'] = listTransactions6thLastMonth



    # The dataframes below are used for identification of accounts having purchased in the last week, last 2 weeks, etc
    dfLastWeek    = df1[df1['saleDateTime'] >= pd.Timestamp('20131224')]
    dfLastWeek   ['totalSpending'       ] = dfLastWeek   .groupby('accountName')['priceInEUR' ].transform('sum')
    dfLastWeek   ['numberOfTransactions'] = dfLastWeek   .groupby('accountName')['accountName'].transform('count')
    dfLast2Weeks  = df1[df1['saleDateTime'] >= pd.Timestamp('20131217')]
    dfLast2Weeks ['totalSpending'       ] = dfLast2Weeks .groupby('accountName')['priceInEUR' ].transform('sum')
    dfLast2Weeks ['numberOfTransactions'] = dfLast2Weeks .groupby('accountName')['accountName'].transform('count')
    dfLast2Months = df1[df1['saleDateTime'] >= pd.Timestamp('20131101')] # The total sales per customer for the last month can be found in 'dflast'
    dfLast2Months['totalSpending'       ] = dfLast2Months.groupby('accountName')['priceInEUR' ].transform('sum')
    dfLast2Months['numberOfTransactions'] = dfLast2Months.groupby('accountName')['accountName'].transform('count')
    dfLast4Months = df1[df1['saleDateTime'] >= pd.Timestamp('20130901')]
    dfLast4Months['totalSpending'       ] = dfLast4Months.groupby('accountName')['priceInEUR' ].transform('sum')
    dfLast4Months['numberOfTransactions'] = dfLast4Months.groupby('accountName')['accountName'].transform('count')
    dfLast8Months = df1[df1['saleDateTime'] >= pd.Timestamp('20130401')]
    dfLast8Months['totalSpending'       ] = dfLast8Months.groupby('accountName')['priceInEUR' ].transform('sum')
    dfLast8Months['numberOfTransactions'] = dfLast8Months.groupby('accountName')['accountName'].transform('count')
    dfLastWeek   .drop_duplicates(subset=["accountName"],inplace=True)
    dfLast2Weeks .drop_duplicates(subset=["accountName"],inplace=True)
    dfLast2Months.drop_duplicates(subset=["accountName"],inplace=True)
    dfLast4Months.drop_duplicates(subset=["accountName"],inplace=True)
    dfLast8Months.drop_duplicates(subset=["accountName"],inplace=True)

    dfCustomer["spendingLastWeek"   ] = 0 #22
    dfCustomer["spendingLast2Weeks" ] = 0 #23
    dfCustomer["spendingLast2Months"] = 0 #24
    dfCustomer["spendingLast4Months"] = 0 #25
    dfCustomer["spendingLast8Months"] = 0 #26

    dfCustomer["transactionsLastWeek"   ] = 0 #27
    dfCustomer["transactionsLast2Weeks" ] = 0 #28
    dfCustomer["transactionsLast2Months"] = 0 #29
    dfCustomer["transactionsLast4Months"] = 0 #30
    dfCustomer["transactionsLast8Months"] = 0 #31

    d1 = dict((name,0) for name in dfLastWeek   ["accountName"])
    d2 = dict((name,0) for name in dfLast2Weeks ["accountName"])
    d3 = dict((name,0) for name in dfLast2Months["accountName"])
    d4 = dict((name,0) for name in dfLast4Months["accountName"])
    d5 = dict((name,0) for name in dfLast8Months["accountName"])
    d1 = defaultdict(lambda: 0, d1)
    d2 = defaultdict(lambda: 0, d2)
    d3 = defaultdict(lambda: 0, d3)
    d4 = defaultdict(lambda: 0, d4)
    d5 = defaultdict(lambda: 0, d5)

    for index, row in dfLastWeek.iterrows():
        d1[row["accountName"]] = row["totalSpending"]
    for index, row in dfLast2Weeks.iterrows():
        d2[row["accountName"]] = row["totalSpending"]
    for index, row in dfLast2Months.iterrows():
        d3[row["accountName"]] = row["totalSpending"]
    for index, row in dfLast4Months.iterrows():
        d4[row["accountName"]] = row["totalSpending"]
    for index, row in dfLast8Months.iterrows():
        d5[row["accountName"]] = row["totalSpending"]

    listSpendingLastWeek    = [0] * len(dfCustomer)
    listSpendingLast2Weeks  = [0] * len(dfCustomer)
    listSpendingLast2Months = [0] * len(dfCustomer)
    listSpendingLast4Months = [0] * len(dfCustomer)
    listSpendingLast8Months = [0] * len(dfCustomer)
    for i in range(0,len(dfCustomer.index)) :
        listSpendingLastWeek[i]    = d1[listCustomerAccountName[i]]
        listSpendingLast2Weeks[i]  = d2[listCustomerAccountName[i]]
        listSpendingLast2Months[i] = d3[listCustomerAccountName[i]]
        listSpendingLast4Months[i] = d4[listCustomerAccountName[i]]
        listSpendingLast8Months[i] = d5[listCustomerAccountName[i]]
    dfCustomer['spendingLastWeek'   ] = listSpendingLastWeek
    dfCustomer['spendingLast2Weeks' ] = listSpendingLast2Weeks
    dfCustomer['spendingLast2Months'] = listSpendingLast2Months
    dfCustomer['spendingLast4Months'] = listSpendingLast4Months
    dfCustomer['spendingLast8Months'] = listSpendingLast8Months


    for index, row in dfLastWeek.iterrows():
        d1[row["accountName"]] = row["numberOfTransactions"]
    for index, row in dfLast2Weeks.iterrows():
        d2[row["accountName"]] = row["numberOfTransactions"]
    for index, row in dfLast2Months.iterrows():
        d3[row["accountName"]] = row["numberOfTransactions"]
    for index, row in dfLast4Months.iterrows():
        d4[row["accountName"]] = row["numberOfTransactions"]
    for index, row in dfLast8Months.iterrows():
        d5[row["accountName"]] = row["numberOfTransactions"]


    listTransactionsLastWeek    = [0] * len(dfCustomer)
    listTransactionsLast2Weeks  = [0] * len(dfCustomer)
    listTransactionsLast2Months = [0] * len(dfCustomer)
    listTransactionsLast4Months = [0] * len(dfCustomer)
    listTransactionsLast8Months = [0] * len(dfCustomer)
    for i in range(0,len(dfCustomer.index)) :
        listTransactionsLastWeek[i]    = d1[listCustomerAccountName[i]]
        listTransactionsLast2Weeks[i]  = d2[listCustomerAccountName[i]]
        listTransactionsLast2Months[i] = d3[listCustomerAccountName[i]]
        listTransactionsLast4Months[i] = d4[listCustomerAccountName[i]]
        listTransactionsLast8Months[i] = d5[listCustomerAccountName[i]]
    dfCustomer['transactionsLastWeek'   ] = listTransactionsLastWeek
    dfCustomer['transactionsLast2Weeks' ] = listTransactionsLast2Weeks
    dfCustomer['transactionsLast2Months'] = listTransactionsLast2Months
    dfCustomer['transactionsLast4Months'] = listTransactionsLast4Months
    dfCustomer['transactionsLast8Months'] = listTransactionsLast8Months

    return;



def settarget():
    global df1, df2, dfCustomer
    print ("busy setting target")
    dfCustomer['repurchase'] = False
    dfBefore = pd.DataFrame(); dfBefore["accountName"] = df1["accountName"]
    dfAfter  = pd.DataFrame(); dfAfter ["accountName"] = df2["accountName"]
    # dfIntersect will consist of the accountNames that have purchased before 1 jan 2014 as well as after.
    dfIntersect = pd.merge(dfBefore,dfAfter, how = 'inner', on=['accountName','accountName'])
    dfIntersect.drop_duplicates(subset=["accountName"],inplace=True)
    # setting hash table

    d = dict((name,1) for name in dfIntersect["accountName"])
    d = defaultdict(lambda: -1, d)

    #start = time.time()
    #for i in range(0,len(dfCustomer.index)):
    #    if (d[dfCustomer.iloc[i]["accountName"]] == 1):
    #        dfCustomer.loc[i,'repurchase'] = True
    #end = time.time()
    #print(end - start)

    listCustomerAccountName = dfCustomer["accountName"]
    listRepurchase = [False]*len(dfCustomer)
    start = time.time()
    for i in range(0,len(dfCustomer.index)):
        if (d[listCustomerAccountName[i]] == 1):
            listRepurchase[i] = True
    dfCustomer['repurchase'] = listRepurchase
    end = time.time()
    #print(end - start)

    return;


#Main
readdata()
splitdata()
start = time.time()
setfeatures()
end = time.time()
print("Elapsed time setting the features: ", end - start, "s\n")
settarget()
printdata()

print ("Completed.")