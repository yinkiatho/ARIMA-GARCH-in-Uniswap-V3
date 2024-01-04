from gql import gql, Client
from gql.transport.requests import RequestsHTTPTransport
import pandas as pd
from datetime import datetime, timedelta


def convert_to_unix(date_str):
    # Convert the date string to a datetime object
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')

    # Set the time to 8:00 AM
    date_obj = date_obj.replace(hour=8, minute=0, second=0, microsecond=0)

    # Calculate the Unix timestamp
    unix_timestamp = int(date_obj.timestamp())

    return unix_timestamp


def graph(network,Adress,fromdate):

    if network == 1:

        sample_transport=RequestsHTTPTransport(
        url='https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3',
        verify=True,
        retries=5,
        )
        client = Client(
        transport=sample_transport
        )


    print(fromdate)

    query = gql('''
    query ($fromdate: Int!)
    {
    poolHourDatas(where:{pool:"'''+str(Adress)+'''",periodStartUnix_gt:$fromdate},orderBy:periodStartUnix,orderDirection:desc,first:1000)
    {
    periodStartUnix
    liquidity
    high
    low
    pool{
        
        totalValueLockedUSD
        totalValueLockedToken1
        totalValueLockedToken0
        token0
            {decimals
            }
        token1
            {decimals
            }
        }
    close
    feeGrowthGlobal0X128
    feeGrowthGlobal1X128
    }
 
    }
    ''')
    params = {
    "fromdate": fromdate
    }

    response = client.execute(query,variable_values=params)
    dpd =pd.json_normalize(response['poolHourDatas'])
    dpd=dpd.astype(float)
    return dpd



def graphTwo(network, Adress, fromdate, todate):
    if network == 1:
        sample_transport = RequestsHTTPTransport(
            url='https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3',
            verify=True,
            retries=5,
        )
        client = Client(
            transport=sample_transport
        )

    # Initialize an empty DataFrame to store all data
    all_data = pd.DataFrame()

    # Initial value for skip to start fetching data
    skip = 0

    while True:
        query = gql('''
        query ($fromdate: Int!, $todate: Int!, $skip: Int)
        {
            poolHourDatas(where: {pool: "'''+str(Adress)+'''", periodStartUnix_gt: $fromdate, periodStartUnix_lt: $todate}, orderBy: periodStartUnix, orderDirection: desc, skip: $skip, first: 1000)
            {
            periodStartUnix
            liquidity
            high
            low
            pool{
        
                totalValueLockedUSD
                totalValueLockedToken1
                totalValueLockedToken0
                token0
                    {decimals
                    }
                token1
                    {decimals
                    }
                }
            close
            feeGrowthGlobal0X128
            feeGrowthGlobal1X128
            }
            }
        ''')

        params = {
            "fromdate": fromdate,
            "todate": todate,
            "skip": skip
        }

        response = client.execute(query, variable_values=params)

        # Extract data from the response
        pool_hour_datas = response.get('poolHourDatas', [])
        if not pool_hour_datas:
            break  # No more data to fetch

        # Extract and append data to the DataFrame
        chunk_data = pd.json_normalize(pool_hour_datas, sep='.')
        chunk_data = chunk_data.astype(float)
        all_data = pd.concat([all_data, chunk_data], ignore_index=True)

        # Update the skip for the next iteration
        skip += 1000  # Assuming you want to skip 1000 records at a time

    return all_data



if __name__ == "__main__":
    Adress= "0xcbcdf9626bc03e24f779434178a73a0b4bad62ed" # WBTC/WETH 
    dpd= graphTwo(1,Adress,convert_to_unix("2023-05-25"),convert_to_unix("2023-12-24"))
    print(dpd)
    
    
    
