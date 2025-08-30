# test_api_connection.py
import asyncio
from binance import AsyncClient
from dotenv import load_dotenv
import os


async def test_connection():
    load_dotenv()

    client = await AsyncClient.create(
        api_key=os.getenv('API_BINANCE_API_KEY'),
        api_secret=os.getenv('API_BINANCE_API_SECRET'),
        testnet=True
    )

    try:
        # Test ping
        await client.ping()
        print("✅ Connection successful!")

        # Test getting account info
        account = await client.get_account()
        print(f"✅ Account access works!")

        # Test getting market data
        ticker = await client.get_ticker(symbol='BTCUSDT')
        print(f"✅ Market data access works! BTC Price: ${ticker['lastPrice']}")

    except Exception as e:
        print(f"❌ Connection failed: {e}")
    finally:
        await client.close_connection()


asyncio.run(test_connection())