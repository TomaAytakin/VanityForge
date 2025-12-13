import requests
import json

RPC_URL = "https://mainnet.helius-rpc.com/?api-key=03ad30eb-2c13-45e9-83bd-19187d14e21a"

def check_account(pubkey):
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getAccountInfo",
        "params": [
            pubkey,
            {"encoding": "base58"}
        ]
    }
    response = requests.post(RPC_URL, json=payload)
    print(f"Checking {pubkey}: {response.status_code}")
    data = response.json()
    if 'result' in data and data['result'] and data['result']['value']:
        print(f"FOUND! Owner: {data['result']['value']['owner']}")
    else:
        print("Not found or empty.")

print("Variant 1:")
check_account("7RT5xcbwSrXnRVBSMUqS78J3dinRGvSVEFeFXfdN3pL3")

print("\nVariant 2:")
check_account("RJG3kcwd5bGtA1cr529Set3KczBR5WfSY1oSaF32jhP")
