import requests, json
evs = requests.get("https://gamma-api.polymarket.com/events",
                   params={"active":"true","closed":"false","limit":30,"order":"volume24hr","ascending":"false"},
                   timeout=30).json()
for ev in evs:
    for m in ev.get("markets", []):
        if _num := (m.get("rewardsMaxSpread") or m.get("clobRewards")):
            print(json.dumps({"q": m.get("question","")[:50],
                              "rewardsMaxSpread": m.get("rewardsMaxSpread"),
                              "rewardsMinSize": m.get("rewardsMinSize"),
                              "clobRewards": m.get("clobRewards")}, indent=2))
            break
    else:
        continue
    break