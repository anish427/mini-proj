import requests, json, sys

img = sys.argv[1] if len(sys.argv) > 1 else "../1.jpg"
r = requests.post(
    "http://127.0.0.1:5000/api/detect-image",
    files={"file": (img.split("/")[-1], open(img, "rb"), "image/jpeg")}
)
d = r.json()
print(f"\n=== {img} ===")
print(f"VERDICT: {d['label']}  ({d['confidence']*100:.1f}% confidence)")
print(f"  prob_fake={d['prob_fake']:.4f}  prob_real={d['prob_real']:.4f}")
print(f"  engine: {d.get('backend','?')}")
signals = d.get("signals", {})
for k, v in signals.items():
    score = v.get("prob_fake", v.get("score", "?"))
    if isinstance(score, float):
        score = f"{score:.4f}"
    print(f"  {k:12s}: {score}  ({v['name']})")
