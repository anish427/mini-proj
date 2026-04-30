import requests, json, sys

images = [
    ("../1.jpg", "Real photo 1 (woman face)"),
    ("../2.jpg", "Real photo 2 (man with cap)"),
    ("../ai_test.png", "AI-generated (diffusion portrait)"),
    ("../gan_face_test.png", "AI-generated (headshot face)"),
]

for path, desc in images:
    try:
        r = requests.post(
            "http://127.0.0.1:5000/api/detect-image",
            files={"file": (path.split("/")[-1], open(path, "rb"), "image/jpeg")}
        )
        d = r.json()
        verdict = d["label"]
        conf = d["confidence"] * 100
        pf = d["prob_fake"]
        signals = d.get("signals", {})
        sig_str = " | ".join(
            f"{k}={v.get('prob_fake', v.get('score', '?')):.2f}"
            for k, v in signals.items()
            if isinstance(v.get('prob_fake', v.get('score')), float)
        )
        status = "✓" if (("Real" in desc and verdict == "Real") or ("AI" in desc and verdict == "Fake")) else "✗ WRONG"
        print(f"{status} {desc:40s} → {verdict:5s} {conf:5.1f}%  (fake={pf:.3f})  [{sig_str}]")
    except Exception as e:
        print(f"ERROR {desc}: {e}")
