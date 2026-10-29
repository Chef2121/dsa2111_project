"""Synthetic Yelp-format data generator.

The real Yelp Open Dataset (~9 GB) must be downloaded manually after accepting
Yelp's licence, so this module produces a *drop-in substitute* in exactly the
same JSON-lines schema (business / review / checkin / tip files). That lets
the full pipeline, the tests and the dashboard run anywhere in minutes.

The simulator is not random text: each restaurant has latent quality for every
aspect that evolves over time. Some restaurants go through "deterioration
episodes" (e.g. staff turnover -> service drops) that ramp up gradually and may
recover. Reviewers mention aspects more often when they are bad, early
degradation produces *mild* complaints that barely move the star rating, and
footfall (check-ins) reacts with a lag. This reproduces the core phenomenon
Radview targets: text complaints lead the star rating.

Because we know the latent truth, the generator also writes:
  * ``_ground_truth_aspects.jsonl`` - per-review aspect labels, used as the
    "hand-labelled" M2 test set when running on synthetic data;
  * ``_ground_truth_episodes.json`` - injected episodes, for analysis.

Usage:  python -m radview.m1_ingest.synthetic --restaurants 220
"""
from __future__ import annotations

import argparse
import json
import string
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from radview.config import ASPECTS, RANDOM_SEED, RAW_DIR

# --------------------------------------------------------------------------
# Text banks. A mix of phrasings that hit the seed lexicon and phrasings that
# do not (implicit aspects) so that weak supervision is genuinely noisy.
# --------------------------------------------------------------------------
PHRASES: dict[str, dict[str, list[str]]] = {
    "food": {
        "pos": [
            "The {dish} was absolutely delicious.", "Every dish we tried was full of flavor.",
            "The {dish} tasted incredibly fresh.", "Portions were generous and everything was cooked perfectly.",
            "I could eat the {dish} here every single day.", "The {dish} was seasoned just right.",
            "Honestly the best {dish} I have had in this city.", "We cleaned our plates, the {dish} was that good.",
            "The kitchen clearly knows what it is doing.", "The {dish} came out hot and tasty.",
        ],
        "neg": [
            "The {dish} was cold and bland.", "My {dish} was overcooked and dry.",
            "The {dish} tasted like it had been sitting under a heat lamp for an hour.",
            "Found a hair in my {dish}, completely lost my appetite.",
            "The {dish} was inedible, we sent it back.", "Everything on the plate was greasy and soggy.",
            "The {dish} was undercooked in the middle.", "Food quality has really gone downhill.",
            "My partner's {dish} arrived burnt on one side.", "The {dish} was so salty we could not finish it.",
        ],
        "mild": [
            "The {dish} was a little lukewarm this time.", "The {dish} was fine but not as good as it used to be.",
            "Fries were slightly soggy tonight.", "The {dish} needed a bit more seasoning than before.",
            "The {dish} seemed smaller than I remember.", "Not quite the {dish} I remember from last year.",
        ],
    },
    "service": {
        "pos": [
            "Our server was friendly and attentive.", "The staff made us feel really welcome.",
            "Service was quick even though it was packed.", "Our waitress checked on us at just the right times.",
            "The host found us a table right away.", "The team here goes above and beyond.",
            "Drinks were refilled before we even asked.", "Everyone working here was polite and helpful.",
        ],
        "neg": [
            "Our server was rude and dismissive.", "We waited 45 minutes for our food.",
            "Nobody came to take our order for twenty minutes.", "The staff ignored us the entire night.",
            "We had to flag someone down twice just to get water.", "They got our order wrong and never apologized.",
            "The manager rolled his eyes when we asked about the delay.", "Service was painfully slow.",
            "Our drinks never arrived and the check took forever.", "We were seated and then forgotten about.",
        ],
        "mild": [
            "Service was a bit slower than usual.", "Our server seemed distracted tonight.",
            "Took a while to get the check.", "Had to ask twice for more napkins.",
            "Staff seemed stretched thin this visit.",
        ],
    },
    "cleanliness": {
        "pos": [
            "The place was spotless.", "Bathrooms were clean and well stocked.",
            "You can tell they take hygiene seriously.", "Tables were wiped down as soon as people left.",
            "Very tidy dining room.",
        ],
        "neg": [
            "The tables were sticky and the floor was filthy.", "The bathroom was disgusting and smelled awful.",
            "Saw a cockroach crawling near the counter.", "There were crumbs all over the booth and the floor felt tacky.",
            "Our glasses had lipstick marks on them.", "Flies everywhere around the food station.",
            "The restroom had no soap and the trash was overflowing.",
        ],
        "mild": [
            "The table was a little sticky when we sat down.", "Restroom could use more attention.",
            "Floor near our table was not the cleanest.", "Menus felt a bit grimy.",
        ],
    },
    "price": {
        "pos": [
            "Great value for the money.", "Prices are very reasonable for the quality.",
            "Affordable and filling.", "Totally worth the price.", "Happy hour deals are fantastic.",
        ],
        "neg": [
            "Way overpriced for what you get.", "Paid almost $60 for two plates, which felt steep.",
            "Prices went up again and portions got smaller.", "Not worth the money at all.",
            "They charged us for bread without telling us.", "A $22 burger should be much better than this.",
        ],
        "mild": [
            "Prices have crept up a bit lately.", "A little pricey for lunch.",
            "Slightly expensive for the portion size.",
        ],
    },
    "ambience": {
        "pos": [
            "Cozy atmosphere with great lighting.", "Lovely decor and relaxed vibe.",
            "Perfect spot for a date night.", "The patio is charming in the evening.",
            "Nice background music, not too loud.",
        ],
        "neg": [
            "Music was so loud we could not hear each other.", "Cramped seating, we were practically on top of the next table.",
            "The dining room was freezing.", "It felt chaotic and noisy the whole time.",
            "Lighting was so dark we needed our phone flashlights to read the menu.",
        ],
        "mild": [
            "A bit noisy on a Friday night.", "Seating is a little tight.", "Room was slightly stuffy.",
        ],
    },
}

OPENERS = [
    "Came here with my family on a Saturday night.", "Stopped by for a quick lunch.",
    "Been coming here for a couple of years now.", "First time trying this place.",
    "Visited with some coworkers after work.", "Ordered dinner here for my birthday.",
    "We were in the area and decided to try it.", "Back again after a few months away.",
]
FILLERS = [
    "We ordered the {dish} and the {dish2}.", "Parking was easy to find.", "It was pretty busy for a weekday.",
    "We sat by the window.", "They have a decent drinks list.", "My friend recommended this spot.",
]
CLOSERS = {
    "high": ["Will definitely be back.", "Highly recommend!", "Can't wait to come again.", "Five stars from us."],
    "mid": ["It was okay overall.", "Might give it another try.", "Decent but nothing special."],
    "low": ["Won't be returning.", "Not sure we will come back.", "Save your money and go elsewhere.", "Really let down."],
}

CUISINES = {
    "Italian": ["lasagna", "carbonara", "margherita pizza", "risotto", "tiramisu"],
    "Mexican": ["carnitas tacos", "burrito", "enchiladas", "guacamole", "churros"],
    "American (New)": ["burger", "fried chicken", "mac and cheese", "steak frites", "wings"],
    "Japanese": ["ramen", "salmon sushi", "katsu curry", "gyoza", "tempura"],
    "Chinese": ["dumplings", "mapo tofu", "fried rice", "kung pao chicken", "noodles"],
    "Thai": ["pad thai", "green curry", "tom yum soup", "basil chicken", "mango sticky rice"],
    "Indian": ["butter chicken", "biryani", "garlic naan", "tikka masala", "samosas"],
    "Breakfast & Brunch": ["pancakes", "eggs benedict", "french toast", "omelette", "avocado toast"],
}
NAME_A = ["Golden", "Rustic", "Blue", "Little", "Urban", "Lucky", "Olive", "Copper", "Saffron", "Maple",
          "Red Lantern", "Harbor", "Velvet", "Sunny", "Old Town", "Fig", "Silver", "Wild Thyme", "Juniper", "Corner"]
NAME_B = ["Bistro", "Kitchen", "Grill", "Diner", "Noodle House", "Taqueria", "Trattoria", "Cafe", "Eatery",
          "Smokehouse", "Table", "Social", "Canteen", "Brasserie"]
CITIES = {"Philadelphia": "PA", "Tampa": "FL", "Indianapolis": "IN", "Nashville": "TN", "Tucson": "AZ", "New Orleans": "LA"}
NON_RESTAURANT_CATS = ["Hair Salons, Beauty & Spas", "Auto Repair, Automotive", "Dentists, Health & Medical"]

SALIENCE = {"food": 0.85, "service": 0.55, "cleanliness": 0.15, "price": 0.25, "ambience": 0.25}
STAR_WEIGHT = {"food": 1.2, "service": 1.0, "cleanliness": 1.1, "price": 0.6, "ambience": 0.5}
EPISODE_ASPECT_P = {"food": 0.3, "service": 0.32, "cleanliness": 0.14, "price": 0.14, "ambience": 0.1}

START, END = pd.Period("2019-01", "M"), pd.Period("2024-12", "M")


def _sigmoid(x: float) -> float:
    """Logistic function."""
    return 1.0 / (1.0 + np.exp(-x))


def _rand_id(rng: np.random.Generator, n: int = 22) -> str:
    """Random Yelp-style id of ``n`` characters."""
    alphabet = np.array(list(string.ascii_letters + string.digits + "-_"))
    return "".join(rng.choice(alphabet, n))


def _episode_effect(ep: dict, t: int) -> float:
    """Signed quality shift of one episode at month index t (negative = worse)."""
    if t < ep["onset"]:
        return 0.0
    ramp = min(1.0, (t - ep["onset"] + 1) / ep["ramp"])
    eff = ramp
    if ep["recover_at"] is not None and t >= ep["recover_at"]:
        eff = max(0.0, 1.0 - (t - ep["recover_at"] + 1) / ep["recover_len"])
    return ep["sign"] * ep["magnitude"] * eff


def generate(n_restaurants: int = 220, seed: int = RANDOM_SEED, out_dir=RAW_DIR) -> dict:
    """Generate the four Yelp files plus ground truth into ``out_dir``."""
    rng = np.random.default_rng(seed)
    months = pd.period_range(START, END, freq="M")
    n_months = len(months)

    users = [_rand_id(rng) for _ in range(15000)]
    leniency = rng.normal(0, 0.35, len(users))

    businesses, reviews, truth, checkins, tips, episodes_out = [], [], [], [], [], []
    used_names: set[str] = set()

    n_other = max(1, n_restaurants // 10)
    for b in range(n_restaurants + n_other):
        is_restaurant = b < n_restaurants
        bid = _rand_id(rng)
        city = rng.choice(list(CITIES))
        cuisine = rng.choice(list(CUISINES))
        while True:
            name = f"{rng.choice(NAME_A)} {rng.choice(NAME_B)}"
            if name not in used_names or len(used_names) > 250:
                break
        if name in used_names:
            name = f"{name} {city}"
        used_names.add(name)
        categories = f"Restaurants, {cuisine}, Food" if is_restaurant else rng.choice(NON_RESTAURANT_CATS)

        open_idx = int(rng.integers(0, 18))
        base_q = {a: rng.normal(0.95, 0.45) for a in ASPECTS}
        lam0 = float(np.clip(rng.lognormal(np.log(6.0), 0.45), 2, 22))

        # ---- deterioration / improvement episodes ------------------------
        eps = []
        n_eps = rng.choice([0, 1, 2], p=[0.35, 0.45, 0.2])
        for _ in range(n_eps):
            aspect = rng.choice(ASPECTS, p=list(EPISODE_ASPECT_P.values()))
            onset = int(rng.integers(open_idx + 14, n_months - 2))
            ramp = int(rng.integers(3, 8))
            recover = rng.random() < 0.5
            recover_at = onset + ramp + int(rng.integers(2, 8)) if recover else None
            eps.append({"aspect": aspect, "onset": onset, "ramp": ramp, "sign": -1.0,
                        "magnitude": float(rng.uniform(1.3, 2.8)), "recover_at": recover_at,
                        "recover_len": int(rng.integers(4, 9))})
        if rng.random() < 0.15:
            eps.append({"aspect": rng.choice(ASPECTS), "onset": int(rng.integers(open_idx + 6, n_months - 6)),
                        "ramp": 4, "sign": 1.0, "magnitude": float(rng.uniform(0.5, 1.0)),
                        "recover_at": None, "recover_len": 1})
        # market-wide price pressure in 2022 (inflation) - small, affects everyone
        eps.append({"aspect": "price", "onset": months.get_loc(pd.Period("2022-03", "M")), "ramp": 9,
                    "sign": -1.0, "magnitude": 0.35, "recover_at": None, "recover_len": 1})

        if is_restaurant:
            for ep in eps:
                if ep["magnitude"] > 1.0:
                    episodes_out.append({"business_id": bid, "aspect": ep["aspect"], "sign": ep["sign"],
                                         "onset_month": str(months[ep["onset"]]), "ramp_months": ep["ramp"],
                                         "magnitude": round(ep["magnitude"], 2),
                                         "recover_month": str(months[ep["recover_at"]]) if ep["recover_at"] and ep["recover_at"] < n_months else None})

        drift = {a: 0.0 for a in ASPECTS}
        dishes = CUISINES[cuisine]
        star_sum, n_rev = 0.0, 0
        checkin_dates: list[str] = []

        for t in range(open_idx, n_months):
            q = {}
            for a in ASPECTS:
                drift[a] = 0.85 * drift[a] + rng.normal(0, 0.08)
                q[a] = base_q[a] + drift[a] + sum(_episode_effect(e, t) for e in eps if e["aspect"] == a)
            # footfall reacts to problems with a two-month lag
            lagged_bad = sum(min(0.0, _episode_effect(e, t - 2)) for e in eps)
            lam = lam0 * float(np.clip(1 + 0.12 * lagged_bad, 0.35, 1.4))
            month_start = months[t].to_timestamp()
            days_in_month = months[t].days_in_month

            for _ in range(int(rng.poisson(lam * 9))):  # check-ins (~9x reviews)
                ts = month_start + timedelta(days=int(rng.integers(0, days_in_month)), hours=int(rng.integers(10, 23)),
                                             minutes=int(rng.integers(0, 60)))
                checkin_dates.append(ts.strftime("%Y-%m-%d %H:%M:%S"))

            if not is_restaurant:
                n_month_reviews = int(rng.poisson(2))
            else:
                n_month_reviews = int(rng.poisson(lam))

            for _ in range(n_month_reviews):
                u = int(rng.integers(0, len(users)))
                mentioned, labels = [], {a: 0 for a in ASPECTS}
                for a in ASPECTS:
                    p = min(0.95, SALIENCE[a] * (1 + 1.3 * max(0.0, -q[a])))
                    if rng.random() < p:
                        mentioned.append(a)
                if not mentioned:
                    mentioned = ["food"]
                latent = 3.35 + leniency[u] + 0.2 * np.mean(list(q.values())) + rng.normal(0, 0.45)
                sentences = []
                for a in mentioned:
                    pos = rng.random() < _sigmoid(1.6 * q[a] + rng.normal(0, 0.6))
                    if pos:
                        kind, labels[a] = "pos", 1
                        latent += 0.42 * STAR_WEIGHT[a]
                    else:
                        labels[a] = 2
                        mild = q[a] > -0.7 and rng.random() < 0.75
                        kind = "mild" if mild else "neg"
                        latent -= (0.3 if mild else 1.15) * STAR_WEIGHT[a]
                    sentences.append(rng.choice(PHRASES[a][kind]))
                stars = int(np.clip(round(latent), 1, 5))
                rng.shuffle(sentences)
                parts = []
                if rng.random() < 0.6:
                    parts.append(rng.choice(OPENERS))
                parts.extend(sentences)
                if rng.random() < 0.3:
                    parts.insert(min(1, len(parts)), rng.choice(FILLERS))
                if rng.random() < 0.7:
                    parts.append(rng.choice(CLOSERS["high" if stars >= 4 else "mid" if stars == 3 else "low"]))
                d1, d2 = rng.choice(dishes, 2, replace=False)
                text = " ".join(parts).replace("{dish2}", d2).replace("{dish}", d1)
                text = text[0].upper() + text[1:]
                ts = month_start + timedelta(days=int(rng.integers(0, days_in_month)), hours=int(rng.integers(0, 24)),
                                             minutes=int(rng.integers(0, 60)), seconds=int(rng.integers(0, 60)))
                rid = _rand_id(rng)
                rec = {"review_id": rid, "user_id": users[u], "business_id": bid, "stars": float(stars),
                       "useful": int(rng.poisson(0.8)), "funny": int(rng.poisson(0.2)), "cool": int(rng.poisson(0.3)),
                       "text": text, "date": ts.strftime("%Y-%m-%d %H:%M:%S")}
                reviews.append(rec)
                if is_restaurant:
                    truth.append({"review_id": rid, **labels})
                star_sum += stars
                n_rev += 1
                # ~1% exact duplicates to exercise M1 de-duplication
                if rng.random() < 0.01:
                    reviews.append(dict(rec))
                # occasional tip
                if rng.random() < 0.08 and is_restaurant:
                    tips.append({"user_id": users[u], "business_id": bid, "text": rng.choice(PHRASES[mentioned[0]]["pos" if labels[mentioned[0]] == 1 else "mild"]).replace("{dish}", d1),
                                 "date": rec["date"], "compliment_count": 0})

        businesses.append({
            "business_id": bid, "name": name, "address": f"{int(rng.integers(10, 9999))} {rng.choice(['Main', 'Oak', 'Market', 'Broad', 'Elm', 'Pine'])} St",
            "city": city, "state": CITIES[city], "postal_code": f"{int(rng.integers(10000, 99999))}",
            "latitude": float(rng.uniform(25, 45)), "longitude": float(rng.uniform(-120, -75)),
            "stars": round(round(star_sum / max(n_rev, 1) * 2) / 2, 1), "review_count": n_rev,
            "is_open": 1, "attributes": {"RestaurantsTakeOut": "True"} if is_restaurant else None,
            "categories": categories, "hours": None,
        })
        if checkin_dates:
            checkins.append({"business_id": bid, "date": ", ".join(sorted(checkin_dates))})

    rng.shuffle(reviews)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "yelp_academic_dataset_business.json": businesses,
        "yelp_academic_dataset_review.json": reviews,
        "yelp_academic_dataset_checkin.json": checkins,
        "yelp_academic_dataset_tip.json": tips,
        "_ground_truth_aspects.jsonl": truth,
    }
    for fname, rows in files.items():
        with open(out_dir / fname, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, separators=(",", ":")) + "\n")
    with open(out_dir / "_ground_truth_episodes.json", "w", encoding="utf-8") as fh:
        json.dump(episodes_out, fh, indent=1)

    summary = {"businesses": len(businesses), "restaurants": n_restaurants, "reviews": len(reviews),
               "checkin_rows": len(checkins), "tips": len(tips), "episodes": len(episodes_out),
               "generated_at": datetime.now().isoformat(timespec="seconds")}
    print("[synthetic]", summary)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--restaurants", type=int, default=220)
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = ap.parse_args()
    generate(args.restaurants, args.seed)
