"""
generate_json_samples.py
-------------------------
Produces many varied JSON value-sets for a document type, matching the
exact field schema your YAML configs expect. No Faker dependency —
stdlib random/datetime only, plus curated Cameroonian name/place lists
(generic Western-name generators don't reflect local naming patterns:
compound surnames, hyphenated given names, etc.)

Usage:
    python generate_json_samples.py --doc-type nic_v1 --count 500 --out-dir data/json/nic_v1
"""

import os
import json
import random
import argparse
from datetime import date, timedelta

# --- Curated reference lists -------------------------------------------

SURNAMES = [
    "FONING", "TCHUENTE", "NGUEMA", "ATANGANA", "MBIDA", "EWANE",
    "TAKOUGANG", "KAMGA", "FOTSO", "NDONGO", "ESSOMBA", "MBARGA",
    "ONANA", "TCHIO", "MEKONGO", "AWONO", "NKOUM", "TABI", "NJOYA",
    "MOUNCHILI", "NCHOUNKEU", "TALLA", "WANDJI", "DJOUMESSI",
    "KENFACK", "SIMO", "LACKMATA", "MASSEHE", "FOLEFACK",
    "NGUEFACK", "TAGNE", "DONGMO", "KEMAJOU", "NANFACK",
    "TCHOUAMENI", "FEUDJIO", "MEKONTSO", "TCHATCHOUA", "YOUMBI",
    "NOUBISSIE", "TCHINDA", "KAMDEM", "NANA", "TEDONGMO", "MAGNE",
    "ABEGA", "ELOUNDOU", "OWONA", "ZE", "ENGO", "MVOGO", "BIYIDI",
    "EDZOA", "MINKOO", "OYONO", "MEDZO", "NDOUMBE", "NGO BASSONG",
    "HAMAN", "MBOG", "NYOBE", "EYENGA", "LIKIBI", "MBOUOMBOUO",
    "NCHARE", "TANYI", "AWA", "NKWENTI", "NDIFOR", "FRU", "ACHU",
    "NDANGOH", "MBAH", "NCHE", "ASHU", "TANKOU", "BEBOH", "EMINANG",
    "OUMAROU", "ABOUBAKAR", "ISSA", "HAMIDOU", "BELLO", "MOHAMADOU",
    "YOUSSOUFA", "DAIROU", "GARBA", "NGONO", "BIYA", "ETOUNDI",
]

GIVEN_NAMES_MALE = [
    "LUC-XAVIER", "JEAN", "PAUL", "ERIC", "PATRICK", "FRANCK", "YVES",
    "BLAISE", "ARMAND", "RAOUL", "STEVE", "WILLY", "HERVE", "CYRILLE",
    "DESIRE", "ALAIN", "BERTRAND", "CLAUDE", "SERGE", "MARTIN",
    "EMMANUEL", "CHRISTIAN", "LIONEL", "ARSENE", "RODRIGUE", "PACOME",
    "ROMUALD", "THIERRY", "GAETAN", "FABRICE", "LANDRY", "JORES",
    "JORDAN", "MERLIN", "GUY-ROLAND", "DONALD", "KEVIN", "BRICE",
    "OLIVIER", "PASCAL", "VALENTIN", "GILBERT", "ROGER", "SAMUEL",
    "DANIEL", "MARCEL", "JOSEPH", "PIERRE", "ANDRE", "HONORE",
    "PROSPER", "FIDELE", "CALIXTE", "LAZARE", "EMILE", "SYLVAIN",
    "NARCISSE", "GERVAIS", "LEOPOLD", "MAXIME",
]

GIVEN_NAMES_FEMALE = [
    "MARIE-CLAIRE", "SANDRINE", "CHRISTELLE", "ESTELLE", "PATRICIA",
    "FLORENCE", "AUDREY", "NATHALIE", "GISELE", "CARINE", "LAURE",
    "SOLANGE", "VALERIE", "BRIGITTE", "PAULINE", "AMINA", "FATIMATOU",
    "DELPHINE", "ODILE", "JEANNINE-CAINE", "CHANTAL", "YOLANDE",
    "PRISCILLE", "LEONIE", "HORTENSE", "CLARISSE", "VIVIANE",
    "ARMELLE", "BEATRICE", "JULIENNE", "MONIQUE", "THERESE", "CECILE",
    "ROSINE", "HUGUETTE", "ANASTASIE", "CONSTANCE", "GRACE",
    "MERVEILLE", "DIVINE", "PRECIEUSE", "CHRISTIANE", "MIREILLE",
    "ROSELINE", "EDWIGE", "GENEVIEVE", "ANNE-MARIE", "MARGUERITE",
    "ELISABETH", "FRANCOISE", "SUZANNE", "MADELEINE", "HABIBA",
    "RAMATOU", "AISSATOU", "NOELLA", "PERPETUE", "JUSTINE",
    "VIRGINIE", "ODETTE",
]

PLACES = [
    "YAOUNDE", "DOUALA", "LIMBE", "BAFOUSSAM", "GAROUA", "BAMENDA",
    "KRIBI", "BUEA", "NGAOUNDERE", "EBOLOWA", "EDEA", "MAROUA",
    "BERTOUA", "KUMBA", "DSCHANG", "FOUMBAN", "MBALMAYO",
    "SANGMELIMA", "TIKO", "MUTENGENE", "NKONGSAMBA", "BAFANG",
    "BANGANGTE", "MBOUDA", "MOKOLO", "KOUSSERI", "YAGOUA", "GUIDER",
    "MEIGANGA", "BATOURI", "ABONG-MBANG", "NANGA-EBOKO", "OBALA",
    "MFOU", "AKONOLINGA", "WUM", "KUMBO", "FUNDONG", "MAMFE",
    "NKAMBE", "NDOP", "AMBAM", "KYE-OSSI", "LOLODORF", "ESEKA",
    "BAFIA", "NTUI", "YOKADOUMA", "DJOUM", "TIBATI", "BANYO",
    "TIGNERE", "POLI", "MORA", "KAELE", "FOUMBOT", "BANDJOUN",
    "BAHAM", "DIMAKO", "LOMIE", "SOA", "MELONG", "MANJO", "PENJA",
    "LOUM",
]

OCCUPATIONS = [
    "ETUDIANT", "ETUDIANTE", "ENSEIGNANT", "ENSEIGNANTE", "COMMERCANT",
    "COMMERCANTE", "INFIRMIER", "INFIRMIERE", "INGENIEUR",
    "FONCTIONNAIRE", "CHAUFFEUR", "AGRICULTEUR", "MENUISIER",
    "COUTURIERE", "MEDECIN", "AVOCAT", "COMPTABLE", "ELECTRICIEN",
    "MECANICIEN", "SANS EMPLOI", "JOURNALISTE", "ARCHITECTE",
    "PHARMACIEN", "PHARMACIENNE", "NOTAIRE", "HUISSIER", "MAGISTRAT",
    "POLICIER", "GENDARME", "MILITAIRE", "DOUANIER", "BANQUIER",
    "SECRETAIRE", "INFORMATICIEN", "TECHNICIEN", "PLOMBIER", "MACON",
    "SOUDEUR", "CUISINIER", "RESTAURATEUR", "COIFFEUR", "COIFFEUSE",
    "ESTHETICIENNE", "BOUCHER", "BOULANGER", "PECHEUR", "ELEVEUR",
    "TRANSPORTEUR", "LOGISTICIEN", "CONSULTANT", "ENTREPRENEUR",
    "GERANT", "DIRECTEUR", "VENDEUR", "VENDEUSE", "TAILLEUR",
    "PASTEUR", "RETRAITE", "MENAGERE", "ARTISAN",
]


# --- Field generators -----------------------------------------------------

def _fmt(d: date) -> str:
    return d.strftime("%d.%m.%Y")


def _random_date(start: date, end: date) -> date:
    span = (end - start).days
    return start + timedelta(days=random.randint(0, max(span, 0)))


def gen_full_name():
    """Returns (surname, given_names, sex)."""
    sex = random.choice(["M", "F"])
    surname = " ".join(random.sample(SURNAMES, k=random.choice([1, 2])))
    pool = GIVEN_NAMES_MALE if sex == "M" else GIVEN_NAMES_FEMALE
    given = random.choice(pool)
    return surname, given, sex


def gen_dob():
    today = date.today()
    # adult card holder, 18-70 years old
    start = date(today.year - 70, 1, 1)
    end = date(today.year - 18, 12, 31)
    return _fmt(_random_date(start, end))


def gen_issue_expiry(validity_years=10):
    today = date.today()
    start = date(today.year - 9, 1, 1)  # issued sometime in the last ~9 years
    issue = _random_date(start, today)
    expiry = date(issue.year + validity_years, issue.month, issue.day)
    return _fmt(issue), _fmt(expiry)


def gen_parent_name():
    surname = random.choice(SURNAMES)
    given = random.choice(GIVEN_NAMES_MALE + GIVEN_NAMES_FEMALE)
    return f"{surname} {given}"


def gen_height(sex):
    lo, hi = (1.60, 1.90) if sex == "M" else (1.50, 1.75)
    return f"{round(random.uniform(lo, hi), 2):.2f} m"


def gen_nic_number():
    letters = "".join(random.choices("ABCDEFGHJKLMNPQRSTUVWXYZ", k=2))
    digits = "".join(random.choices("0123456789", k=8))
    return f"{letters}{digits}"


# --- Document-type builders -------------------------------------------

def build_nic_v1():
    surname, given, sex = gen_full_name()
    issue_date, expiry_date = gen_issue_expiry(validity_years=10)
    return {
        "surname": surname,
        "given_names": given,
        "date_of_birth": gen_dob(),
        "sex": sex,
        "expiry_date": expiry_date,
        "father_name": gen_parent_name(),
        "mother_name": gen_parent_name(),
        "place_of_birth": random.choice(PLACES),
        "occupation": random.choice(OCCUPATIONS),
        "issue_date": issue_date,
        "height": gen_height(sex),
        "nic_number": gen_nic_number(),
    }


DOC_BUILDERS = {
    "nic_v1": build_nic_v1,
    # add "nic_v2", "passport", "driving_license" here later —
    # same helper functions above can be reused for shared fields
}


def main():
    parser = argparse.ArgumentParser(description="Generate varied document JSON samples")
    parser.add_argument("--doc-type", required=True, choices=list(DOC_BUILDERS.keys()))
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    builder = DOC_BUILDERS[args.doc_type]

    for i in range(args.count):
        sample = builder()
        path = os.path.join(args.out_dir, f"sample_{i:04d}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(sample, f, ensure_ascii=False, indent=2)

    print(f"Wrote {args.count} samples to {args.out_dir}")


if __name__ == "__main__":
    main()