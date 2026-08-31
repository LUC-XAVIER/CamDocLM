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

NEIGHBORHOODS = [
    "BASTOS", "MELEN", "EMOMBO", "NLONGKAK", "MVOG-ADA", "ETOUDI",
    "NKOLBISSON", "BIYEM-ASSI", "MENDONG", "OLEZOA", "NKOMO",
    "NKAMOENG", "MVAN", "NSAM", "TSINGA", "MESSA", "NKOLNDONGO",
    "ESSOS", "MVOG-MBI", "NGOUSSO", "AWAE", "ETOA-MEKI", "EKOUNOU",
    "EFOULAN", "SIMBOCK", "ODZA", "NKOLMESSENG", "MIMBOMAN", "EMANA",
    "NKOABANG", "OBILI", "NGOA-EKELLE", "CITE VERTE", "WARDA",
    "NKOLBIKOK", "MBALLA", "AHALA", "ELIG-ESSONO", "ELIG-EDZOA",
    "BRIQUETERIE", "MADAGASCAR", "AKWA", "BONANJO", "BONAPRISO",
    "DEIDO", "NEW-BELL", "BALI", "BONABERI", "NDOGBONG", "MAKEPE",
    "LOGBABA", "BEPANDA", "KOTTO", "NDOKOTI", "BASSA", "VILLAGE",
    "PK8", "PK10", "PK14", "CITE SIC", "YASSA", "KAMKOP", "TAMDJA",
    "BANENGO", "DJELENG", "TOUGANG",
]
 
CITY_ABBREV = {
    "YAOUNDE": "YDE", "DOUALA": "DLA", "BAFOUSSAM": "BAF",
    "GAROUA": "GRA", "BAMENDA": "BDA", "BUEA": "BUE", "MAROUA": "MRA",
    "BERTOUA": "BTA", "KUMBA": "KMB", "LIMBE": "LBE",
    "NGAOUNDERE": "NGD", "EBOLOWA": "EBW", "EDEA": "EDA",
    "DSCHANG": "DSC", "FOUMBAN": "FMB", "KRIBI": "KRB",
    "MBALMAYO": "MBM", "SANGMELIMA": "SGM", "TIKO": "TKO",
    "MUTENGENE": "MTG",
}


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
    try:
        expiry = date(issue.year + validity_years, issue.month, issue.day)
    except ValueError:
        expiry = date(issue.year + validity_years, 2, 28)
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

def gen_nic_number_numeric(length=9):
    return "".join(random.choices("0123456789", k=length))

def _fmt_slash(d: date) -> str:
    return d.strftime("%d/%m/%Y")
 
 
def gen_dob_slash():
    today = date.today()
    start = date(today.year - 70, 1, 1)
    end = date(today.year - 18, 12, 31)
    return _fmt_slash(_random_date(start, end))
 
 
def gen_issue_expiry_slash(validity_years=10):
    today = date.today()
    start = date(today.year - 9, 1, 1)
    issue = _random_date(start, today)
    try:
        expiry = date(issue.year + validity_years, issue.month, issue.day)
    except ValueError:
        expiry = date(issue.year + validity_years, 2, 28)
    return _fmt_slash(issue), _fmt_slash(expiry)

def gen_height_comma(sex, suffix=""):
    lo, hi = (1.60, 1.90) if sex == "M" else (1.50, 1.75)
    val = f"{round(random.uniform(lo, hi), 2):.2f}".replace(".", ",")
    return f"{val}{suffix}"
 
 
def gen_address():
    place = random.choice(PLACES)
    abbrev = CITY_ABBREV.get(place)
    if abbrev and random.random() < 0.5:
        return f"{abbrev} - {random.choice(NEIGHBORHOODS)}"
    return random.choice(NEIGHBORHOODS)
 
 
def gen_post_id():
    letters = "".join(random.choices("ABCDEFGHJKLMNPQRSTUVWXYZ", k=2))
    digits = "".join(random.choices("0123456789", k=2))
    return f"{letters}{digits}"
 
 
def gen_unique_id(length=17):
    # format/length-matched placeholder — real meaning of this field unknown
    return "".join(random.choices("0123456789", k=length))
 
 
def gen_matrimonial_code():
    # format-matched placeholder — real meaning of this code unknown
    return "".join(random.choices("0123456789", k=random.choice([4, 6])))
 
def _fmt_dash(d: date) -> str:
    return d.strftime("%d-%m-%Y")
 
 
def gen_dob_dash():
    today = date.today()
    start = date(today.year - 70, 1, 1)
    end = date(today.year - 18, 12, 31)
    return _fmt_dash(_random_date(start, end))
 
 
def gen_issue_expiry_dash(validity_years=10):
    today = date.today()
    start = date(today.year - 9, 1, 1)
    issue = _random_date(start, today)
    try:
        expiry = date(issue.year + validity_years, issue.month, issue.day)
    except ValueError:
        expiry = date(issue.year + validity_years, 2, 28)  
    return _fmt_dash(issue), _fmt_dash(expiry)
 
 
def gen_dob_range(min_age, max_age):
    """Dot-format DOB over a wider age range than the 18-70 default —
    used for passports, which are issued to minors too."""
    today = date.today()
    start = date(today.year - max_age, 1, 1)
    end = date(today.year - min_age, 12, 31)
    return _fmt(_random_date(start, end))
 
 
def gen_issued_by():
    initials = "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=1))
    initials2 = "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=1))
    return f"{random.choice(SURNAMES)} {initials}. {initials2}."
 
 
def gen_document_discriminator():
    return f"DTR-{random.randint(0, 999):03d}-{random.randint(0, 9999):04d}-{random.randint(0, 99):02d}"
 
 
def gen_licence_number():
    return f"DTR-{random.randint(0, 999999):06d}-{random.randint(0, 99):02d}"
 
 
def gen_vehicle_category():
    categories = ["A1", "A", "B", "BE", "C", "CE", "D", "DE", "FA1", "FA", "FB", "G"]
    weights = [5, 10, 50, 10, 8, 5, 4, 3, 1, 1, 1, 2]  # B is by far the most common
    return random.choices(categories, weights=weights)[0]
 
 
def gen_national_reg_number():
    return f"NEW: CE-{random.randint(0, 99999999):08d}"
 
 
def gen_passport_number():
    letters = "".join(random.choices("ABCDEFGHJKLMNPQRSTUVWXYZ", k=2))
    digits = "".join(random.choices("0123456789", k=6))
    return f"{letters}{digits}"
 
 
def gen_can():
    return "".join(random.choices("0123456789", k=6))
 
 
def gen_nationality(sex):
    return "CAMEROUNAISE/ CAMEROONIAN"
 
 
REGIONAL_CAPITALS = [
    "YAOUNDE", "DOUALA", "BAMENDA", "BUEA", "BAFOUSSAM", "GAROUA",
    "MAROUA", "NGAOUNDERE", "BERTOUA", "EBOLOWA",
]


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


def build_nic_v2():
    """Specimen-style layout: post_id/unique_id fields, dot dates,
    comma-decimal height with no unit suffix."""
    surname, given, sex = gen_full_name()
    issue_date, expiry_date = gen_issue_expiry(validity_years=10)
    return {
        "surname": surname,
        "given_names": given,
        "date_of_birth": gen_dob(),
        "place_of_birth": random.choice(PLACES),
        "sex": sex,
        "height": gen_height_comma(sex),
        "occupation": random.choice(OCCUPATIONS),
        "father_name": gen_parent_name(),
        "mother_name": gen_parent_name(),
        "proffessional_matrimonial_situatn": gen_matrimonial_code(),
        "address": gen_address(),
        "issue_date": issue_date,
        "expiry_date": expiry_date,
        "post_id": gen_post_id(),
        "unique_id": gen_unique_id(),
    }
 
 
def build_nic_v3():
    """Slash-date layout with spouse field, comma-decimal height with
    trailing 'M', numeric-only nic_number."""
    surname, given, sex = gen_full_name()
    issue_date, expiry_date = gen_issue_expiry_slash(validity_years=10)
    has_spouse = random.random() < 0.6
    return {
        "surname": surname,
        "spouse": random.choice(SURNAMES) if has_spouse else "",
        "given_names": given,
        "date_of_birth": gen_dob_slash(),
        "place_of_birth": random.choice(PLACES),
        "father_name": gen_parent_name(),
        "mother_name": gen_parent_name(),
        "occupation": random.choice(OCCUPATIONS),
        "address": gen_address(),
        "height": gen_height_comma(sex, suffix="M"),
        "sex": sex,
        "proffessional_matrimonial_situatn": gen_matrimonial_code(),
        "issue_date": issue_date,
        "expiry_date": expiry_date,
        "nic_number": gen_nic_number_numeric(),
    }
 
def build_driving_license():
    surname, given, sex = gen_full_name()
    issue_date, expiry_date = gen_issue_expiry_dash(validity_years=10)
    return {
        "surname": surname,
        "given_names": given,
        "date_of_birth": gen_dob_dash(),
        "place_of_birth": random.choice(PLACES),
        "issue_date": issue_date,
        "expiry_date": expiry_date,
        "issued_by": gen_issued_by(),
        "document_discriminator": gen_document_discriminator(),
        "licence_number": gen_licence_number(),
        "vehicle_category": gen_vehicle_category(),
        "national_reg_number": gen_national_reg_number(),
    }
 
 
def build_passport():
    surname, given, sex = gen_full_name()
    issue_date, expiry_date = gen_issue_expiry(validity_years=5)
    return {
        "type": "PP",
        "country_code": "CMR",
        "passport_number": gen_passport_number(),
        "surname": surname,
        "given_names": given,
        "nationality": gen_nationality(sex),
        "occupation": random.choice(OCCUPATIONS + ["ELEVE"]),
        "date_of_birth": gen_dob_range(1, 85),
        "height": gen_height(sex),
        "can": gen_can(),
        "sex": sex,
        "place_of_birth": random.choice(PLACES),
        "issue_date": issue_date,
        "place_of_issue": random.choice(REGIONAL_CAPITALS),
        "expiry_date": expiry_date,
    }
 
 
DOC_BUILDERS = {
    "nic_v1": build_nic_v1,
    "nic_v2": build_nic_v2,
    "nic_v3": build_nic_v3,
    "driving_license": build_driving_license,
    "passport": build_passport,
}
 
 
def main():
    parser = argparse.ArgumentParser(description="Generate varied document JSON samples")
    parser.add_argument("--doc-type", required=True, choices=list(DOC_BUILDERS.keys()))
    parser.add_argument("--count", type=int, default=500)
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