import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

# ========= LOAD ENV =========
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("No se encontró DATABASE_URL en el .env")

# ========= CONFIG =========
FILE_PATH = r"C:\Users\nicog\Documents\Proyectos Personales\FraudShield\credit_card_transactions.csv"
TABLE_NAME = "transactions"

VALID_COLUMNS = [
    "trans_num",
    "trans_date_trans_time",
    "unix_time",
    "cc_num",
    "merchant",
    "category",
    "amt",
    "first",
    "last",
    "gender",
    "job",
    "dob",
    "street",
    "city",
    "state",
    "zip",
    "lat",
    "longitude",
    "city_pop",
    "merch_lat",
    "merch_long",
    "merch_zipcode",
    "is_fraud"
]

# ========= LOAD =========
print("Leyendo CSV...")
df = pd.read_csv(FILE_PATH)

# ========= LIMPIEZA =========

# Renombrar columna conflictiva
df.rename(columns={"long": "longitude"}, inplace=True, errors="ignore")

# Filtrar columnas válidas
df = df[[col for col in VALID_COLUMNS if col in df.columns]]

# Validar columnas faltantes
missing = set(VALID_COLUMNS) - set(df.columns)
if missing:
    print(f"Columnas faltantes: {missing}")

# Eliminar duplicados
if "trans_num" in df.columns:
    before = len(df)
    df = df.drop_duplicates(subset=["trans_num"])
    print(f"Duplicados eliminados: {before - len(df)}")

# Convertir is_fraud de 0/1 → boolean
if "is_fraud" in df.columns:
    df["is_fraud"] = df["is_fraud"].astype(bool)
    
# Convertir fechas
print("Formateando fechas...")
df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"], errors="coerce")
df["dob"] = pd.to_datetime(df["dob"], errors="coerce")

# Manejo básico de nulos
df = df.fillna({
    "amt": 0,
    "city_pop": 0
})

# ========= CONEXIÓN =========
print("Conectando a PostgreSQL...")
engine = create_engine(DATABASE_URL)

# ========= INSERT =========
print("Insertando datos...")
df.to_sql(
    TABLE_NAME,
    engine,
    if_exists="append",
    index=False,
    chunksize=10000
)

print("✅ Carga completada.")