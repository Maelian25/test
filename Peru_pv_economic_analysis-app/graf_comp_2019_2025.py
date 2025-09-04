import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# === Fonctions économiques (modèle TALAVERA) ===
def EPV(Hopt, P, PR): return P * Hopt * PR
def PVI(P, Cu): return Cu * P
def PVOM(PVI, COM): return COM * PVI
def EPVs(epv, SCI): return epv * SCI
def EPVg(epv, SCI): return epv * (1 - SCI)
def EPV_discounted(Hopt, P, PR, N, rd, d): return EPV(Hopt, P, PR) * sum(((1 - rd)/(1 + d))**i for i in range(1, N + 1))

def PWCO(pvi, pvom, N, T, Xd, Nd, Xl, Xec, Xis, il, dec, Nis, Nl, d, rom):
    q = 1 / (1 + d)
    Kp = (1 + rom) / (1 + d)
    PWPVOM = pvom * (1 - T) * Kp * (1 - Kp**N) / (1 - Kp)
    DEP = 0 if Nd == 0 else pvi * Xd / Nd
    PWDEP = DEP * q * (1 - q**Nd) / (1 - q)
    return pvi + PWPVOM - PWDEP * T

def LCOE(pvi, Hopt, P, PR, rd, d, N, rom, pvom, T, Xd, Nd, Xl, Xec, Xis, il, dec, Nis, Nl):
    epv_discounted = EPV_discounted(Hopt, P, PR, N, rd, d)
    pwco = PWCO(pvi, pvom, N, T, Xd, Nd, Xl, Xec, Xis, il, dec, Nis, Nl, d, rom)
    return pwco / epv_discounted

def PWCI(ps, pg, N, T, epvs, epvg, rd, d, rps, rpg):
    Ks = (1 + rps) * (1 - rd) / (1 + d)
    Kg = (1 + rpg) * (1 - rd) / (1 + d)
    return (ps * epvs * (1 - T) * Ks * (1 - Ks**N) / (1 - Ks)) + (pg * epvg * (1 - T) * Kg * (1 - Kg**N) / (1 - Kg))

def NPV_calc(pvi, pvom, pg, ps, rpg, rps, rd, d, N, T, Nd, Xd, epvs, epvg, Xl, Xec, Xis, il, dec, Nis, Nl, rom):
    pwci = PWCI(ps, pg, N, T, epvs, epvg, rd, d, rps, rpg)
    pwco = PWCO(pvi, pvom, N, T, Xd, Nd, Xl, Xec, Xis, il, dec, Nis, Nl, d, rom)
    return pwci - pwco

# === Lecture des fichiers ===
files = {
    "2019": "parameters/parameters_mater_2019_updated.csv",
    "2025": "parameters/parameters_mater_2025.csv"
}

results = []

for year, filepath in files.items():
    df = pd.read_csv(filepath, sep=";", decimal=",")
    scenarios = [col for col in df.columns if "_" in col]

    for col in scenarios:
        try:
            city, scenario = col.split("_")
            label = f"{city.title()}_{scenario.upper()}"
            values = dict(zip(df["parameter"], df[col]))

            # Extraction des paramètres
            Hopt = float(values["Hopt"])
            P = float(values["P"])
            PR = float(values["PR"]) / 100
            rd = float(values["rd"]) / 100
            El = float(values["El"])
            N = int(values["N"])
            SCI = float(values["SCI"]) / 100
            Cu = float(values["Cu"])
            COM = float(values["COM"]) / 100
            d = float(values["d"]) / 100
            pg = float(values["pg"])
            ps = float(values["ps"])
            rpg = float(values["rpg"]) / 100
            rps = float(values["rps"]) / 100
            rom = float(values["rom"]) / 100
            T = float(values["T"]) / 100
            Nd = int(values["Nd"])
            Xd = float(values["Xd"]) / 100
            Xl = float(values["Xl"]) / 100
            Xec = float(values["Xec"]) / 100
            Xis = float(values["Xis"]) / 100
            il = float(values["il"]) / 100
            Nis = int(values["Nis"])
            Nl = int(values["Nl"])
            dec = float(values["dec"]) / 100

            # Calculs
            pvi = PVI(P, Cu)
            pvom = PVOM(pvi, COM)
            epv = EPV(Hopt, P, PR)
            epvs = EPVs(epv, SCI)
            epvg = EPVg(epv, SCI)

            lcoe = LCOE(pvi, Hopt, P, PR, rd, d, N, rom, pvom, T, Xd, Nd,
                        Xl, Xec, Xis, il, dec, Nis, Nl)
            npv_val = NPV_calc(pvi, pvom, pg, ps, rpg, rps, rd, d, N, T, Nd, Xd,
                               epvs, epvg, Xl, Xec, Xis, il, dec, Nis, Nl, rom)

            results.append({
                "Year": int(year),
                "City": city.title(),
                "Scenario": scenario.upper(),
                "Label": label,
                "LCOE (USD/kWh)": round(lcoe, 4),
                "NPV (USD)": round(npv_val, 2),
                "ps (USD/kWh)": round(ps, 4)
            })

        except Exception as e:
            print(f"Erreur dans {col} ({year}) : {e}")

# === Export Excel ===
df_final = pd.DataFrame(results)
df_final.to_excel("LCOE_NPV_results.xlsx", index=False)

# === Graphiques par scénario (comparaison 2019–2025) ===
df_pivot = df_final.pivot_table(index="Label", columns="Year", values=["LCOE (USD/kWh)", "ps (USD/kWh)"])
os.makedirs("scenario_charts", exist_ok=True)

for label in df_pivot.index:
    try:
        years = [2019, 2025]
        lcoe_vals = df_pivot.loc[label, ("LCOE (USD/kWh)", 2019)], df_pivot.loc[label, ("LCOE (USD/kWh)", 2025)]
        ps_vals = df_pivot.loc[label, ("ps (USD/kWh)", 2019)], df_pivot.loc[label, ("ps (USD/kWh)", 2025)]

        plt.figure(figsize=(6, 4))
        plt.plot(years, lcoe_vals, label="LCOE", marker='o', color="green")
        plt.plot(years, ps_vals, label="Tarif électricité (ps)", marker='o', color="orange")
        plt.title(f"{label} — LCOE vs ps (2019–2025)")
        plt.xlabel("Année")
        plt.ylabel("USD/kWh")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"scenario_charts/{label}_comparaison.png")
        plt.close()
    except:
        print(f"❌ Erreur graphique pour {label}")

print("✅ Résultats enregistrés dans 'LCOE_NPV_results.xlsx'")
print("📊 Graphiques enregistrés dans le dossier 'scenario_charts'")
