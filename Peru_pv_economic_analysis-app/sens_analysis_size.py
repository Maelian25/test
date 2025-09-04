import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === Load parameter data ===
df = pd.read_csv("parameters/parameters_mater_2025_new_prices.csv", sep=";", decimal=",")
import pandas as pd

import pandas as pd

data = [
    # Lima
    {"City": "Lima", "Technology": "PERC", "Power_kWp": 3, "Cost_USD_per_kWp": 1571.67},
    {"City": "Lima", "Technology": "PERC", "Power_kWp": 10, "Cost_USD_per_kWp": 1174.90},
    {"City": "Lima", "Technology": "PERC", "Power_kWp": 50, "Cost_USD_per_kWp": 923.94},
    {"City": "Lima", "Technology": "PERC", "Power_kWp": 100, "Cost_USD_per_kWp": 789.54},
    {"City": "Lima", "Technology": "PERC", "Power_kWp": 200, "Cost_USD_per_kWp": 698.01},
    {"City": "Lima", "Technology": "HIT",  "Power_kWp": 3, "Cost_USD_per_kWp": 1488.75},
    {"City": "Lima", "Technology": "HIT",  "Power_kWp": 10, "Cost_USD_per_kWp": 1086.28},
    {"City": "Lima", "Technology": "HIT",  "Power_kWp": 50, "Cost_USD_per_kWp": 832.02},
    {"City": "Lima", "Technology": "HIT",  "Power_kWp": 100, "Cost_USD_per_kWp": 697.55},
    {"City": "Lima", "Technology": "HIT",  "Power_kWp": 200, "Cost_USD_per_kWp": 606.02},

    # Chachapoyas
    {"City": "Chachapoyas", "Technology": "PERC", "Power_kWp": 3, "Cost_USD_per_kWp": 1837.00},
    {"City": "Chachapoyas", "Technology": "PERC", "Power_kWp": 10, "Cost_USD_per_kWp": 1254.90},
    {"City": "Chachapoyas", "Technology": "PERC", "Power_kWp": 50, "Cost_USD_per_kWp": 955.74},
    {"City": "Chachapoyas", "Technology": "PERC", "Power_kWp": 100, "Cost_USD_per_kWp": 813.41},
    {"City": "Chachapoyas", "Technology": "PERC", "Power_kWp": 200, "Cost_USD_per_kWp": 714.01},
    {"City": "Chachapoyas", "Technology": "HIT", "Power_kWp": 3, "Cost_USD_per_kWp": 1754.08},
    {"City": "Chachapoyas", "Technology": "HIT", "Power_kWp": 10, "Cost_USD_per_kWp": 1166.28},
    {"City": "Chachapoyas", "Technology": "HIT", "Power_kWp": 50, "Cost_USD_per_kWp": 863.82},
    {"City": "Chachapoyas", "Technology": "HIT", "Power_kWp": 100, "Cost_USD_per_kWp": 721.42},
    {"City": "Chachapoyas", "Technology": "HIT", "Power_kWp": 200, "Cost_USD_per_kWp": 622.02},

    # Tacna
    {"City": "Tacna", "Technology": "PERC", "Power_kWp": 3, "Cost_USD_per_kWp": 1819.33},
    {"City": "Tacna", "Technology": "PERC", "Power_kWp": 10, "Cost_USD_per_kWp": 1249.90},
    {"City": "Tacna", "Technology": "PERC", "Power_kWp": 50, "Cost_USD_per_kWp": 953.74},
    {"City": "Tacna", "Technology": "PERC", "Power_kWp": 100, "Cost_USD_per_kWp": 812.94},
    {"City": "Tacna", "Technology": "PERC", "Power_kWp": 200, "Cost_USD_per_kWp": 713.01},
    {"City": "Tacna", "Technology": "HIT", "Power_kWp": 3, "Cost_USD_per_kWp": 1736.42},
    {"City": "Tacna", "Technology": "HIT", "Power_kWp": 10, "Cost_USD_per_kWp": 1161.28},
    {"City": "Tacna", "Technology": "HIT", "Power_kWp": 50, "Cost_USD_per_kWp": 861.82},
    {"City": "Tacna", "Technology": "HIT", "Power_kWp": 100, "Cost_USD_per_kWp": 720.95},
    {"City": "Tacna", "Technology": "HIT", "Power_kWp": 200, "Cost_USD_per_kWp": 621.02},

    # Arequipa
    {"City": "Arequipa", "Technology": "PERC", "Power_kWp": 3, "Cost_USD_per_kWp": 1801.33},
    {"City": "Arequipa", "Technology": "PERC", "Power_kWp": 10, "Cost_USD_per_kWp": 1244.90},
    {"City": "Arequipa", "Technology": "PERC", "Power_kWp": 50, "Cost_USD_per_kWp": 950.74},
    {"City": "Arequipa", "Technology": "PERC", "Power_kWp": 100, "Cost_USD_per_kWp": 809.44},
    {"City": "Arequipa", "Technology": "PERC", "Power_kWp": 200, "Cost_USD_per_kWp": 712.01},
    {"City": "Arequipa", "Technology": "HIT", "Power_kWp": 3, "Cost_USD_per_kWp": 1718.42},
    {"City": "Arequipa", "Technology": "HIT", "Power_kWp": 10, "Cost_USD_per_kWp": 1156.28},
    {"City": "Arequipa", "Technology": "HIT", "Power_kWp": 50, "Cost_USD_per_kWp": 858.82},
    {"City": "Arequipa", "Technology": "HIT", "Power_kWp": 100, "Cost_USD_per_kWp": 717.45},
    {"City": "Arequipa", "Technology": "HIT", "Power_kWp": 200, "Cost_USD_per_kWp": 620.02},

    # Juliaca
    {"City": "Juliaca", "Technology": "PERC", "Power_kWp": 3, "Cost_USD_per_kWp": 1837.00},
    {"City": "Juliaca", "Technology": "PERC", "Power_kWp": 10, "Cost_USD_per_kWp": 1254.90},
    {"City": "Juliaca", "Technology": "PERC", "Power_kWp": 50, "Cost_USD_per_kWp": 955.74},
    {"City": "Juliaca", "Technology": "PERC", "Power_kWp": 100, "Cost_USD_per_kWp": 813.41},
    {"City": "Juliaca", "Technology": "PERC", "Power_kWp": 200, "Cost_USD_per_kWp": 714.01},
    {"City": "Juliaca", "Technology": "HIT", "Power_kWp": 3, "Cost_USD_per_kWp": 1754.08},
    {"City": "Juliaca", "Technology": "HIT", "Power_kWp": 10, "Cost_USD_per_kWp": 1166.28},
    {"City": "Juliaca", "Technology": "HIT", "Power_kWp": 50, "Cost_USD_per_kWp": 863.82},
    {"City": "Juliaca", "Technology": "HIT", "Power_kWp": 100, "Cost_USD_per_kWp": 721.42},
    {"City": "Juliaca", "Technology": "HIT", "Power_kWp": 200, "Cost_USD_per_kWp": 622.02},
]

df_costs = pd.DataFrame(data)

def get_Cu(city, tech, P):
    filtered = df_costs[
        (df_costs["City"].str.lower() == city.lower()) &
        (df_costs["Technology"].str.lower() == tech.lower())
    ]
    # Chercher la plus petite puissance >= P
    applicable = filtered[filtered["Power_kWp"] >= P]
    if not applicable.empty:
        return applicable.sort_values("Power_kWp").iloc[0]["Cost_USD_per_kWp"]
    else:
        # Si P dépasse toutes les catégories, on prend la plus grande dispo
        return filtered.sort_values("Power_kWp").iloc[-1]["Cost_USD_per_kWp"]


# === Define parameters ===
P_values = np.linspace(2, 200, num=100)  # 100 values from 2 to 200 kW
target_cities = ["lima", "arequipa", "tacna", "chachapoyas", "juliaca"]
technologies = ["hit", "perc"]

# === Storage for results ===
sensitivity_data = []

# === Loop through columns like 'lima_hit', 'tacna_perc' etc. ===
for col in df.columns:
    if "_" not in col:
        continue

    city, tech = col.lower().split("_")

    if city not in target_cities or tech not in technologies:
        continue

    try:
        values = dict(zip(df["parameter"], df[col]))

        # Fixed values per city/technology
        Hopt = float(values["Hopt"])
        PR = float(values["PR"]) / 100
        rd = float(values["rd"]) / 100
        N = int(values["N"])
        Cu = float(values["Cu"])
        COM = float(values["COM"]) / 100
        d = float(values["d"]) / 100
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
        

    except Exception as e:
        print(f"⚠️ Error parsing column '{col}': {e}")
        continue


# === WACC Calculation ===
    if Nis == 0:
        subsidy_part = 0
    else:
        subsidy_part = Xis * (1 / Nis)
    if Nl == 0:
        loan_part = 0
    else:
        loan_part = Xl * (il / (1 - (1 + il) ** -Nl))
    equity_part = Xec * dec
    WACC = loan_part + equity_part + subsidy_part
    d = WACC *100 # Set discount rate to WACC for LCOE calculation


    for P in P_values:
        Cu = get_Cu(city, tech, P)
        pvi = Cu * P
        pvom = COM * pvi

        # --- Economic formulas ---
        def EPV(Hopt, P, PR): return P * Hopt * PR

        def EPV_discounted(Hopt, P, PR, N, rd, d):
            factor = (1 - rd) / (1 + d)
            return EPV(Hopt, P, PR) * sum(factor ** i for i in range(1, N + 1))

        def PWCO(pvi, pvom, N, T, Xd, Nd, Xl, Xec, Xis, il, dec, Nis, Nl, source):
            q = 1 / (1 + d)
            Kp = (1 + rom) / (1 + d)
            OPEX_term = (
                pvom * (1 - T) * Kp * (1 - Kp ** N) / (1 - Kp)
                if source == "talavera"
                else pvom * Kp * (1 - Kp ** N) / (1 - Kp)
            )
            DEP = 0 if Nd == 0 else pvi * Xd / Nd
            PWDEP = DEP * q * (1 - q ** Nd) / (1 - q)
            PVl = Xl * pvi
            PVec = Xec * pvi
            PVis = Xis * pvi
            loan_term = (
                (PVl * il * (1 - T)) / (1 - (1 + il * (1 - T)) ** (-Nl)) * q * (1 - q ** Nl) / (1 - q)
                if Xl > 0 else 0
            )
            equity_term = (
                dec * PVec * q * (1 - q ** N) / (1 - q) if Xec > 0 else 0
            )
            dividend_term = PVec * (q ** N) if Xec > 0 else 0
            subsidy_term = (
                (PVis / Nis) * T * q * (1 - q ** Nis) / (1 - q)
                if Xis > 0 and Nis > 0 else 0
            )
            pvinv = loan_term + equity_term + dividend_term + subsidy_term
            return pvinv + OPEX_term - PWDEP * T

        def LCOE(pvi, Hopt, P, PR, rd, d, N, rom, pvom, T,
                 Xd, Nd, Xl, Xec, Xis, il, dec, Nis, Nl, source):
            energy = EPV_discounted(Hopt, P, PR, N, rd, d)
            cost = PWCO(pvi, pvom, N, T, Xd, Nd, Xl, Xec, Xis,
                        il, dec, Nis, Nl, source)
            return cost / energy

        lcoe = LCOE(pvi, Hopt, P, PR, rd, d, N, rom, pvom, T,
                    Xd, Nd, Xl, Xec, Xis, il, dec, Nis, Nl, source="talavera")

        sensitivity_data.append({
            "City": city.title(),
            "Technology": tech.upper(),
            "P": P,
            "LCOE": lcoe
        })

# === Convert to DataFrame ===
df_sens = pd.DataFrame(sensitivity_data)
print(df_sens.head())

# === Plotting ===
for tech in ["HIT", "PERC"]:
    plt.figure(figsize=(12, 6))
    subset = df_sens[df_sens["Technology"] == tech]
    for city in subset["City"].unique():
        city_data = subset[subset["City"] == city]
        plt.plot(city_data["P"], city_data["LCOE"], label=city)

    plt.title(f"LCOE Sensitivity to Peak Power (P) — Technology: {tech}")
    plt.xlabel("Peak Power (kW)")
    plt.ylabel("LCOE (USD/kWh)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(title="City")
    plt.tight_layout()
    plt.show()
