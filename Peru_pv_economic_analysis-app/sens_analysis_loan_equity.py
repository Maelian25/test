import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === Load parameter data ===
df = pd.read_csv("parameters/parameters_mater_2025_new_prices.csv", sep=";", decimal=",")

# === Define parameters ===
Xl_values = np.linspace(0, 1, 100)
target_cities = ["lima", "arequipa", "tacna", "chachapoyas", "juliaca"]
technologies = ["hit", "perc"]

sensitivity_data = []

for col in df.columns:
    if "_" not in col:
        continue

    city, tech = col.lower().split("_")

    if city not in target_cities or tech not in technologies:
        continue

    try:
        values = dict(zip(df["parameter"], df[col]))

        # 🔁 Extraire tous les paramètres, même inutilisés
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
        Xl_default = float(values["Xl"]) / 100
        Xec_default = float(values["Xec"]) / 100
        Xis = float(values["Xis"]) / 100
        il = float(values["il"]) / 100
        Nis = int(values["Nis"])
        Nl = int(values["Nl"])
        dec = float(values["dec"]) / 100

        pvi = Cu * P
        pvom = COM * pvi

    except Exception as e:
        print(f"⚠️ Error in column {col}: {e}")
        continue

    for Xl in Xl_values:
        Xec = 1 - Xl  # Equity share

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

        # === Energy Calculations ===
        def EPV(Hopt, P, PR):
            return P * Hopt * PR

        def EPV_discounted(Hopt, P, PR, N, rd, d):
            factor = (1 - rd) / (1 + d)
            return EPV(Hopt, P, PR) * sum(factor ** i for i in range(1, N + 1))

        def PWCO(pvi, pvom, N, T, Xd, Nd, Xl, Xec, Xis, il, dec, Nis, Nl, source):
            q = 1 / (1 + d)
            Kp = (1 + rom) / (1 + d)
            PWPVOM = (
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
            return pvinv + PWPVOM - PWDEP * T

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
            "Loan_Share": Xl,
            "Equity_Share": Xec,
            "LCOE": lcoe,
            "ps": ps  # Pour pouvoir tracer plus tard
        })

# === Convert to DataFrame ===
df_sens_Xl = pd.DataFrame(sensitivity_data)

# === Plot results for each technology ===
for tech in ["HIT", "PERC"]:
    plt.figure(figsize=(12, 6))
    subset = df_sens_Xl[df_sens_Xl["Technology"] == tech]
    cities = subset["City"].unique()

    # Générer une palette de couleurs pour chaque ville
    color_map = plt.cm.get_cmap("tab10", len(cities))
    color_dict = {city: color_map(i) for i, city in enumerate(cities)}

    for city in cities:
        city_data = subset[subset["City"] == city]
        color = color_dict[city]

        # Tracer LCOE
        plt.plot(city_data["Loan_Share"], city_data["LCOE"], label=f"{city} LCOE", color=color)

        # Tracer le tarif d’électricité de la même couleur, ligne horizontale jusqu’à Loan_Share = 1
        ps_value = city_data["ps"].iloc[0]
        plt.hlines(y=ps_value, xmin=0, xmax=1, colors=color, linestyles="--", linewidth=1.2,
                   label=f"{city} Electricity Tariff")

    plt.title(f"LCOE Sensitivity to Financing Structure — Technology: {tech}")
    plt.xlabel("Loan Share (Xl)")
    plt.ylabel("LCOE (USD/kWh)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(title="City", fontsize=9)
    plt.xlim(0, 1)  # S’assurer que l’axe X va de 0 à 1
    plt.tight_layout()
    plt.show()