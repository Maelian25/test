import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy_financial as npf

# === Fonctions économiques ===


def EPV(Hopt, P, PR):
    return P * Hopt * PR


def PVI(P, Cu):
    return Cu * P


def PVOM(PVI, COM):
    return COM * PVI


def EPV_discounted(Hopt, P, PR, N, rd, d):
    factor = (1 - rd) / (1 + d)
    sum_factors = sum(factor**i for i in range(1, N + 1))
    return EPV(Hopt, P, PR) * sum_factors


def PWCO(pvi, pvom, N, T, Xd, Nd, Xl, Xec, Xis, il, dec, Nis, Nl, source):
    q = 1 / (1 + d)
    Kp = (1 + rom) / (1 + d)
    PWPVOM = (
        pvom * (1 - T) * Kp * (1 - Kp**N) / (1 - Kp)
        if source == "talavera"
        else pvom * Kp * (1 - Kp**N) / (1 - Kp)
    )
    DEP = 0 if Nd == 0 else pvi * Xd / Nd
    PWDEP = DEP * q * (1 - q**Nd) / (1 - q)
    PVl = Xl * pvi
    PVec = Xec * pvi
    PVis = Xis * pvi
    loan_term = (
        (PVl * il * (1 - T))
        / (1 - (1 + il * (1 - T)) ** (-Nl))
        * q
        * (1 - q**Nl)
        / (1 - q)
        if Xl > 0
        else 0
    )
    equity_term = dec * PVec * q * (1 - q**N) / (1 - q) if Xec > 0 else 0
    dividend_term = PVec * (q**N) if Xec > 0 else 0
    subsidy_term = (
        (PVis / Nis) * T * q * (1 - q**Nis) / (1 - q) if Xis > 0 and Nis > 0 else 0
    )
    pvinv = loan_term + equity_term + dividend_term + subsidy_term
    return pvinv + PWPVOM - PWDEP * T


def LCOE(
    pvi,
    Hopt,
    P,
    PR,
    rd,
    d,
    N,
    rom,
    pvom,
    T,
    Xd,
    Nd,
    Xl,
    Xec,
    Xis,
    il,
    dec,
    Nis,
    Nl,
    source,
):
    epv_discounted = EPV_discounted(Hopt, P, PR, N, rd, d)
    pwco = PWCO(pvi, pvom, N, T, Xd, Nd, Xl, Xec, Xis, il, dec, Nis, Nl, source)
    return pwco / epv_discounted


# === Fichiers Excel ===
files = {"2019": "parameters/parameters_mater_2019.csv", "2025": "parameters/parameters_mater_2025.csv"}

results = []

for year, filepath in files.items():
    df = pd.read_csv(filepath, sep=";", decimal=",")
    all_scenarios = [col for col in df.columns if "_" in col]

    for col in all_scenarios:
        try:
            city, scenario = col.split("_")
            label = f"{city.title()}_{scenario.lower()}"
            values = dict(zip(df["parameter"], df[col]))

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

        except Exception as e:
            print(f"⚠️ Erreur dans {col} ({year}) : {e}")
            continue


        
        # Calculs
        pvi = PVI(P, Cu)
        pvom = PVOM(pvi, COM)
        lcoe = LCOE(
            pvi,
            Hopt,
            P,
            PR,
            rd,
            d,
            N,
            rom,
            pvom,
            T,
            Xd,
            Nd,
            Xl,
            Xec,
            Xis,
            il,
            dec,
            Nis,
            Nl,
            source="talavera",
        )

        results.append(
            {
                "Year": year,
                "City": city.title(),
                "Scenario": scenario.lower(),
                "Label": label,
                "LCOE": round(lcoe, 4),
                "ps": round(ps, 4),
            }
        )

# === DataFrame des résultats ===
df_results = pd.DataFrame(results).sort_values(["City", "Year"])

# Filtrage
perc_data = df_results[df_results["Scenario"] == "perc"]
hit_data = df_results[df_results["Scenario"] == "hit"]

# === Style général ===
plt.style.use("seaborn-v0_8-whitegrid")  # Look plus propre
colors = ["#1f77b4", "#ff7f0e"]  # Bleu & orange

fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharey=True)


def plot_scenario(ax, data, scenario_name):
    labels = [f"{row['City']} - {row['Year']}" for _, row in data.iterrows()]
    x = np.arange(len(data))
    width = 0.35

    bars1 = ax.bar(x - width / 2, data["LCOE"], width, label="LCOE", color=colors[0])
    bars2 = ax.bar(
        x + width / 2,
        data["ps"],
        width,
        label="Electricity tariff (ps)",
        color=colors[1],
    )

    # Ajouter les valeurs au-dessus des barres
    for bar in bars1:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{bar.get_height():.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    for bar in bars2:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{bar.get_height():.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # Axes et style
    ax.set_title(f"Scenario {scenario_name.upper()}", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.tick_params(axis="y", labelsize=9)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.3)


# Tracer les deux sous-graphiques
plot_scenario(axs[0], perc_data, "perc")
plot_scenario(axs[1], hit_data, "hit")

# Titre global et légende
fig.supylabel("USD/kWh", fontsize=12)
fig.suptitle(
    "LCOE vs Electricity Tariff Comparison — PERC and HIT Scenarios",
    fontsize=14,
    fontweight="bold",
)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
