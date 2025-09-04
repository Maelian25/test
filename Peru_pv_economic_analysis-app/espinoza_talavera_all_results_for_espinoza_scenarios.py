import numpy as np
import pandas as pd
import os
from datetime import datetime
import numpy_financial as npf
import matplotlib.pyplot as plt
import re

# --- Select scenario and city ---
# This section is the only one meant to be modified by the user.
scenario = 2
city = "arequipa"
save_results = True  # whether to save results to Excel or not

# --- Import data ---
df_all = pd.read_csv("parameters/parameters_clean.csv", sep=";", decimal=",")

# Get all city_scenario columns
scenario_columns = [col for col in df_all.columns if "_" in col and re.match(r".+_\d+", col)]

# Define core functions (simplified for brevity)
def EPV(Hopt, P, PR): return P * Hopt * PR
def EPVs(epv, SCI): return epv * SCI
def EPVg(epv, SCI): return epv * (1 - SCI)
def PVI(P, Cu): return Cu * P
def PVOM(PVI, COM): return COM * PVI
def WACC(Xl, Xec, il, dec, T): return ((Xec/(Xec+Xl)) * dec + (Xl/(Xec+Xl)) * il * (1 - T)) * 100
def EPV_discounted(Hopt, P, PR, N, rd, d): return EPV(Hopt, P, PR) * sum(((1 - rd) / (1 + d)) ** i for i in range(1, N + 1))

def run_simplified_model(param_dict, source="talavera"):
    def val(name): return float(param_dict[name]) if name not in ["N", "Nd", "Nis", "Nl"] else int(param_dict[name])
    Hopt, P, PR, rd, El, N, SCI = val("Hopt"), val("P"), val("PR")/100, val("rd")/100, val("El"), val("N"), val("SCI")/100
    Cu, COM, d, pg, ps = val("Cu"), val("COM")/100, val("d")/100, val("pg"), val("ps")
    rpg, rps, rom, T = val("rpg")/100, val("rps")/100, val("rom")/100, val("T")/100
    Nd, Xd, Xl, Xec, Xis = val("Nd"), val("Xd")/100, val("Xl")/100, val("Xec")/100, val("Xis")/100
    il, Nis, Nl, dec = val("il")/100, val("Nis"), val("Nl"), val("dec")/100

    q = 1 / (1 + d)
    Kp = (1 + rom) / (1 + d)
    Ks = (1 + rps) * (1 - rd) / (1 + d)
    Kg = (1 + rpg) * (1 - rd) / (1 + d)

    epv = EPV(Hopt, P, PR)
    epvs = EPVs(epv, SCI)
    epvg = EPVg(epv, SCI)
    pvi = PVI(P, Cu)
    pvom = PVOM(pvi, COM)
    epv_disc = EPV_discounted(Hopt, P, PR, N, rd, d)
    wacc = WACC(Xl, Xec, il, dec, T)

    pwci = (ps * epvs * (1 - T) * Ks * (1 - Ks**N) / (1 - Ks)) + (pg * epvg * (1 - T) * Kg * (1 - Kg**N) / (1 - Kg)) if source == "talavera" else (ps * epvs * Ks * (1 - Ks**N) / (1 - Ks)) + (pg * epvg * Kg * (1 - Kg**N) / (1 - Kg))
    PWPVOM = pvom * (1 - T) * Kp * (1 - Kp**N) / (1 - Kp) if source == "talavera" else pvom * Kp * (1 - Kp**N) / (1 - Kp)
    DEP = 0 if Nd == 0 else pvi * Xd / Nd
    PWDEP = DEP * q * (1 - q ** Nd) / (1 - q)
    pwco = pvi + PWPVOM - PWDEP * T

    lcoe = pwco / epv_disc
    npv = pwci - pwco

    return {
        "Model": source,
        "LCOE (USD/kWh)": round(lcoe, 4),
        "NPV (USD)": round(npv, 2),
        "WACC (%)": round(wacc, 2)
    }

# Run simulations and save results
results = []
for column_key in scenario_columns:
    param_dict = dict(zip(df_all["parameter"], df_all[column_key]))
    city, scenario = column_key.split("_")
    scenario = int(scenario)

    talavera_result = run_simplified_model(param_dict, source="talavera")
    espinoza_result = run_simplified_model(param_dict, source="espinoza")

    for result in [talavera_result, espinoza_result]:
        results.append({
            "City": city,
            "Scenario": scenario,
            **result
        })

# Export to Excel if enabled
if save_results:
    df_all_results = pd.DataFrame(results)
    filename = f"results_all_models_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.xlsx"
    df_all_results.to_excel(filename, index=False)
    print(f"Results saved to: {filename}")
else:
    print(pd.DataFrame(results))
