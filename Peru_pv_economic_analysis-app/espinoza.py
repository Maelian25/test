import numpy as np
import pandas as pd
import os
from datetime import datetime
import numpy_financial as npf
import matplotlib.pyplot as plt

# --- Select scenario and city ---
scenario = 2
city = "arequipa"
save_results = False  # whether to save results to Excel or not

# --- Import data ---
df = pd.read_csv("parameters/parameters_clean.csv", sep=";", decimal=",")

column_key = f"{city.lower()}_{scenario}"
if column_key not in df.columns:
    available = [col for col in df.columns if "_" in col]
    raise ValueError(
        f"\n Error: The column '{column_key}' does not exist in the parameter file.\n"
        f" Available combinations: {', '.join(available)}\n"
        f"  Please check the 'city' and 'scenario' inputs."
    )

param_dict = dict(zip(df["parameter"], df[column_key]))

# --- Parameters ---
Hopt = float(param_dict["Hopt"])
P = float(param_dict["P"])
PR = float(param_dict["PR"]) / 100
rd = float(param_dict["rd"]) / 100
El = float(param_dict["El"])
N = int(param_dict["N"])
SCI = float(param_dict["SCI"]) / 100
Cu = float(param_dict["Cu"])
COM = float(param_dict["COM"]) / 100
d = float(param_dict["d"]) / 100
pg = float(param_dict["pg"])
ps = float(param_dict["ps"])
rpg = float(param_dict["rpg"]) / 100
rps = float(param_dict["rps"]) / 100
rom = float(param_dict["rom"]) / 100
T = float(param_dict["T"]) / 100
Nd = int(param_dict["Nd"])
Xd = float(param_dict["Xd"]) / 100
Xl = float(param_dict["Xl"]) / 100
Xec = float(param_dict["Xec"]) / 100
Xis = float(param_dict["Xis"]) / 100
il = float(param_dict["il"]) / 100
Nis = int(param_dict["Nis"])
Nl = int(param_dict["Nl"])
dec = float(param_dict["dec"]) / 100

# --- Shared factors ---
q = 1 / (1 + d)
Kp = (1 + rom) / (1 + d)
Ks = (1 + rps) * (1 - rd) / (1 + d)
Kg = (1 + rpg) * (1 - rd) / (1 + d)

# --- Functions ---
def EPV(Hopt, P, PR):
    return P * Hopt * PR

def EPVs(epv, SCI):
    return epv * SCI

def EPVg(epv, SCI):
    return epv * (1 - SCI)

def PVI(P, Cu):
    return Cu * P

def PVOM(PVI, COM):
    return COM * PVI

def PWCO(pvi, pvom, N, T, Xd, Nd, Xl, Xec, Xis, il, dec, Nis, Nl):
    PWPVOM = pvom * Kp * (1 - Kp**N) / (1 - Kp)
    DEP = 0 if Nd == 0 else pvi * Xd / Nd
    PWDEP = DEP * q * (1 - q ** Nd) / (1 - q)
    PVl = Xl * pvi
    PVec = Xec * pvi
    PVis = Xis * pvi
    loan_term = (PVl * il) / (1 - (1 + il) ** (-Nl)) * (q * (1 - q ** Nl)) / (1 - q) if Xl > 0 else 0
    equity_term = dec * PVec * (q * (1 - q ** N)) / (1 - q) if Xec > 0 else 0
    dividend_term = PVec * (q ** N) if Xec > 0 else 0
    subsidy_term = (PVis / Nis) * T * (q * (1 - q ** Nis)) / (1 - q) if Xis > 0 and Nis > 0 else 0
    return loan_term + equity_term + dividend_term + subsidy_term + PWPVOM

def PWCI(ps, pg, N, epvs, epvg):
    return (ps * epvs * Ks * (1 - Ks**N) / (1 - Ks)) + (pg * epvg * Kg * (1 - Kg**N) / (1 - Kg))

def WACC(Xl, Xec, il, dec):
    return ((Xec/(Xec+Xl)) * dec + (Xl/(Xec+Xl)) * il) * 100

def EPV_discounted(Hopt, P, PR, N, rd, d):
    factor = (1 - rd) / (1 + d)
    return EPV(Hopt, P, PR) * sum(factor ** i for i in range(1, N + 1))

def LCOE(pvi, Hopt, P, PR, rd, d, N, rom, pvom):
    epv_discounted = EPV_discounted(Hopt, P, PR, N, rd, d)
    pwco = PWCO(pvi, pvom, N, T, Xd, Nd, Xl, Xec, Xis, il, dec, Nis, Nl)
    return pwco / epv_discounted

def NPV(pvi, pvom, pg, ps, rpg, rps, rd, N, epvs, epvg):
    pwci = PWCI(ps, pg, N, epvs, epvg)
    pwco = PWCO(pvi, pvom, N, T, Xd, Nd, Xl, Xec, Xis, il, dec, Nis, Nl)
    return pwci - pwco

def build_cashflow_series(pvi, epvs, epvg, ps, pg, rps, rpg, rd, d, N, pvom):
    flows = [-pvi]
    for n in range(1, N+1):
        inflation_ps = ps * ((1 + rps) ** (n - 1))
        inflation_pg = pg * ((1 + rpg) ** (n - 1))
        energy_self = epvs * ((1 - rd) ** n)
        energy_grid = epvg * ((1 - rd) ** n)
        income = (inflation_ps * energy_self + inflation_pg * energy_grid)
        opex = pvom * ((1 + rom) ** (n - 1))
        net = income - opex
        flows.append(net)
    return flows

def compute_irr(cashflows):
    return npf.irr(cashflows)

def compute_dpbt(pvi, cashflows, d):
    discounted_sum = 0
    for year, cf in enumerate(cashflows[1:], start=1):
        discounted_sum += cf / ((1 + d) ** year)
        if discounted_sum >= pvi:
            return year
    return None

def run_simulation():
    pvi = PVI(P, Cu)
    epv = EPV(Hopt, P, PR)
    epvs = EPVs(epv, SCI)
    epvg = EPVg(epv, SCI)
    pvom = PVOM(pvi, COM)
    epv_discounted = EPV_discounted(Hopt, P, PR, N, rd, d)
    wacc_result = WACC(Xl, Xec, il, dec)
    pwco_result = PWCO(pvi, pvom, N, T, Xd, Nd, Xl, Xec, Xis, il, dec, Nis, Nl)
    pwci_result = PWCI(ps, pg, N, epvs, epvg)
    lcoe_result = LCOE(pvi, Hopt, P, PR, rd, d, N, rom, pvom)
    npv_result = NPV(pvi, pvom, pg, ps, rpg, rps, rd, N, epvs, epvg)
    cashflows = build_cashflow_series(pvi, epvs, epvg, ps, pg, rps, rpg, rd, d, N, pvom)
    irr_result = compute_irr(cashflows)
    dpbt_result = compute_dpbt(pvi, cashflows, d)

    return {
        "PVI (USD)": round(pvi, 3),
        "EPV (kWh)": round(epv, 3),
        "EPVs (kWh)": round(epvs, 3),
        "EPVg (kWh)": round(epvg, 3),
        "EPV discounted (kWh)": round(epv_discounted, 3),
        "WACC (%)": round(wacc_result, 3),
        "PWCO (USD)": round(pwco_result, 3),
        "PWCI (USD)": round(pwci_result, 3),
        "LCOE (USD/kWh)": round(lcoe_result, 4),
        "NPV (USD)": round(npv_result, 3),
        "IRR (%)": round(irr_result * 100, 2) if irr_result is not None else None,
        "DPBT (years)": dpbt_result if dpbt_result is not None else "Not recovered",
    }

def save_results_to_excel(city, scenario, parameters_dict, espinoza_result, save=True):
    if not save:
        return
    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M")
    filename = f"results_{city.lower()}_{scenario}.xlsx"
    sheet_name = timestamp
    data = {
        "Parameter": list(parameters_dict.keys()) + list(espinoza_result.keys()),
        "Value": list(parameters_dict.values()) + list(espinoza_result.values())
    }
    df_out = pd.DataFrame(data)
    with pd.ExcelWriter(filename, mode='a' if os.path.exists(filename) else 'w', engine='openpyxl') as writer:
        df_out.to_excel(writer, sheet_name=sheet_name, index=False)

def plot_lcoe_vs_tariff(lcoe_espinoza, base_tariff, rps, start_year=2018, N=25):
    years = np.arange(start_year, start_year + N)
    tariff_projection = base_tariff * (1 + rps) ** (years - start_year)

    plt.figure(figsize=(10, 6))
    plt.plot(years, [lcoe_espinoza] * len(years), label="LCOE (Espinoza)", linestyle="--", color="green")
    plt.plot(years, tariff_projection, label="Projected Electricity Tariff", color="gray")
    plt.title("LCOE vs Electricity Tariff")
    plt.xlabel("Year")
    plt.ylabel("USD/kWh")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    espinoza_result = run_simulation()

    df_results = pd.DataFrame([espinoza_result])
    print("\n--- ESPINOZA MODEL RESULTS ---")
    print(df_results)

    save_results_to_excel(
        city=city,
        scenario=scenario,
        parameters_dict=param_dict,
        espinoza_result=espinoza_result,
        save=save_results
    )

    plot_lcoe_vs_tariff(
        lcoe_espinoza=espinoza_result["LCOE (USD/kWh)"],
        base_tariff=ps,
        rps=rps,
        start_year=2018,
        N=N
    )
