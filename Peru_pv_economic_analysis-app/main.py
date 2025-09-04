# Formulas from Talavera et al. (2019)
# Economic evaluation of a photovoltaic self-consumption project

import numpy as np

def npv(ci, co):
    return ci - co

def irr(cashflows):
    return np.irr(cashflows)

def lcoe(pw_co, epv, rd, d, n):
    denom = sum([(1 - rd)**t / (1 + d)**t for t in range(1, n + 1)])
    return pw_co / (epv * denom)

def cs(lcoe, epv, epvs, rd, d, n):
    num = lcoe * epv * sum([(1 - rd)**t / (1 + d)**t for t in range(1, n + 1)])
    denom = epvs * sum([1 / (1 + d)**t for t in range(1, n + 1)])
    return num / denom

def pw_cash_inflows(ps, epvs, T, Ks, N, pg=0, epvg=0, Kg=0):
    inflow_self = (ps * epvs * (1 - T)) * (1 - Ks**N) / (1 - Ks)
    inflow_grid = (pg * epvg * (1 - T)) * (1 - Kg**N) / (1 - Kg)
    return inflow_self + inflow_grid

def pw_cash_outflows(pvi, pvom, T, Kp, N, depy=0, Nd=0, d=0):
    opex = (pvom * (1 - T)) * (1 - Kp**N) / (1 - Kp)
    if Nd > 0:
        q = 1 / (1 + d)
        dep = depy * q * (1 - q**Nd) / (1 - q)
    else:
        dep = 0
    return pvi + opex - dep * T

def profitability_index(pw_ci, pw_opex, dep, T, pvi):
    return (pw_ci - pw_opex + dep * T) / pvi

def modified_profitability_index(npv_value, pvi):
    return npv_value / pvi

def discounted_payback_time(cashflows, d):
    cum_cashflow = 0
    for i, cf in enumerate(cashflows):
        cum_cashflow += cf / (1 + d)**(i + 1)
        if cum_cashflow >= 0:
            return i + 1
    return None

if __name__ == "__main__":
    # --- Technical Parameters ---
    Hopt = float(input("Hopt global irradiation in optimum inclined plane (kWh/m2*year): "))
    P = float(input("P (peak power) (kWp): "))
    PR = float(input("PR (performance ratio) (%): ")) / 100
    rd = float(input("rd (annual degredation rate) (%/year): ")) / 100
    El = float(input("El (annual household electricity consumption) (kWh/year): "))
    N = int(input("N (life cycle) (years): "))
    SCI = float(input("SCI (self consumption index) (%): ")) / 100
    opex = float(input("Co&m (OPEX) (pourcentage of PVI): ")) / 100

    # --- Economic Parameters ---
    Cu = float(input("Cu (initial investment by kWp) (USD/kWp): "))
    d = float(input("d (discount rate) (%): ")) / 100
    pg = float(input("pg (price at which electricity is sold to the grid) (USD/kWh): "))
    ps = float(input("ps (price at which electricity is self consumed) (USD/kWh): "))
    rpg = float(input("rpg (annual escalation rate of pg) (%): ")) / 100
    rps = float(input("rps (annual escalation rate of ps) (%): ")) / 100
    rOM = float(input("rom (annual escalation rate of opex) (%): ")) / 100
    T = float(input("T (income tax rate) (%): ")) / 100
    Nd = int(input("Nd (period of amortization of the investment for tax purposes) (years): "))
    Xd = float(input("Xd (total depreciation allowance) (%): ")) / 100

    # --- Financial Parameters ---
    Xl = float(input("Xl (part of PVi financed with loan) (%): ")) / 100
    Xec = float(input("Xec (part of PVi financed with equity capital) (%): ")) / 100
    Xis = float(input("Xis (part of PVi financed with subsidy or grant) (%): ")) / 100
    il = float(input("il (annual loan interest) (%): ")) / 100
    Nis = int(input("Nis (amortization of investment subsidy) (years): "))
    Nl = int(input("Nl (amortization of loan) (years): "))
    dec = float(input("dec (annual dividend of the equity capital or return on equity) (%): ")) / 100

    # --- Auxiliary Calculations ---
    Ks = (1 + rps) * (1 - rd) / (1 + d)
    Kg = (1 + rpg) * (1 - rd) / (1 + d)
    Kp = (1 + rOM) / (1 + d)

    # --- Computation ---
    pw_ci = pw_cash_inflows(ps, epvs, T, Ks, N, pg, epvg, Kg)
    pw_co = pw_cash_outflows(pvi, pvom, T, Kp, N, depy, Nd, d)

    lcoe_val = lcoe(pw_co, epv, rd, d, N)
    cs_val = cs(lcoe_val, epv, epvs, rd, d, N)
    npv_val = npv(pw_ci, pw_co)
    pi_val = profitability_index(pw_ci, pvom, depy, T, pvi)
    pim_val = modified_profitability_index(npv_val, pvi)

    print("\n--- Results ---")
    print(f"LCOE: {lcoe_val:.4f} €/kWh")
    print(f"Cs (cost of self-consumed electricity): {cs_val:.4f} €/kWh")
    print(f"NPV: {npv_val:.2f} €")
    print(f"Profitability Index (PI): {pi_val:.3f}")
    print(f"Modified Profitability Index (PIM): {pim_val:.3f}")
