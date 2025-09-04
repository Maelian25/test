import numpy as np


# --- Functions ---
def calculate_npv():
    # Revenus actualisés
    PWCI_self = ps * EPVs * (1 - T) * Ks * (1 - Ks**N) / (1 - Ks)
    PWCI_grid = pg * EPVg * (1 - T) * Kg * (1 - Kg**N) / (1 - Kg)
    PWCI = PWCI_self + PWCI_grid

    # Coûts actualisés
    PMPVOM = PVom * (1 - T) * Kp * (1 - Kp**N) / (1 - Kp)
    PWCO = PVI + PMPVOM - PWDEP * T

    NPV = PWCI - PWCO
    return NPV, PWCO


def calculate_lcoe(PWCO):
    sum_LCOE = 0
    for i in range(1, N):
        sum_LCOE += ((1 - rd) ** i) / ((1 + d) ** i)

    LCOE = PWCO / (EPV * sum_LCOE)
    return LCOE



if __name__ == "__main__":
    # --- Technical parameters ---
    Hopt = float(input("Hopt global irradiation in optimum inclined plane (kWh/m2*year): "))
    P = float(input("P (peak power) (kWp): "))
    PR = float(input("PR (performance ratio) (%): ")) / 100
    rd = float(input("rd (annual degradation rate) (%/year): ")) / 100
    El = float(input("El (annual household electricity consumption) (kWh/year): "))
    N = int(input("N (life cycle) (years): "))
    SCI = float(input("SCI (self consumption index) (%): ")) / 100
    opex = float(input("Co&m (OPEX) (percentage of PVI): ")) / 100
    EPV = float(input("Annual PV electricity generated: "))

    # --- Economic Parameters ---
    #Cu = float(input("Cu (initial investment by kWp) (USD/kWp): "))
    PVI = float(input("PVI (Initial investment cost of the PV system) (USD): "))
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

    # --- Intermediate Calculations ---
    EPVs = EPV * SCI
    EPVg = EPV - EPVs
    Ks = (1 + rps) * (1 - rd) / (1 + d)
    Kg = (1 + rpg) * (1 - rd) / (1 + d)
    Cu = PVI / P
    PVom = opex * PVI
    Kp = (1 + rOM) / (1 + d)
    q = 1 / (1 + d)
    DEP = Xd * PVI / Nd
    PWDEP = DEP * q * (1 - q**Nd) / (1 - q)



# --- Results ---
npv_result, pwco_result = calculate_npv()
lcoe_result = calculate_lcoe(pwco_result)

print(f"\nNPV: {npv_result:.2f} USD")
print(f"LCOE: {lcoe_result:.4f} USD/kWh")
