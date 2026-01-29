"""
Simple calculation of convective heat transfer coefficient for chamber heating

Step 1: Calculate Reynolds number
Step 2: Calculate Nusselt number and convective heat transfer coefficient
Step 3: Calculate temperature distribution along chamber and create figure
"""

from iapws import IAPWS97
from VLM_experiment import experiment_data
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# GIVEN PARAMETERS
# ============================================================================
# Chamber dimensions
CHAMBER_HEIGHT = 100e-6  # m (100 μm)
CHAMBER_WIDTH = 2979e-6  # m (2979 μm)
CHAMBER_LENGTH = 2.56e-3  # m (2.56 mm)

# Operational conditions
pressure_bar = 4.8  # bar
pressure_mpa = pressure_bar / 10.0  # MPa
T_inlet = 20.0 + 273.15  # K (inlet temperature, 20°C)

# Import mass flow rate from VLM_experiment.py
mass_flow_mg_s = experiment_data["thruster_chip_5"]["mass_flow"]  # mg/s
mass_flow_kg_s = mass_flow_mg_s / 1e6  # kg/s

# ============================================================================
# STEP 1: CALCULATE REYNOLDS NUMBER
# ============================================================================
# Calculate hydraulic diameter
# D_h = 2 * W * H / (W + H)
D_h = 2 * CHAMBER_WIDTH * CHAMBER_HEIGHT / (CHAMBER_WIDTH + CHAMBER_HEIGHT)  # m

# Get fluid properties at inlet conditions
water = IAPWS97(T=T_inlet, P=pressure_mpa)
rho = water.rho  # kg/m³
mu = water.mu  # Pa·s (dynamic viscosity)

# Calculate cross-sectional area
A_cross = CHAMBER_HEIGHT * CHAMBER_WIDTH  # m²

# Calculate average flow velocity
# U = G / (rho * A)
# where G is the mass flow rate, rho is the density, and A is the cross-sectional area
U = mass_flow_kg_s / (rho * A_cross)  # m/s

# Calculate Reynolds number
# Re = rho * U * D_h / mu
Re = rho * U * D_h / mu

# ============================================================================
# STEP 2: CALCULATE NUSSELT NUMBER AND CONVECTIVE HEAT TRANSFER COEFFICIENT
# ============================================================================
# Calculate aspect ratio
# alpha = H / W
alpha = CHAMBER_HEIGHT / CHAMBER_WIDTH

# Calculate Nusselt number
# Nu = 8.235 * (1 - 2.0421*alpha + 3.0853*alpha^2 - 2.4765*alpha^3 + 1.0578*alpha^4 - 0.1861*alpha^5)
Nu = 8.235 * (1 - 2.0421*alpha + 3.0853*alpha**2 - 2.4765*alpha**3 + 1.0578*alpha**4 - 0.1861*alpha**5)

# Get thermal conductivity
k = water.k  # W/(m·K)

# Calculate convective heat transfer coefficient
# Nu = h * D_h / k
# Rearranging: h = Nu * k / D_h
h = Nu * k / D_h  # W/(m²·K)

# Result
print(f"\n \n Reynolds number Re: {Re:.2f} [-]")
print(f"Nusselt number Nu: {Nu:.2f} [-]")
print(f"Hydraulic diameter D_h (μm): {D_h*1e6:.2f} μm")
print(f"Convective heat transfer coefficient h: {h:.2f} W/(m²·K) \n\n")

T_wall = 424  # K (wall temperature)
wetted_perimeter = 2 * (CHAMBER_HEIGHT + CHAMBER_WIDTH)  # m (wetted perimeter per unit length)

# Get specific heat capacity at inlet conditions
cp = water.cp * 1000  # J/(kg·K) (convert from kJ/(kg·K) to J/(kg·K))

# Calculate saturation temperature
liq_sat = IAPWS97(P=pressure_mpa, x=0)
T_sat = liq_sat.T  # K

# ============================================================================
# STEP 3: CALCULATE TEMPERATURE DISTRIBUTION
# ============================================================================
# Temperature distribution equation: ΔT(x) = ΔT(x=0) * exp(-hPx/(m_dot*cp))
# where ΔT(x) = T_wall - T(x) and ΔT(x=0) = T_wall - T_inlet
# Therefore: T(x) = T_wall - (T_wall - T_inlet) * exp(-hPx/(m_dot*cp))

# Calculate temperature difference at x=0
delta_T_0 = T_wall - T_inlet  # K

# Solve for L where T = T_sat
# T_sat = T_wall - (T_wall - T_inlet) * exp(-hPx/(m_dot*cp))
# Rearranging: L = -ln[(T_wall - T_sat)/(T_wall - T_inlet)] * (m_dot*cp)/(h*P)
ratio = (T_wall - T_sat) / (T_wall - T_inlet)
L_sat = -np.log(ratio) * (mass_flow_kg_s * cp) / (h * wetted_perimeter)  # m

print(f"Length to reach saturation: L = {L_sat*1e3:.4f} mm")

# Only calculate and plot up to saturation temperature
# Use the minimum of L_sat and CHAMBER_LENGTH
x_max = min(L_sat, CHAMBER_LENGTH)  # m
x = np.linspace(0, x_max, 1000)  # m

# Calculate temperature distribution
# ΔT(x) = ΔT(x=0) * exp(-hPx/(m_dot*cp))
delta_T = delta_T_0 * np.exp(-h * wetted_perimeter * x / (mass_flow_kg_s * cp))

# Calculate actual temperature: T(x) = T_wall - ΔT(x)
T = T_wall - delta_T  # K

# ============================================================================
# CREATE FIGURE
# ============================================================================
fig, ax = plt.subplots(figsize=(8, 6))

# Plot temperature distribution
ax.plot(x * 1e3, T, 'b-', linewidth=2, label='Fluid temperature')

# Add saturation temperature line
ax.axhline(y=T_sat, color='r', linestyle='--', linewidth=1.5, label=f'Saturation temperature ({T_sat:.1f} K)')

# Mark the point where saturation is reached
ax.plot(L_sat * 1e3, T_sat, 'ro', markersize=8, label=f'Saturation reached at {L_sat*1e3:.2f} mm')

# Add wall temperature line
ax.axhline(y=T_wall, color='k', linestyle=':', linewidth=1, alpha=0.5, label=f'Wall temperature ({T_wall} K)')

# Formatting
ax.set_xlabel('Position along chamber, $x$ [mm]', fontsize=12)
ax.set_ylabel('Temperature, $T$ [K]', fontsize=12)
ax.set_title('Temperature Distribution in VLM Thruster Chamber', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

# Set x-axis limits to show only up to saturation point
ax.set_xlim(0, x_max * 1e3)

# Add text annotation with key parameters (positioned in bottom right, above legend)
textstr = f'$\\dot{{m}}$ = {mass_flow_mg_s} mg/s\n$h$ = {h/1e4:.2f}×10$^4$ W/(m²·K)\n$T_w$ = {T_wall} K\n$T_{{inlet}}$ = {T_inlet:.1f} K'
ax.text(0.98, 0.25, textstr, transform=ax.transAxes, fontsize=9,
        verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Add legend (positioned below the text box)
ax.legend(loc='lower right', fontsize=10)

plt.tight_layout()

# Save figure
figure_path = '../reporting/figures/chamber_temperature_distribution.png'
plt.savefig(figure_path, dpi=300, bbox_inches='tight')
print(f"\nFigure saved to: {figure_path}")

