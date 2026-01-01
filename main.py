from iapws import IAPWS97
import numpy as np
import matplotlib.pyplot as plt
from VLM_experiment import experiment_data

"""
VLM Thruster Chip System
Single class for thruster chips with serpentine microchannels
"""


class ThrusterChip:
    """VLM Thruster Chip containing heater, dimensions, and microchannels"""
    
    def __init__(
        self,
        power_heater_input: float,
        power_environmental_loss: float,
        # overall_dimensions: tuple,  # (length, width, height) in meters
        mass_flow_mg_s: float,
        num_channels_per_chip: int,
        total_surface_area_per_module: float,
        height: float,
        width: float
    ):
        """
        Initialize a thruster chip
        
        Parameters:
        -----------
        power_heater_input : float
            Heater input power in Watts
        power_environmental_loss : float
            Environmental loss (could be in Watts or as a fraction)
        overall_dimensions : tuple
            Overall chip dimensions (length, width, height) in meters
        mass_flow_mg_s : float
            Overall mass flow rate m_dot in mg/s
        num_channels_per_chip : int
            Number of channels per chip
        total_surface_area_per_module : float
            Total surface area per module in m²
        height : float
            Channel height in meters
        width : float
            Channel width in meters
        """
        self.power_heater_input = power_heater_input  # W
        self.power_environmental_loss = power_environmental_loss
        self.mass_flow_mg_s = mass_flow_mg_s  # mg/s
        self.num_channels_per_chip = num_channels_per_chip
        self.total_surface_area_per_module = total_surface_area_per_module  # m²
        self.height = height  # m
        self.width = width  # m
        
        # Calculate cross-sectional area per channel
        self.cross_sectional_area = self.height * self.width  # A_channel = H × W
    
    def get_G_channel_mass_flow(self) -> float:
        """
        Get the channel-specific mass flow per unit area (G)
        
        Returns:
        --------
        float
            Mass flow per unit area G in kg/(m²·s)
        """
        # Convert mg/s to kg/s and divide by number of channels and cross-sectional area
        mass_flow_kg_s = self.mass_flow_mg_s / 1e6  # Convert mg/s to kg/s
        return mass_flow_kg_s / self.num_channels_per_chip / self.cross_sectional_area
    
    def calculate_uniform_heat_flux(self) -> float:
        """
        Calculate uniform heat flux q
        Equation: q = (P_heater - P_lost) / A_channel [kW/m²]
        where A_channel is the total surface area per module (A_large)
        
        Returns:
        --------
        float
            Uniform heat flux q in kW/m²
        """
        # Heat flux = power / surface area
        # Power available after environmental loss
        net_power = self.power_heater_input - self.power_environmental_loss
        # Use total surface area per module (A_large)
        # Convert W to kW
        heat_flux_kw_m2 = (net_power / self.total_surface_area_per_module) / 1e3  # Convert W/m² to kW/m²
        return heat_flux_kw_m2
    
    def calculate_nusselt_number(self, serpentine: bool = False, E_Nu: float = 2.0) -> float:
        """
        Calculate Nusselt number for single phase laminar flow
        Equation: Nu = 8.235(1 - 2.0421α + 3.0853α² - 2.4765α³ + 1.0578α⁴ - 0.1861α⁵)
        where α = H / W
        
        For serpentine channels, applies enhancement factor: Nu_serpentine = Nu_straight × E_Nu
        
        Parameters:
        -----------
        serpentine : bool
            Whether to apply serpentine enhancement factor (default: False)
        E_Nu : float
            Nusselt number enhancement factor for serpentine channels (default: 2.0)
        
        Returns:
        --------
        float
            Nusselt number Nu [-]
        """
        # Aspect ratio: α = H / W
        alpha = self.height / self.width
        
        # Nusselt number: Nu = 8.235(1 - 2.0421α + 3.0853α² - 2.4765α³ + 1.0578α⁴ - 0.1861α⁵)
        Nu = 8.235 * (1 - 2.0421 * alpha + 3.0853 * alpha**2 - 2.4765 * alpha**3 + 
                      1.0578 * alpha**4 - 0.1861 * alpha**5)
        
        # Apply serpentine enhancement if requested
        if serpentine:
            Nu *= E_Nu
            
        return Nu
    
    def calculate_dean_number(self, Re: float, R_curvature: float) -> float:
        """
        Calculate Dean number for serpentine channels
        Equation: K = Re × sqrt(D_h / R_c)
        
        Parameters:
        -----------
        Re : float
            Reynolds number [-]
        R_curvature : float
            Radius of curvature [m]
        
        Returns:
        --------
        float
            Dean number K [-]
        """
        import math
        # Calculate hydraulic diameter
        D_h = 2 * self.height * self.width / (self.height + self.width)
        
        # Dean number: K = Re × sqrt(D_h / R_c)
        K = Re * math.sqrt(D_h / R_curvature)
        return K
    
    def calculate_serpentine_vaporization_length(self, L_chi_1_straight: float, E_Nu: float = 2.0) -> float:
        """
        Calculate vaporization length for serpentine channels
        Equation: L_χ=1,serpentine = L_χ=1,straight / E_Nu
        
        Parameters:
        -----------
        L_chi_1_straight : float
            Vaporization length for straight channel [m]
        E_Nu : float
            Nusselt number enhancement factor (default: 2.0)
        
        Returns:
        --------
        float
            Vaporization length for serpentine channel [m]
        """
        return L_chi_1_straight / E_Nu
    
    def calculate_serpentine_results(self, results_straight: dict, E_Nu: float = 2.0, 
                                     n_positions: int = 100) -> dict:
        """
        Calculate serpentine channel results based on straight channel results
        The vapor quality distribution follows the same linear relationship but with shorter L_chi_1
        
        Parameters:
        -----------
        results_straight : dict
            Results dictionary from vapour_quality() for straight channel
        E_Nu : float
            Nusselt number enhancement factor (default: 2.0)
        n_positions : int
            Number of positions along the channel (default: 100)
        
        Returns:
        --------
        dict
            Dictionary with serpentine channel results, same structure as vapour_quality()
        """
        import numpy as np
        
        # Calculate serpentine vaporization length
        L_chi_1_serpentine = results_straight['L_chi_1'] / E_Nu
        
        # Calculate serpentine Nusselt number
        Nu_serpentine = self.calculate_nusselt_number(serpentine=True, E_Nu=E_Nu)
        
        # Generate z positions from 0 to L_chi_1_serpentine
        z_positions = np.linspace(0, L_chi_1_serpentine, n_positions)
        
        # Extract parameters from straight channel results
        chi_0 = results_straight['chi_0']
        G = results_straight['G']
        tau = results_straight['tau']
        rho_l = results_straight.get('rho_l', 917.0)  # Default if not in dict
        rho_v = results_straight.get('rho_v', 2.67)   # Default if not in dict
        
        # Initialize arrays
        chi_values = np.zeros(n_positions)
        L_l_values = np.zeros(n_positions)
        L_v_values = np.zeros(n_positions)
        U_values = np.zeros(n_positions)
        t_l_values = np.zeros(n_positions)
        t_v_values = np.zeros(n_positions)
        
        # Calculate all quantities at each position (same equations as straight channel)
        for i, z in enumerate(z_positions):
            # Vapour quality: χ = χ_0 + (z / L_χ=1) × (1 - χ_0)
            chi = chi_0 + (z / L_chi_1_serpentine) * (1 - chi_0)
            
            # Liquid slug length: L_l = (G × τ × (1-χ)) / ρ_l
            L_l = (G * tau * (1 - chi)) / rho_l
            
            # Vapour slug length: L_v = (G × τ × χ) / ρ_v
            L_v = (G * tau * chi) / rho_v
            
            # Flow velocity: U = (G × χ) / ρ_v + (G × (1-χ)) / ρ_l
            U = (G * chi) / rho_v + (G * (1 - chi)) / rho_l
            
            # Liquid slug resident time
            if chi >= 1.0:
                t_l = 0.0
            else:
                denominator_tl = 1 + (rho_l / rho_v) * (chi / (1 - chi))
                t_l = tau / denominator_tl
            
            # Vapour slug resident time
            if chi <= 0.0:
                t_v = 0.0
            else:
                denominator_tv = 1 + (rho_v / rho_l) * ((1 - chi) / chi)
                t_v = tau / denominator_tv
            
            # Store values
            chi_values[i] = chi
            L_l_values[i] = L_l
            L_v_values[i] = L_v
            U_values[i] = U
            t_l_values[i] = t_l
            t_v_values[i] = t_v
        
        # Return dictionary with same structure as straight channel results
        return {
            'z': z_positions,
            'chi': chi_values,
            'L_l': L_l_values,
            'L_v': L_v_values,
            'U': U_values,
            't_l': t_l_values,
            't_v': t_v_values,
            'q': results_straight['q'],
            'G': results_straight['G'],
            'D_h': results_straight['D_h'],
            'Bo': results_straight['Bo'],
            'tau': results_straight['tau'],
            'frequency': results_straight['frequency'],
            'L_l0': results_straight['L_l0'],
            'L_v0': results_straight['L_v0'],
            'chi_0': results_straight['chi_0'],
            'L_chi_1': L_chi_1_serpentine,
            'alpha': results_straight['alpha'],
            'Nu': Nu_serpentine,
            'E_Nu': E_Nu,
            'serpentine': True
        }
    
    def calculate_dimensionless_numbers(self, rho_l: float, rho_v: float, mu_l: float, 
                                        sigma: float, h_lv: float, U_mean: float) -> dict:
        """
        Calculate dimensionless numbers for two-phase flow analysis
        
        Parameters:
        -----------
        rho_l : float
            Liquid density [kg/m³]
        rho_v : float
            Vapour density [kg/m³]
        mu_l : float
            Dynamic viscosity of liquid [Pa·s]
        sigma : float
            Surface tension [N/m]
        h_lv : float
            Latent heat of vaporization [kJ/kg]
        U_mean : float
            Mean flow velocity [m/s]
        
        Returns:
        --------
        dict
            Dictionary with keys: 'We', 'Ca', 'Ja', 'D_h'
        """
        # Get channel-specific mass flow
        G = self.get_G_channel_mass_flow()
        
        # Calculate hydraulic diameter
        D_h = 2 * self.height * self.width / (self.height + self.width)
        
        # Calculate heat flux
        q = self.calculate_uniform_heat_flux() * 1000  # Convert kW/m² to W/m²
        
        # Calculate surface area and cross-sectional area per channel
        A_c = self.height * self.width  # Cross-sectional area
        # For a rectangular channel, surface area per unit length = perimeter
        perimeter = 2 * (self.height + self.width)
        
        # Weber number: We = D_h * G² / (ρ_l * σ)
        We = (D_h * G**2) / (rho_l * sigma)
        
        # Capillary number: Ca = μ_l * U / σ
        Ca = (mu_l * U_mean) / sigma
        
        # Jakob number: Ja = q * A_s / (A_c * U * ρ_v * h_lv)
        # For unit length: A_s/length = perimeter, so A_s = perimeter (per unit length)
        h_lv_J_kg = h_lv * 1000  # Convert kJ/kg to J/kg
        Ja = (q * perimeter) / (A_c * U_mean * rho_v * h_lv_J_kg)
        
        return {
            'We': We,
            'Ca': Ca,
            'Ja': Ja,
            'D_h': D_h,
            'G': G,
            'U_mean': U_mean
        }
    
    def vapour_quality(self, rho_l: float, rho_v: float, h_lv: float, n_positions: int = 100) -> dict:
        """
        Calculate vapour quality and related quantities along the channel
        
        Parameters:
        -----------
        rho_l : float
            Liquid density [kg/m³]
        rho_v : float
            Vapour density [kg/m³]
        h_lv : float
            Latent heat of vaporization [kJ/kg]
        n_positions : int
            Number of positions along the channel (default: 100)
        
        Returns:
        --------
        dict
            Dictionary with keys: 'z', 'chi', 'L_l', 'L_v', 'U', 't_l', 't_v', 
            'q', 'G', 'D_h', 'Bo', 'tau', 'frequency', 'L_l0', 'L_v0', 'chi_0', 
            'L_chi_1', 'alpha', 'Nu'
            Position-dependent arrays contain n_positions values
            Single values are scalars
        """
        import math
        
        # Get channel-specific mass flow
        G = self.get_G_channel_mass_flow()
        
        # Calculate heat flux
        q = self.calculate_uniform_heat_flux()  # kW/m²
        
        # Calculate hydraulic diameter: D_h = 2 × H × W / (H + W)
        D_h = 2 * self.height * self.width / (self.height + self.width)
        
        # Calculate Boiling number: Bo = q / (G × h_lv)
        # Convert q from kW/m² to W/m² and h_lv from kJ/kg to J/kg
        q_W_m2 = q * 1000  # Convert kW/m² to W/m²
        h_lv_J_kg = h_lv * 1000  # Convert kJ/kg to J/kg
        Bo = q_W_m2 / (G * h_lv_J_kg)
        
        # Calculate bubble formation period: f = 1/τ = 0.0491(1000 × Bo)^6.8195 × G / (ρ_l × D_h)
        # Therefore: τ = 1 / f
        #frequency = 0.0491 * ((1000 * Bo) ** 6.8195) * G / (rho_l * D_h)
        
        # Bubble formation frequency (hardcoded)
        frequency = 4.0  # Hz
        # Bubble formation period: τ = 1 / f
        tau = 1.0 / frequency

        # Calculate initial conditions
        # Initial liquid slug length: L_l0 = (G × τ) / ρ_l
        L_l0 = (G * tau) / rho_l
        
        # Initial vapour slug length: L_v0 = (π × H²) / (6 × W)
        L_v0 = (math.pi * self.height**2) / (6 * self.width)
        
        # Initial vapour quality: χ_0 = 1 / (1 + (6 × G × τ × W) / (ρ_v × π × H²))
        denominator_chi0 = 1 + (6 * G * tau * self.width) / (rho_v * math.pi * self.height**2)
        chi_0 = 1 / denominator_chi0
        
        # Length at vapour quality of 1: L_χ=1 = (G × (1-χ_0) × W × H × h_lv) / (q × 2 × (W+H))
        numerator_L_chi1 = G * (1 - chi_0) * self.width * self.height * h_lv_J_kg
        denominator_L_chi1 = q_W_m2 * 2 * (self.width + self.height)
        L_chi_1 = numerator_L_chi1 / denominator_L_chi1
        
        # Calculate aspect ratio and Nusselt number
        alpha = self.height / self.width
        Nu = self.calculate_nusselt_number()
        
        # Generate z positions from 0 to L_chi_1
        z_positions = np.linspace(0, L_chi_1, n_positions)
        
        # Initialize arrays for storing results
        chi_values = np.zeros(n_positions)
        L_l_values = np.zeros(n_positions)
        L_v_values = np.zeros(n_positions)
        U_values = np.zeros(n_positions)
        t_l_values = np.zeros(n_positions)
        t_v_values = np.zeros(n_positions)
        
        # Calculate all quantities at each position
        for i, z in enumerate(z_positions):
            # Vapour quality: χ = χ_0 + (z / L_χ=1) × (1 - χ_0)
            chi = chi_0 + (z / L_chi_1) * (1 - chi_0)
            
            # Liquid slug length: L_l = (G × τ × (1-χ)) / ρ_l
            L_l = (G * tau * (1 - chi)) / rho_l
            
            # Vapour slug length: L_v = (G × τ × χ) / ρ_v
            L_v = (G * tau * chi) / rho_v
            
            # Flow velocity: U = (G × χ) / ρ_v + (G × (1-χ)) / ρ_l
            U = (G * chi) / rho_v + (G * (1 - chi)) / rho_l
            
            # Liquid slug resident time: t_l = τ / (1 + (ρ_l / ρ_v) × (χ / (1-χ)))
            if chi >= 1.0:
                t_l = 0.0
            else:
                denominator_tl = 1 + (rho_l / rho_v) * (chi / (1 - chi))
                t_l = tau / denominator_tl
            
            # Vapour slug resident time: t_v = τ / (1 + (ρ_v / ρ_l) × ((1-χ) / χ))
            if chi <= 0.0:
                t_v = 0.0
            else:
                denominator_tv = 1 + (rho_v / rho_l) * ((1 - chi) / chi)
                t_v = tau / denominator_tv
            
            # Store values
            chi_values[i] = chi
            L_l_values[i] = L_l
            L_v_values[i] = L_v
            U_values[i] = U
            t_l_values[i] = t_l
            t_v_values[i] = t_v
        
        # Return dictionary with plotting data and intermediate values
        return {
            'z': z_positions,
            'chi': chi_values,
            'L_l': L_l_values,
            'L_v': L_v_values,
            'U': U_values,
            't_l': t_l_values,
            't_v': t_v_values,
            # Intermediate values for printing
            'q': q,
            'G': G,
            'D_h': D_h,
            'Bo': Bo,
            'tau': tau,
            'frequency': frequency,
            'L_l0': L_l0,
            'L_v0': L_v0,
            'chi_0': chi_0,
            'L_chi_1': L_chi_1,
            'alpha': alpha,
            'Nu': Nu
        }


# Example usage
if __name__ == "__main__":
    # Load data from experiment_data for thruster chip 5
    chip_data = experiment_data["thruster_chip_5"]
    general_params = experiment_data["general_parameters"]
    large_channels = general_params["dimensions"]["large_channels"]
    
    # Extract operational parameters
    pressure_bar = chip_data["pressure"]  # bar
    pressure_mpa = pressure_bar / 10.0  # Convert bar to MPa (1 bar = 0.1 MPa)
    power_heater = chip_data["power_heater"]  # W
    environmental_loss = chip_data["environmental_loss"]  # W
    mass_flow_mg_s = chip_data["mass_flow"]  # mg/s
    
    # Extract channel dimensions from large_channels
    channel_height = general_params["dimensions"]["channel_depth"]  # m
    channel_width = large_channels["channel_width"]  # m
    
    # Map channel type: "large_serpentine" -> "large"
    channel_type_str = chip_data["channel_type"]
    if "large" in channel_type_str.lower():
        channel_type = "large"
    else:
        channel_type = channel_type_str.lower()
    
    # Calculate IAPWS97 properties at the operating pressure
    p = pressure_mpa  # MPa
    
    # Saturated liquid
    liq = IAPWS97(P=p, x=0)
    
    # Saturated vapor
    vap = IAPWS97(P=p, x=1)
    
    T_sat = liq.T                # K
    rho_liq = liq.rho            # kg/m3
    rho_vap = vap.rho            # kg/m3
    lambda_liq = liq.k           # W/m·K
    lambda_vap = vap.k           # W/m·K
    h_liq = liq.h                # kJ/kg
    h_vap = vap.h                # kJ/kg
    latent_heat = h_vap - h_liq  # kJ/kg
    
    # Calculate Prandtl number: Pr = μ * cp / k
    # where μ is dynamic viscosity [Pa·s], cp is specific heat [J/(kg·K)], k is thermal conductivity [W/(m·K)]
    dynamic_viscosity_liq = liq.mu      # Pa·s
    specific_heat_liq = liq.cp          # kJ/(kg·K)
    thermal_conductivity_liq = liq.k    # W/(m·K)
    # Convert cp from kJ/(kg·K) to J/(kg·K) by multiplying by 1000
    prandtl_liq = dynamic_viscosity_liq * specific_heat_liq * 1000 / thermal_conductivity_liq  # Prandtl number [-]
    
    # Get surface tension at saturation conditions
    surface_tension = liq.sigma  # N/m

    thruster = ThrusterChip(
        power_heater_input=power_heater,
        power_environmental_loss=environmental_loss,
        mass_flow_mg_s=mass_flow_mg_s,
        num_channels_per_chip=large_channels["nr_channels_per_chip"],
        total_surface_area_per_module=large_channels["total_surface_area"],
        height=channel_height,
        width=channel_width
    )
    
    # Print operational conditions
    print("=== Operational Conditions ===")
    print(f"Pressure: {pressure_bar} bar ({pressure_mpa} MPa)")
    print(f"Saturation temperature: {T_sat:.2f} K")
    print(f"Heater input power P_heater: {power_heater} W")
    print(f"Environmental loss P_lost: {environmental_loss} W")
    print(f"Net power available: {power_heater - environmental_loss} W")
    print(f"Mass flow rate m_dot: {mass_flow_mg_s} mg/s")
    print(f"Liquid density ρ_l: {rho_liq:.2f} kg/m³")
    print(f"Vapour density ρ_v: {rho_vap:.2f} kg/m³")
    print(f"Latent heat of vaporization h_lv: {latent_heat:.2f} kJ/kg")
    print(f"Dynamic viscosity μ_l: {dynamic_viscosity_liq:.4e} Pa·s")
    print(f"Liquid Prandtl number Pr: {prandtl_liq:.4f} [-]")
    
    # Print channel geometry
    print("\n=== Channel Geometry ===")
    print(f"Channel type: {channel_type}")
    print(f"Number of channels per chip N_channels: {thruster.num_channels_per_chip}")
    print(f"Channel height H: {thruster.height * 1e6:.2f} μm ({thruster.height:.2e} m)")
    print(f"Channel width W: {thruster.width * 1e6:.2f} μm ({thruster.width:.2e} m)")
    print(f"Cross-sectional area per channel A_channel: {thruster.cross_sectional_area:.2e} m²")
    print(f"Total surface area per module A_channels: {thruster.total_surface_area_per_module:.2e} m²")
    
    # Calculate vapour quality data using the combined function
    # (includes calculations for hydraulic diameter, boiling number, and bubble formation period)
    print("\n=== Calculated Channel-Specific Operational Conditions ===")
    n_positions = 100
    results = thruster.vapour_quality(rho_liq, rho_vap, latent_heat, n_positions)
    
    # Calculate Dean number K using average flow velocity along the channel
    import math
    # Get average flow velocity from the calculated results along the channel
    U_mean = np.mean(results['U'])  # Average flow velocity [m/s] along the channel
    
    # Get hydraulic diameter from results (already calculated)
    D_h = results['D_h']
    
    # Calculate Reynolds number: Re = ρ × U × D_h / μ
    Re = rho_liq * U_mean * D_h / dynamic_viscosity_liq
    
    # Calculate Dean number: K = Re × sqrt(D_h / R)
    # R is the radius of curvature of the serpentine channel, which is d2 from large_channels
    R_curvature = large_channels["d2"] * 1e-6  # Convert from μm to m
    K = thruster.calculate_dean_number(Re, R_curvature)  # Dean number [-]
    
    print(f"Average flow velocity U_mean: {U_mean:.4f} m/s")
    print(f"Reynolds number Re: {Re:.2f} [-]")
    print(f"Dean number K: {K:.2f} [-]")
    
    # Calculate dimensionless numbers for two-phase flow
    dim_numbers = thruster.calculate_dimensionless_numbers(
        rho_liq, rho_vap, dynamic_viscosity_liq, surface_tension, latent_heat, U_mean
    )
    print(f"\n=== Dimensionless Numbers for Two-Phase Flow ===")
    print(f"Weber number We: {dim_numbers['We']:.4e} [-]")
    print(f"Capillary number Ca: {dim_numbers['Ca']:.4e} [-]")
    print(f"Jakob number Ja: {dim_numbers['Ja']:.4e} [-]")
    
    # Print variables in the order they appear in chapter-3.tex
    
    print("\n=== Calculated Channel-Specific Operational Conditions ===")
    # Equation 80: Wall heat flux q
    print(f"Wall heat flux q: {results['q']:.2e} kW/m²")
    
    # Equation 88: Channel-specific mass flow G
    print(f"Channel-specific mass flow G: {results['G']:.2e} kg/(m²·s)")
    
    # Hydraulic diameter (needed for calculations)
    print(f"Hydraulic diameter D_h: {results['D_h']*1e6:.2f} μm")
    
    # Equation 107: Boiling number Bo
    print(f"Boiling number Bo: {results['Bo']:.2e} [-]")
    
    # Equation 101: Bubble formation frequency f and period τ
    print(f"Bubble formation frequency f: {results['frequency']:.2e} Hz")
    print(f"Bubble formation period τ: {results['tau']:.2e} s")
    
    # Equation 95: Initial liquid slug length L_l0
    print(f"Initial liquid slug length L_l0: {results['L_l0']*1e6:.2f} μm")
    
    # Equation 119: Initial vapour slug length L_v0
    print(f"Initial vapour slug length L_v0: {results['L_v0']*1e6:.2f} μm")
    
    # Equation 127: Initial vapour quality χ_0
    print(f"Initial vapour quality χ_0: {results['chi_0']:.4f} [-]")
    
    # Equation 135: Length at vapour quality of 1 L_χ=1
    print(f"Length at vapour quality of 1 L_χ=1: {results['L_chi_1']*1e6:.2f} μm")
    
    # Aspect ratio α for Nusselt number
    print(f"Aspect ratio α = H/W: {results['alpha']:.4f} [-]")
    
    # Equation 171: Nusselt number Nu
    print(f"Nusselt number Nu: {results['Nu']:.2f} [-]")
    
    # Calculate serpentine channel results
    print("\n=== SERPENTINE CHANNEL MODEL ===")
    E_Nu = 2.0  # Enhancement factor based on literature (conservative estimate)
    
    # Add density values to results for serpentine calculation
    results['rho_l'] = rho_liq
    results['rho_v'] = rho_vap
    
    # Calculate full serpentine results
    results_serpentine = thruster.calculate_serpentine_results(results, E_Nu=E_Nu, n_positions=n_positions)
    
    Nu_serpentine = results_serpentine['Nu']
    L_chi_1_serpentine = results_serpentine['L_chi_1']
    
    print(f"Nusselt enhancement factor E_Nu: {E_Nu:.2f} [-]")
    print(f"Nusselt number Nu_serpentine: {Nu_serpentine:.2f} [-]")
    print(f"Length at vapour quality of 1 L_χ=1,serpentine: {L_chi_1_serpentine*1e6:.2f} μm")
    print(f"Vaporization length reduction: {(1 - L_chi_1_serpentine/results['L_chi_1'])*100:.1f}%")
    print(f"Enhancement ratio: {results['L_chi_1']/L_chi_1_serpentine:.2f}×")
    
    print("\n=== Position-dependent values along the channel ===")
    print(f"Calculated at {n_positions} positions from z=0 to z=L_χ=1")
    
    # Extract data from both straight and serpentine results
    z_positions_straight = results['z']
    chi_values_straight = results['chi']
    L_l_values_straight = results['L_l']
    L_v_values_straight = results['L_v']
    U_values_straight = results['U']
    t_l_values_straight = results['t_l']
    t_v_values_straight = results['t_v']
    
    z_positions_serpentine = results_serpentine['z']
    chi_values_serpentine = results_serpentine['chi']
    L_l_values_serpentine = results_serpentine['L_l']
    L_v_values_serpentine = results_serpentine['L_v']
    U_values_serpentine = results_serpentine['U']
    t_l_values_serpentine = results_serpentine['t_l']
    t_v_values_serpentine = results_serpentine['t_v']
    
    # Convert z positions to micrometers for plotting
    z_um_straight = z_positions_straight * 1e6
    L_l_um_straight = L_l_values_straight * 1e6
    L_v_um_straight = L_v_values_straight * 1e6
    
    z_um_serpentine = z_positions_serpentine * 1e6
    L_l_um_serpentine = L_l_values_serpentine * 1e6
    L_v_um_serpentine = L_v_values_serpentine * 1e6
    
    # Create plots - 2x2 layout comparing straight vs serpentine
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot 1: Vapour quality comparison
    ax1 = axes[0, 0]
    ax1.plot(z_um_straight, chi_values_straight, 'b-', linewidth=2, label='Straight channel')
    ax1.plot(z_um_serpentine, chi_values_serpentine, 'r--', linewidth=2, label='Serpentine channel')
    ax1.set_xlabel('Axial position $z$ [μm]', fontsize=12)
    ax1.set_ylabel('Vapour quality $\\chi$ [-]', fontsize=12)
    ax1.set_title('Vapour Quality Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Liquid slug length comparison
    ax2 = axes[0, 1]
    ax2.plot(z_um_straight, L_l_um_straight, 'b-', linewidth=2, label='Straight channel')
    ax2.plot(z_um_serpentine, L_l_um_serpentine, 'r--', linewidth=2, label='Serpentine channel')
    ax2.set_xlabel('Axial position $z$ [μm]', fontsize=12)
    ax2.set_ylabel('Liquid slug length [μm]', fontsize=12)
    ax2.set_title('Liquid Slug Length Comparison', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Vapour slug length comparison
    ax3 = axes[1, 0]
    ax3.plot(z_um_straight, L_v_um_straight, 'b-', linewidth=2, label='Straight channel')
    ax3.plot(z_um_serpentine, L_v_um_serpentine, 'r--', linewidth=2, label='Serpentine channel')
    ax3.set_xlabel('Axial position $z$ [μm]', fontsize=12)
    ax3.set_ylabel('Vapour slug length [μm]', fontsize=12)
    ax3.set_title('Vapour Slug Length Comparison', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Flow velocity comparison
    ax4 = axes[1, 1]
    ax4.plot(z_um_straight, U_values_straight, 'b-', linewidth=2, label='Straight channel')
    ax4.plot(z_um_serpentine, U_values_serpentine, 'r--', linewidth=2, label='Serpentine channel')
    ax4.set_xlabel('Axial position $z$ [μm]', fontsize=12)
    ax4.set_ylabel('Flow velocity [m/s]', fontsize=12)
    ax4.set_title('Flow Velocity Comparison', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Create figures directory if it doesn't exist
    import os
    figures_dir = 'figures'
    os.makedirs(figures_dir, exist_ok=True)
    
    # Save figure with appropriate name
    figure_path = os.path.join(figures_dir, 'thruster_analysis_straight_vs_serpentine.png')
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    print(f"\nPlots saved to '{figure_path}'")
    plt.show()
    
    # Print comparison summary
    print("\n=== COMPARISON SUMMARY ===")
    print(f"Straight channel vaporization length: {results['L_chi_1']*1e6:.2f} μm")
    print(f"Serpentine channel vaporization length: {L_chi_1_serpentine*1e6:.2f} μm")
    print(f"Vaporization length reduction: {(1 - L_chi_1_serpentine/results['L_chi_1'])*100:.1f}%")
    print(f"Nusselt number enhancement: {Nu_serpentine/results['Nu']:.2f}×")
    
    # ===== SENSITIVITY ANALYSIS: THREE CHANNEL WIDTHS =====
    print("\n=== SENSITIVITY ANALYSIS: CHANNEL WIDTH VARIATION ===")
    width_factors = [0.5, 1.0, 1.5]  # 50% less, standard, 50% more
    width_labels = ['50% narrower', 'Standard', '50% wider']
    width_results = []
    
    for i, factor in enumerate(width_factors):
        test_width = channel_width * factor
        print(f"\n--- Width factor: {factor:.1f}x (W = {test_width*1e6:.2f} μm) ---")
        
        # Create thruster with modified width
        thruster_width = ThrusterChip(
            power_heater_input=power_heater,
            power_environmental_loss=environmental_loss,
            mass_flow_mg_s=mass_flow_mg_s,
            num_channels_per_chip=large_channels["nr_channels_per_chip"],
            total_surface_area_per_module=large_channels["total_surface_area"],
            height=channel_height,
            width=test_width
        )
        
        # Calculate results for this width
        results_width = thruster_width.vapour_quality(rho_liq, rho_vap, latent_heat, n_positions)
        results_width_serpentine = thruster_width.calculate_serpentine_results(
            results_width, E_Nu=E_Nu, n_positions=n_positions
        )
        
        # Store results
        width_results.append({
            'factor': factor,
            'width_um': test_width * 1e6,
            'width_label': width_labels[i],
            'results_straight': results_width,
            'results_serpentine': results_width_serpentine,
            'thruster': thruster_width
        })
        
        print(f"  Hydraulic diameter D_h: {results_width['D_h']*1e6:.2f} μm")
        print(f"  Straight L_chi_1: {results_width['L_chi_1']*1e6:.2f} μm")
        print(f"  Serpentine L_chi_1: {results_width_serpentine['L_chi_1']*1e6:.2f} μm")
        print(f"  Nusselt number Nu: {results_width['Nu']:.2f} [-]")
        print(f"  Serpentine Nu: {results_width_serpentine['Nu']:.2f} [-]")
    
    # Create comparison figure for three widths
    fig_widths, axes_widths = plt.subplots(2, 2, figsize=(12, 8))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, orange, green
    
    # Plot 1: Vapour quality
    ax1 = axes_widths[0, 0]
    for i, wr in enumerate(width_results):
        z_um = wr['results_serpentine']['z'] * 1e6
        chi = wr['results_serpentine']['chi']
        ax1.plot(z_um, chi, '-', linewidth=2, color=colors[i], 
                label=f"W = {wr['width_um']:.1f} μm ({wr['width_label']})")
    ax1.set_xlabel('Axial position $z$ [μm]', fontsize=12)
    ax1.set_ylabel('Vapour quality $\\chi$ [-]', fontsize=12)
    ax1.set_title('Vapour Quality: Channel Width Sensitivity', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Liquid slug length
    ax2 = axes_widths[0, 1]
    for i, wr in enumerate(width_results):
        z_um = wr['results_serpentine']['z'] * 1e6
        L_l_um = wr['results_serpentine']['L_l'] * 1e6
        ax2.plot(z_um, L_l_um, '-', linewidth=2, color=colors[i],
                label=f"W = {wr['width_um']:.1f} μm")
    ax2.set_xlabel('Axial position $z$ [μm]', fontsize=12)
    ax2.set_ylabel('Liquid slug length [μm]', fontsize=12)
    ax2.set_title('Liquid Slug Length: Channel Width Sensitivity', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Vapour slug length
    ax3 = axes_widths[1, 0]
    for i, wr in enumerate(width_results):
        z_um = wr['results_serpentine']['z'] * 1e6
        L_v_um = wr['results_serpentine']['L_v'] * 1e6
        ax3.plot(z_um, L_v_um, '-', linewidth=2, color=colors[i],
                label=f"W = {wr['width_um']:.1f} μm")
    ax3.set_xlabel('Axial position $z$ [μm]', fontsize=12)
    ax3.set_ylabel('Vapour slug length [μm]', fontsize=12)
    ax3.set_title('Vapour Slug Length: Channel Width Sensitivity', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Flow velocity
    ax4 = axes_widths[1, 1]
    for i, wr in enumerate(width_results):
        z_um = wr['results_serpentine']['z'] * 1e6
        U = wr['results_serpentine']['U']
        ax4.plot(z_um, U, '-', linewidth=2, color=colors[i],
                label=f"W = {wr['width_um']:.1f} μm")
    ax4.set_xlabel('Axial position $z$ [μm]', fontsize=12)
    ax4.set_ylabel('Flow velocity [m/s]', fontsize=12)
    ax4.set_title('Flow Velocity: Channel Width Sensitivity', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save width comparison figure
    figure_path_widths = os.path.join(figures_dir, 'thruster_analysis_width_sensitivity.png')
    plt.savefig(figure_path_widths, dpi=300, bbox_inches='tight')
    print(f"\nWidth sensitivity plots saved to '{figure_path_widths}'")
    plt.show()
    
    # Print table data for LaTeX
    print("\n=== TABLE DATA FOR LATEX ===")
    print("\\begin{tabular}{lccc}")
    print("    \\toprule")
    print("    \\textbf{Parameter} & \\textbf{50\\% narrower} & \\textbf{Standard} & \\textbf{50\\% wider} \\\\")
    print("    \\midrule")
    for i, wr in enumerate(width_results):
        if i == 0:
            print(f"    Channel width \$W\$ & {wr['width_um']:.1f} \$\\mu\$m & ", end='')
        elif i == 1:
            print(f"{wr['width_um']:.1f} \$\\mu\$m & ", end='')
        elif i == 2:
            print(f"{wr['width_um']:.1f} \$\\mu\$m \\\\")
    for i, wr in enumerate(width_results):
        if i == 0:
            print(f"    Hydraulic diameter \$D_h\$ & {wr['results_straight']['D_h']*1e6:.1f} \$\\mu\$m & ", end='')
        elif i == 1:
            print(f"{wr['results_straight']['D_h']*1e6:.1f} \$\\mu\$m & ", end='')
        elif i == 2:
            print(f"{wr['results_straight']['D_h']*1e6:.1f} \$\\mu\$m \\\\")
    for i, wr in enumerate(width_results):
        if i == 0:
            print(f"    Vaporisation length (straight) \$L_{{\\chi=1}}\$ & {wr['results_straight']['L_chi_1']*1e6:.0f} \$\\mu\$m & ", end='')
        elif i == 1:
            print(f"{wr['results_straight']['L_chi_1']*1e6:.0f} \$\\mu\$m & ", end='')
        elif i == 2:
            print(f"{wr['results_straight']['L_chi_1']*1e6:.0f} \$\\mu\$m \\\\")
    for i, wr in enumerate(width_results):
        if i == 0:
            print(f"    Vaporisation length (serpentine) \$L_{{\\chi=1,serpentine}}\$ & {wr['results_serpentine']['L_chi_1']*1e6:.0f} \$\\mu\$m & ", end='')
        elif i == 1:
            print(f"{wr['results_serpentine']['L_chi_1']*1e6:.0f} \$\\mu\$m & ", end='')
        elif i == 2:
            print(f"{wr['results_serpentine']['L_chi_1']*1e6:.0f} \$\\mu\$m \\\\")
    for i, wr in enumerate(width_results):
        if i == 0:
            print(f"    Nusselt number (straight) \$Nu\$ & {wr['results_straight']['Nu']:.2f} & ", end='')
        elif i == 1:
            print(f"{wr['results_straight']['Nu']:.2f} & ", end='')
        elif i == 2:
            print(f"{wr['results_straight']['Nu']:.2f} \\\\")
    for i, wr in enumerate(width_results):
        if i == 0:
            print(f"    Nusselt number (serpentine) \$Nu_{{serpentine}}\$ & {wr['results_serpentine']['Nu']:.2f} & ", end='')
        elif i == 1:
            print(f"{wr['results_serpentine']['Nu']:.2f} & ", end='')
        elif i == 2:
            print(f"{wr['results_serpentine']['Nu']:.2f} \\\\")
    print("    \\bottomrule")
    print("\\end{tabular}")
