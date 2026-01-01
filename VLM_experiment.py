experiment_data = {
    "general_parameters": {
        "operational_conditions": {
            "pressure": 0.5,  # MPa
            "alpha": 1.09E-3,  # C^-1
        },
        "dimensions": {
            "channel_depth": 0.0001,  # m
            "small_channels": {
                "nr_channels_per_chip": 0,
                "total_surface_area": 0,  # m²
                "d1": 8.2,  # μm, measured values
                "d2": 76.6,  # μm, measured values
                "channel_width": 0.0000684  # m, d2-d1
            },
            "large_channels": {
                "nr_channels_per_chip": 5,
                "total_surface_area": 3.78e-5,  # m², 5.40e-6 times 7 modules
                "d1": 76.6,  # μm, measured values
                "d2": 289.7,  # μm, measured values
                "channel_width": 0.0002498  # m, d2-d1
            },
            "heater_chip": {
                "width": 12,  # μm
                "length": 3000,  # μm
                "sheet_resistance": 2,
                "type1": {
                    "sets": 3,
                    "lines": 7,
                    "resistance": 3.4  # ohm
                },
                "type2": {
                    "sets": 2,
                    "lines": 15,
                    "resistance": 2.38  # ohm
                }
            }
        }
    },
    "thruster_chip_5": {
        "throat_width": 23.1,  # μm
        "temperature": 423.03,  # K
        "pressure": 4.8,  # bar
        "power_heater": 7.29,  # W
        "mass_flow": 0.55,  # mg/s
        "environmental_loss": 5.82,  # W
        "E": 13.16,  # Jmg-1
        "efficiency": 0.2,
        "channel_type": "large_serpentine"
    },
    "thruster_chip_7": {
        "throat_width": 16.5,  # μm
        "pressure": 5.15,  # bar
        "power_heater": 8.76,  # W
        "mass_flow": 0.75,  # mg/s
        "environmental_loss": 6.0,  # W
        "E": 11.71,  # Jmg-1
        "efficiency": 0.23,
        "channel_type": "large_serpentine"
    },
    "thruster_chip_9": {
        "throat_width": 20.9,  # μm
        "pressure": 5.15,  # bar
        "power_heater": 8.19,  # W
        "mass_flow": 0.83,  # mg/s
        "environmental_loss": 6.0,  # W
        "channel_type": "diamond"
    },
    "thruster_chip_10": {
        "throat_width": 23.6,  # μm
        "pressure": 5.00,  # bar
        "power_heater": 7.72,  # W
        "mass_flow": 0.61,  # mg/s
        "environmental_loss": 6.0,  # W
        "channel_type": "diamond"
    }
}