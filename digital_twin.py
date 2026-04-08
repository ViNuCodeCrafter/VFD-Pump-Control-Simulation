import numpy as np

class PumpDigitalTwin:
    def __init__(self, rated_pressure=6.0, rated_flow=100.0, static_head=1.5):
        """
        Initializes a Digital Twin of a Centrifugal Pump.
        :param rated_pressure: Max pressure at 100% speed (bar)
        :param rated_flow: Max flow at 100% speed (LPM)
        :param static_head: Minimum pressure required to reach the top floor (bar)
        """
        self.rated_pressure = rated_pressure
        self.rated_flow = rated_flow
        self.static_head = static_head

    def simulate(self, speed_percent: float):
        """
        Simulates the pump behavior using Affinity Laws.
        Input: Speed (0-100%) 
        Output: Simulated Pressure, Flow, and Power Consumption
        """
        # Ensure speed is within bounds
        n = max(0.0, min(100.0, speed_percent)) / 100.0
        
        # Affinity Law: Pressure (Head) is proportional to the square of speed
        # P = P_rated * (n^2)
        sim_pressure = self.rated_pressure * (n**2)
        
        # Add static head offset (pressure won't drop below atmospheric/static level)
        sim_pressure = max(self.static_head, sim_pressure)
        
        # Affinity Law: Flow is proportional to speed
        sim_flow = self.rated_flow * n
        
        # Affinity Law: Power is proportional to the cube of speed (Energy Savings Logic)
        # Power = P_rated * (n^3)
        sim_power_kw = 5.5 * (n**3) # Assuming a 5.5kW motor
        
        return {
            "simulated_pressure_bar": round(sim_pressure, 2),
            "simulated_flow_lpm": round(sim_flow, 2),
            "energy_consumption_kw": round(sim_power_kw, 2),
            "status": "Running" if n > 0.2 else "Standby (Idle)"
        }

if __name__ == "__main__":
    # Example usage
    twin = PumpDigitalTwin()
    print("--- Digital Twin Simulation Results ---")
    for speed in [20, 50, 80, 100]:
        print(f"Speed {speed}%: {twin.simulate(speed)}")
