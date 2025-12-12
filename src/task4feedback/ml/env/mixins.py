import torch
import task4feedback.trip as trip

class SimulationLookaheadMixin:
    def run_lookahead(self, steps: int, drain: bool = True, disable_mapper: bool = True) -> float:
        """
        Returns the time of the simulation copy.
        """
        sim_copy = self.simulator.copy()
        
        if disable_mapper:
            # Run with internal mapper (default is EFT if unset in simulator_factory)
            sim_copy.disable_external_mapper()
            
        if steps > 0:
            sim_copy.set_steps(steps)
            sim_copy.run()
            
        if drain:
            sim_copy.start_drain()
            sim_copy.run()
            
        return sim_copy.time

