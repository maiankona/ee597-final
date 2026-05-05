"""
AdaSpray: Adaptive Spray & Wait Routing for ICMNs

Implements and compares four protocols using PyWiSim's encounter model:
  1. Hold-and-Deliver  (baseline: K=1)
  2. Epidemic Flooding (baseline: K=inf)
  3. Fixed Spray-and-Wait (fixed K)
  4. AdaSpray — adaptive K via EWMA encounter-rate estimation

Usage:
    python3 adaSpray.py

Outputs:
    results.json  — raw per-trial results for all protocols/configs
    run plot_results.py to generate figs
"""

import random
import math
import json
from collections import defaultdict


# minimal encounter-based simulator based on PyWiSim EncounterManager API

class Simulator:
    def __init__(self, n_nodes, duration, encounter_rate, seed=42):
        self.n = n_nodes
        self.duration = duration
        self.encounter_rate = encounter_rate  
        self.rng = random.Random(seed)
        self.time = 0.0
        self.nodes = {}   
        self.log = []    

    def add_node(self, node):
        self.nodes[node.nid] = node
        node.sim = self

    def run(self):
        t = 0.0
        while t < self.duration:
            dt = self.rng.expovariate(self.encounter_rate)
            t += dt
            if t >= self.duration:
                break
            self.time = t
            nids = list(self.nodes.keys())
            a, b = self.rng.sample(nids, 2)
            node_a = self.nodes[a]
            node_b = self.nodes[b]
            node_a.on_encounter(node_b)
            node_b.on_encounter(node_a)


# ─────────────────────────────────────────────────────────────────────────────
# Base Node
# ─────────────────────────────────────────────────────────────────────────────

class BaseNode:
    def __init__(self, nid, source, destination):
        self.nid = nid
        self.source = source        
        self.destination = destination  
        self.sim = None
        self.has_packet = (nid == source)
        self.delivery_time = None   
        self.transmissions = 0      

    def on_encounter(self, other):
        raise NotImplementedError

    @property
    def delivered(self):
        return self.delivery_time is not None


# ─────────────────────────────────────────────────────────────────────────────
# Hold and Deliver
# ──────────-──────────────────────────────────────────────────────────────────
class HoldAndDeliverNode(BaseNode):
    def on_encounter(self, other):
        if self.nid == self.source and self.has_packet:
            if other.nid == self.destination:
                self.transmissions += 1
                other.has_packet = True
                other.delivery_time = self.sim.time


# ─────────────────────────────────────────────────────────────────────────────
# Epidemic Flooding
# ─────────────────────────────────────────────────────────────────────────────

class EpidemicNode(BaseNode):
    def on_encounter(self, other):
        if self.has_packet:
            self.transmissions += 1
            if not other.has_packet:
              other.has_packet = True
              if other.nid == self.destination:
                other.delivery_time = self.sim.time


# ─────────────────────────────────────────────────────────────────────────────
# Spray & Wait
# ─────────────────────────────────────────────────────────────────────────────

class FixedSprayNode(BaseNode):
    """
    From lecture 18
    Spray phase: node with m>=2 gives floor(m/2) copies to encountered node.
    Wait phase: node with 1 copy only delivers directly to destination.
    """
    def __init__(self, nid, source, destination, K):
        super().__init__(nid, source, destination)
        self.K = K
        self.copies = K if nid == source else 0

    def on_encounter(self, other):
        if self.copies >= 1 and other.nid == self.destination:
            self.transmissions += 1
            other.has_packet = True
            if other.delivery_time is None:
                other.delivery_time = self.sim.time
            return

        # spray phase
        if self.copies >= 2 and other.copies == 0:
            give = self.copies // 2
            self.transmissions += 1
            other.copies = give
            other.has_packet = True
            self.copies -= give


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive Spray & Wait
# ─────────────────────────────────────────────────────────────────────────────

class AdaSprayNode(BaseNode):
    """
    Calculation based on lecture 14/15 Exponential Weighted Moving Average Estimator -->
    
    Each node maintains an EWMA estimate (ETX) of its local encounter rate.
    K is recomputed at EVERY ENCOUNTER as:

        K = clip(round(K_base * (lambda_ref / lambda_hat)), K_min, K_max)

    Intuition:
      - sparse network (small lambda_hat) -> fewer encounters -> need MORE copies
        to ensure delivery -> K increases
      - dense network (large lambda_hat) -> many encounters -> fewer copies needed
        -> K decreases

    The forwarding rule same as normal Spray & Wait
    """

    def __init__(self, nid, source, destination,
                 K_base=4, lambda_ref=1.0, alpha=0.3,
                 K_min=1, K_max=16):
        super().__init__(nid, source, destination)
        self.K_base = K_base          # copy budget at reference rate
        self.lambda_ref = lambda_ref  # reference encounter rate
        self.alpha = alpha            # EWMA smoothing factor (0=slow, 1=instant)
        self.K_min = K_min
        self.K_max = K_max

        # state
        self.copies = K_base if nid == source else 0
        self.lambda_hat = lambda_ref  # initial estimate = reference
        self.last_encounter_time = 0.0
        self.k_history = []          

    def _update_rate(self):
        """Update EWMA encounter rate estimate."""
        t = self.sim.time
        dt = t - self.last_encounter_time
        if dt > 0:
            lambda_instant = 1.0 / dt
            # EWMA ETX: new = alpha * instant + (1-alpha) * old
            self.lambda_hat = (self.alpha * lambda_instant
                               + (1 - self.alpha) * self.lambda_hat)
        self.last_encounter_time = t

    def _compute_K(self):
        """Compute adaptive K from current encounter rate estimate."""
        if self.lambda_hat <= 0:
            return self.K_max
        # inverse relationship: sparse = more copies
        k = round(self.K_base * (self.lambda_ref / self.lambda_hat))
        k = max(self.K_min, min(self.K_max, k))
        self.k_history.append((self.sim.time, k))
        return k

    def on_encounter(self, other):
        self._update_rate()
        current_K = self._compute_K()

        # rebudget copies toward current K
        if self.copies > current_K:
            self.copies = current_K  # adapt downward immediately

        if self.copies >= 1 and other.nid == self.destination:
            self.transmissions += 1
            other.has_packet = True
            if other.delivery_time is None:
                other.delivery_time = self.sim.time
            return

        # spray
        if self.copies >= 2 and isinstance(other, AdaSprayNode) and other.copies == 0:
            give = self.copies // 2
            self.transmissions += 1
            other.copies = give
            other.has_packet = True
            self.copies -= give


# ─────────────────────────────────────────────────────────────────────────────
# Experiment runs
# ─────────────────────────────────────────────────────────────────────────────

def run_trial(protocol, n_nodes, encounter_rate, duration, K=None,
              K_base=4, lambda_ref=1.0, alpha=0.3, seed=42):
    source = 'N0'
    destination = f'N{n_nodes - 1}'

    sim = Simulator(n_nodes, duration, encounter_rate, seed=seed)

    for i in range(n_nodes):
        nid = f'N{i}'
        if protocol == 'hold':
            node = HoldAndDeliverNode(nid, source, destination)
        elif protocol == 'epidemic':
            node = EpidemicNode(nid, source, destination)
        elif protocol == 'fixed':
            node = FixedSprayNode(nid, source, destination, K=K)
        elif protocol == 'adaspray':
            node = AdaSprayNode(nid, source, destination,
                                K_base=K_base, lambda_ref=lambda_ref,
                                alpha=alpha)
        else:
            raise ValueError(f"Unknown protocol: {protocol}")
        sim.add_node(node)

    sim.run()

    dest_node = sim.nodes[destination]
    total_tx = sum(n.transmissions for n in sim.nodes.values())

    k_history = []
    if protocol == 'adaspray':
        src_node = sim.nodes[source]
        k_history = src_node.k_history

    return {
        'delivered': dest_node.delivered,
        'delivery_time': dest_node.delivery_time if dest_node.delivered else duration,
        'transmissions': total_tx,
        'k_history': k_history,
    }


def run_experiment(n_trials=30, n_nodes=10, duration=100.0,
                   encounter_rates=None, fixed_K_values=None,
                   K_base=4, lambda_ref=1.0, alpha=0.3):
    if encounter_rates is None:
        encounter_rates = [0.2, 0.5, 1.0, 2.0, 4.0]
    if fixed_K_values is None:
        fixed_K_values = [2, 4, 8]

    results = defaultdict(lambda: defaultdict(list))

    protocols = ['hold', 'epidemic', 'adaspray']
    protocols += [f'fixed_K{k}' for k in fixed_K_values]

    total = len(protocols) * len(encounter_rates) * n_trials
    done = 0

    for rate in encounter_rates:
        for trial in range(n_trials):
            seed = trial * 1000 + int(rate * 100)

            r = run_trial('hold', n_nodes, rate, duration, seed=seed)
            results['hold'][rate].append(r)

            r = run_trial('epidemic', n_nodes, rate, duration, seed=seed)
            results['epidemic'][rate].append(r)

            r = run_trial('adaspray', n_nodes, rate, duration,
                          K_base=K_base, lambda_ref=lambda_ref, alpha=alpha,
                          seed=seed)
            results['adaspray'][rate].append(r)

            for k in fixed_K_values:
                r = run_trial('fixed', n_nodes, rate, duration, K=k, seed=seed)
                results[f'fixed_K{k}'][rate].append(r)

            done += len(protocols)
            pct = 100 * done / total
            print(f"\r  Progress: {pct:.0f}%", end='', flush=True)

    print()
    return results


def summarize(results):
    summary = {}
    for proto, rate_dict in results.items():
        summary[proto] = {}
        for rate, trials in rate_dict.items():
            delivered = [t['delivered'] for t in trials]
            times = [t['delivery_time'] for t in trials if t['delivered']]
            txs = [t['transmissions'] for t in trials]

            n = len(trials)
            delivery_ratio = sum(delivered) / n
            mean_delay = (sum(times) / len(times)) if times else float('nan')
            mean_tx = sum(txs) / n

            # std dev
            def std(lst):
                if len(lst) < 2:
                    return 0.0
                m = sum(lst) / len(lst)
                return math.sqrt(sum((x - m)**2 for x in lst) / (len(lst) - 1))

            summary[proto][rate] = {
                'delivery_ratio': delivery_ratio,
                'mean_delay': mean_delay,
                'std_delay': std(times) if times else 0.0,
                'mean_tx': mean_tx,
                'std_tx': std(txs),
                'n_trials': n,
            }
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("AdaSpray Experiment")
    print("===================")

    ENCOUNTER_RATES = [0.2, 0.5, 1.0, 2.0, 4.0]
    FIXED_K_VALUES  = [2, 4, 8]
    N_NODES         = 10
    DURATION        = 150.0
    N_TRIALS        = 50
    K_BASE          = 4      
    LAMBDA_REF      = 1.0    
    ALPHA           = 0.3    

    print(f"Nodes: {N_NODES}, Duration: {DURATION}, Trials: {N_TRIALS}")
    print(f"Encounter rates: {ENCOUNTER_RATES}")
    print(f"AdaSpray: K_base={K_BASE}, lambda_ref={LAMBDA_REF}, alpha={ALPHA}")
    print()

    results = run_experiment(
        n_trials=N_TRIALS,
        n_nodes=N_NODES,
        duration=DURATION,
        encounter_rates=ENCOUNTER_RATES,
        fixed_K_values=FIXED_K_VALUES,
        K_base=K_BASE,
        lambda_ref=LAMBDA_REF,
        alpha=ALPHA,
    )

    summary = summarize(results)

    print("\nDelivery Ratio:")
    print(f"{'Rate':<8}", end='')
    protos = ['hold', 'epidemic', 'adaspray'] + [f'fixed_K{k}' for k in FIXED_K_VALUES]
    for p in protos:
        print(f"{p:<14}", end='')
    print()
    for rate in ENCOUNTER_RATES:
        print(f"{rate:<8.1f}", end='')
        for p in protos:
            v = summary[p][rate]['delivery_ratio']
            print(f"{v:<14.3f}", end='')
        print()

    print("\nMean Delay (delivered packets only):")
    print(f"{'Rate':<8}", end='')
    for p in protos:
        print(f"{p:<14}", end='')
    print()
    for rate in ENCOUNTER_RATES:
        print(f"{rate:<8.1f}", end='')
        for p in protos:
            v = summary[p][rate]['mean_delay']
            print(f"{v:<14.2f}", end='')
        print()

    print("\nMean Transmissions:")
    print(f"{'Rate':<8}", end='')
    for p in protos:
        print(f"{p:<14}", end='')
    print()
    for rate in ENCOUNTER_RATES:
        print(f"{rate:<8.1f}", end='')
        for p in protos:
            v = summary[p][rate]['mean_tx']
            print(f"{v:<14.2f}", end='')
        print()
# plotting stuff
    def jsonify(d):
        if isinstance(d, dict):
            return {str(k): jsonify(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [jsonify(x) for x in d]
        else:
            return d

    with open('results.json', 'w') as f:
        json.dump(jsonify(summary), f, indent=2)
    print("\nResults saved to results.json")
