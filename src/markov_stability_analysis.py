"""
markov_stability_analysis.py - Stabilitätsanalyse für Markov-Ketten
"""

import time
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from .markov_chain import MarkovChain
from src.subgraph import Subgraph


@dataclass
class MarkovState:
    """Repräsentiert einen Markov-Ketten Zustand mit Metadaten."""
    step: int
    timestamp: float
    markov_chain: MarkovChain
    transformation_name: Optional[str] = None
    transition_matrix: Optional[np.ndarray] = None
    adjacency_matrix: Optional[np.ndarray] = None
    stationary_distribution: Optional[np.ndarray] = None
    is_irreducible: bool = False
    is_aperiodic: bool = False
    is_ergodic: bool = False
    
    def __post_init__(self):
        """Berechnet Metriken beim Erstellen."""
        if self.transition_matrix is None:
            self.transition_matrix = self.markov_chain.get_transition_matrix()
        
        if self.adjacency_matrix is None:
            self.adjacency_matrix, _ = self.markov_chain.to_adjacency_matrix()
        
        # Convert numpy bools to Python bools
        self.is_irreducible = bool(self.markov_chain.is_irreducible())
        self.is_aperiodic = bool(self.markov_chain.is_aperiodic())
        self.is_ergodic = bool(self.markov_chain.is_ergodic())
        
        if self.is_ergodic:
            self.stationary_distribution = \
                self.markov_chain.compute_stationary_distribution()


@dataclass
class MarkovSequence:
    """Repräsentiert eine zusammenhängende Markov-Sequenz."""
    start_step: int
    end_step: int
    length: int
    states: List[MarkovState] = field(default_factory=list)
    is_monotone_irreducible: bool = False
    
    def __repr__(self):
        return (f"MarkovSequence(steps={self.start_step}-{self.end_step}, "
                f"length={self.length}, monotone_irreducible={self.is_monotone_irreducible})")


class MarkovStabilityAnalysis:
    """
    Analysiert die Stabilität von Markov-Ketten durch
    Transformation und Subgraph Algorithmus.
    """
    
    def __init__(self):
        """Initialisiert die Analyse."""
        self.states: List[MarkovState] = []
        self.subgraph_algo = Subgraph()
    
    def add_markov_chain(self, mc: MarkovChain, 
                        transformation_name: str = "Manual"):
        """
        Fügt eine Markov-Kette zur Analyse hinzu.
        
        Args:
            mc: Markov-Kette
            transformation_name: Name der angewendeten Transformation
        """
        state = MarkovState(
            step=len(self.states),
            timestamp=time.time(),
            markov_chain=mc.copy(),
            transformation_name=transformation_name
        )
        self.states.append(state)
    
    def analyze(self) -> Dict:
        """
        Führt die vollständige Stabilitätsanalyse durch.
        
        Returns:
            Dictionary mit Analyseergebnissen
        """
        if len(self.states) < 2:
            return {"error": "Nicht genug Zustände für Analyse"}
        
        print("\n" + "="*70)
        print("MARKOV-KETTEN STABILITÄTSANALYSE")
        print("="*70 + "\n")
        
        # Berechne Vergleichsmatrix
        n = len(self.states)
        comparison_matrix = np.zeros((n, n), dtype=int)
        
        print(f"Vergleiche {n} Markov-Ketten paarweise...")
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    comparison_matrix[i][j] = 1
                else:
                    decision, _ = self.subgraph_algo.compare_graphs_with_adj_list(
                        self.states[i].adjacency_matrix,
                        self.states[j].adjacency_matrix
                    )
                    # MC_i ist Subgraph von MC_j wenn decision "keep_B" ist
                    # (B ist states[j], also hat MC_j mehr/gleich Kanten wie MC_i)
                    if decision in ["keep_B", "equal", "equal_keep_B"]:
                        comparison_matrix[i][j] = 1
        
        # Finde längste Subgraph-Sequenzen
        sequences = self._find_longest_sequences(comparison_matrix)
        
        # Analysiere Irreduzibilität-Evolution
        irreducibility_evolution = self._analyze_irreducibility_evolution()
        
        # Analysiere Ergodizität-Evolution
        ergodicity_evolution = self._analyze_ergodicity_evolution()
        
        # Analysiere Stabilität stationärer Verteilungen
        stationary_stability = self._analyze_stationary_stability()
        
        return {
            "total_states": n,
            "comparison_matrix": comparison_matrix,
            "longest_sequences": sequences,
            "irreducibility_evolution": irreducibility_evolution,
            "ergodicity_evolution": ergodicity_evolution,
            "stationary_stability": stationary_stability,
            "states": self.states
        }
    
    def _find_longest_sequences(self, 
                                comparison_matrix: np.ndarray) -> List[MarkovSequence]:
        """
        Findet längste zusammenhängende Subgraph-Sequenzen.
        """
        n = comparison_matrix.shape[0]
        sequences = []
        
        # Dynamische Programmierung
        dp = [1] * n
        parent = [-1] * n
        
        for i in range(1, n):
            for j in range(i):
                if comparison_matrix[j][i] == 1 and j != i:
                    if dp[j] + 1 > dp[i]:
                        dp[i] = dp[j] + 1
                        parent[i] = j
        
        # Rekonstruiere längste Sequenzen
        max_length = max(dp)
        
        for i in range(n):
            if dp[i] == max_length:
                sequence_indices = []
                current = i
                while current != -1:
                    sequence_indices.append(current)
                    current = parent[current]
                
                sequence_indices.reverse()
                
                # Prüfe ob monoton irreduzibel
                is_monotone_irreducible = all(
                    self.states[idx].is_irreducible 
                    for idx in sequence_indices
                )
                
                seq = MarkovSequence(
                    start_step=sequence_indices[0],
                    end_step=sequence_indices[-1],
                    length=len(sequence_indices),
                    states=[self.states[idx] for idx in sequence_indices],
                    is_monotone_irreducible=is_monotone_irreducible
                )
                sequences.append(seq)
        
        return sequences
    
    def _analyze_irreducibility_evolution(self) -> Dict:
        """Analysiert die Evolution der Irreduzibilität."""
        evolution = []
        first_irreducible = None
        
        for i, state in enumerate(self.states):
            evolution.append(state.is_irreducible)
            if state.is_irreducible and first_irreducible is None:
                first_irreducible = i
        
        return {
            "evolution": evolution,
            "first_irreducible_step": first_irreducible,
            "final_irreducible": evolution[-1] if evolution else False,
            "always_irreducible": all(evolution) if evolution else False
        }
    
    def _analyze_ergodicity_evolution(self) -> Dict:
        """Analysiert die Evolution der Ergodizität."""
        evolution = []
        first_ergodic = None
        
        for i, state in enumerate(self.states):
            evolution.append(state.is_ergodic)
            if state.is_ergodic and first_ergodic is None:
                first_ergodic = i
        
        return {
            "evolution": evolution,
            "first_ergodic_step": first_ergodic,
            "final_ergodic": evolution[-1] if evolution else False
        }
    
    def _analyze_stationary_stability(self) -> Dict:
        """Analysiert die Stabilität stationärer Verteilungen."""
        distributions = []
        changes = []
        
        for state in self.states:
            if state.stationary_distribution is not None:
                distributions.append(state.stationary_distribution)
        
        # Berechne L1-Distanzen zwischen aufeinanderfolgenden Verteilungen
        for i in range(1, len(distributions)):
            l1_distance = np.linalg.norm(
                distributions[i] - distributions[i-1], 
                ord=1
            )
            changes.append(l1_distance)
        
        return {
            "distributions": distributions,
            "l1_changes": changes,
            "max_change": max(changes) if changes else 0.0,
            "total_variation": sum(changes) if changes else 0.0
        }
    
    def print_analysis(self, analysis: Dict):
        """Gibt Analyseergebnisse formatiert aus."""
        print(f"\nGesamtanzahl Markov-Ketten: {analysis['total_states']}")
        
        print("\n" + "-"*70)
        print("LÄNGSTE SUBGRAPH-SEQUENZEN")
        print("-"*70)
        
        sequences = analysis['longest_sequences']
        if sequences:
            for idx, seq in enumerate(sequences, 1):
                print(f"\nSequenz {idx}: {seq}")
                print(f"  Schritte: {seq.start_step} → {seq.end_step}")
                print(f"  Länge: {seq.length}")
                print(f"  Monoton irreduzibel: {seq.is_monotone_irreducible}")
        else:
            print("Keine Sequenzen gefunden.")
        
        print("\n" + "-"*70)
        print("IRREDUZIBILITÄT-EVOLUTION")
        print("-"*70)
        
        irr = analysis['irreducibility_evolution']
        print(f"\nEvolution: {irr['evolution']}")
        print(f"Erste irreduzible Kette: Schritt {irr['first_irreducible_step']}")
        print(f"Finale Kette irreduzibel: {irr['final_irreducible']}")
        print(f"Immer irreduzibel: {irr['always_irreducible']}")
        
        print("\n" + "-"*70)
        print("ERGODIZITÄT-EVOLUTION")
        print("-"*70)
        
        erg = analysis['ergodicity_evolution']
        print(f"\nEvolution: {erg['evolution']}")
        print(f"Erste ergodische Kette: Schritt {erg['first_ergodic_step']}")
        print(f"Finale Kette ergodisch: {erg['final_ergodic']}")
        
        print("\n" + "-"*70)
        print("STATIONÄRE VERTEILUNGEN")
        print("-"*70)
        
        stat = analysis['stationary_stability']
        if stat['distributions']:
            print(f"\nAnzahl berechneter Verteilungen: {len(stat['distributions'])}")
            print(f"L1-Änderungen zwischen Schritten: {[f'{x:.4f}' for x in stat['l1_changes']]}")
            print(f"Maximale Änderung: {stat['max_change']:.4f}")
            print(f"Gesamtvariation: {stat['total_variation']:.4f}")
            
            print("\nStationäre Verteilungen:")
            for i, dist in enumerate(stat['distributions']):
                state_names = sorted(self.states[i].markov_chain._nodes.keys())
                print(f"  Schritt {i}: ", end="")
                for j, name in enumerate(state_names):
                    print(f"{name}={dist[j]:.4f} ", end="")
                print()
        else:
            print("Keine stationären Verteilungen (Ketten nicht ergodisch)")
        
        print("\n" + "-"*70)
        print("VERGLEICHSMATRIX")
        print("-"*70)
        print("\nZeile i, Spalte j = 1: MC_i ist Subgraph von MC_j")
        print(f"\n{analysis['comparison_matrix']}")


def create_weather_markov_example() -> List[MarkovChain]:
    """
    Erstellt das Wetter-Markov-Ketten Beispiel.
    
    Returns:
        Liste von Markov-Ketten, die die Evolution zeigen
    """
    states = ['sonnig', 'bewölkt', 'regnerisch']
    chains = []
    
    # MC_0: Nicht irreduzibel (kein Übergang sonnig→regnerisch)
    P0 = np.array([
        [0.7, 0.3, 0.0],
        [0.4, 0.4, 0.2],
        [0.0, 0.5, 0.5]
    ])
    mc0 = MarkovChain(states=states, transition_matrix=P0)
    chains.append(mc0)
    
    # MC_1: Immer noch nicht irreduzibel (kein Übergang regnerisch→sonnig)
    P1 = np.array([
        [0.65, 0.25, 0.1],
        [0.4, 0.4, 0.2],
        [0.0, 0.5, 0.5]
    ])
    mc1 = MarkovChain(states=states, transition_matrix=P1)
    chains.append(mc1)
    
    # MC_2: Irreduzibel und ergodisch (vollständig verbunden)
    P2 = np.array([
        [0.65, 0.25, 0.1],
        [0.4, 0.4, 0.2],
        [0.2, 0.4, 0.4]
    ])
    mc2 = MarkovChain(states=states, transition_matrix=P2)
    chains.append(mc2)
    
    return chains


if __name__ == "__main__":
    print("="*70)
    print("MARKOV-KETTEN STABILITÄTSANALYSE - WETTER BEISPIEL")
    print("="*70)
    
    # Erstelle Markov-Ketten Sequenz
    chains = create_weather_markov_example()
    
    # Initialisiere Analyse
    analysis = MarkovStabilityAnalysis()
    
    transformation_names = [
        "Initial (MC_0)",
        "Hinzufügen: sonnig→regnerisch (MC_1)",
        "Hinzufügen: regnerisch→sonnig (MC_2)"
    ]
    
    for i, (mc, name) in enumerate(zip(chains, transformation_names)):
        print(f"\nFüge hinzu: {name}")
        print(f"  {mc}")
        analysis.add_markov_chain(mc, transformation_name=name)
    
    # Führe Analyse durch
    print("\n" + "="*70)
    results = analysis.analyze()
    
    # Gebe Ergebnisse aus
    analysis.print_analysis(results)
    
    print("\n" + "="*70)
    print("ANALYSE ABGESCHLOSSEN")
    print("="*70)