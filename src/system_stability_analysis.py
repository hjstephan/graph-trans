"""
Systemstabilitätsanalyse durch Transformationsverlauf und Subgraph Algorithmus.

Dieses Modul analysiert die Stabilität und Ruhelagen eines Systems durch:
1. Speicherung aller Systemzustände während der Transformation
2. Anwendung des Subgraph Algorithmus auf die Zustandssequenz
3. Identifikation von längsten zusammenhängenden Subgraph-Sequenzen
"""

import time
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from .graph import Graph
from .transformation import Transformation
from src.subgraph import Subgraph


@dataclass
class SystemState:
    """Repräsentiert einen Systemzustand mit Metadaten."""
    step: int
    timestamp: float
    graph: Graph
    transformation_name: Optional[str] = None
    matrix: Optional[np.ndarray] = None
    node_mapping: Optional[Dict[str, int]] = None
    
    def __post_init__(self):
        """Berechnet Adjazenzmatrix beim Erstellen."""
        if self.matrix is None:
            self.matrix, self.node_mapping = self.graph.to_adjacency_matrix()


@dataclass
class SubgraphSequence:
    """Repräsentiert eine zusammenhängende Subgraph-Sequenz."""
    start_step: int
    end_step: int
    length: int
    states: List[SystemState] = field(default_factory=list)
    
    def __repr__(self):
        return f"SubgraphSequence(steps={self.start_step}-{self.end_step}, length={self.length})"


class SystemSimulation:
    """
    Simuliert ein System durch Graphtransformationen und speichert
    den vollständigen Verlauf für die Stabilitätsanalyse.
    """
    
    def __init__(self):
        """Initialisiert die Simulation."""
        self.states: List[SystemState] = []
        self.transformations: List[Transformation] = []
        self.subgraph_algo = Subgraph()
        
    def add_transformation(self, transformation: Transformation):
        """Fügt eine Transformation zur Simulation hinzu."""
        self.transformations.append(transformation)
    
    def run(self, initial_state: Graph, steps: int, delay: float = 0.0) -> List[SystemState]:
        """
        Führt die Simulation aus und speichert alle Zustände.
        
        Args:
            initial_state: Anfangszustand des Systems
            steps: Anzahl der Transformationsschritte
            delay: Verzögerung zwischen Schritten (für Visualisierung)
            
        Returns:
            Liste aller Systemzustände
        """
        self.states = []
        current_state = initial_state
        
        # Speichere Anfangszustand
        state = SystemState(
            step=0,
            timestamp=time.time(),
            graph=current_state.copy(),
            transformation_name="Initial"
        )
        self.states.append(state)
        
        # Führe Transformationen aus
        for step in range(1, steps + 1):
            if delay > 0:
                time.sleep(delay)
            
            # Wähle Transformation zyklisch
            transformation = self.transformations[(step - 1) % len(self.transformations)]
            
            try:
                # Wende Transformation an
                current_state = transformation.apply(current_state)
                
                # Speichere neuen Zustand
                state = SystemState(
                    step=step,
                    timestamp=time.time(),
                    graph=current_state.copy(),
                    transformation_name=transformation.name
                )
                self.states.append(state)
                
            except Exception as e:
                print(f"Fehler bei Schritt {step}: {e}")
                break
        
        return self.states
    
    def analyze_stability(self) -> Dict:
        """
        Analysiert die Stabilität des Systems durch Subgraph-Vergleiche.
        
        Returns:
            Dictionary mit Analyseergebnissen
        """
        if len(self.states) < 2:
            return {"error": "Nicht genug Zustände für Analyse"}
        
        print("\n" + "="*70)
        print("STABILITÄTSANALYSE")
        print("="*70 + "\n")
        
        # Baue Vergleichsmatrix auf
        n = len(self.states)
        comparison_matrix = np.zeros((n, n), dtype=int)
        
        print(f"Vergleiche {n} Zustände paarweise...")
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    comparison_matrix[i][j] = 1  # Zustand ist immer Subgraph von sich selbst
                else:
                    # Prüfe ob Zustand i ein Subgraph von Zustand j ist
                    decision, _ = self.subgraph_algo.compare_graphs(
                        self.states[i].matrix,
                        self.states[j].matrix
                    )
                    if decision in ["keep_B", "equal", "equal_keep_B"]:
                        comparison_matrix[i][j] = 1
        
        # Finde längste Subgraph-Sequenzen
        sequences = self._find_longest_sequences(comparison_matrix)
        
        # Finde stabile Zustände (Ruhelagen)
        stable_states = self._find_stable_states(comparison_matrix)
        
        # Finde Zyklen
        cycles = self._find_cycles()
        
        return {
            "total_states": n,
            "comparison_matrix": comparison_matrix,
            "longest_sequences": sequences,
            "stable_states": stable_states,
            "cycles": cycles,
            "states": self.states
        }
    
    def _find_longest_sequences(self, comparison_matrix: np.ndarray) -> List[SubgraphSequence]:
        """
        Findet die längsten zusammenhängenden Subgraph-Sequenzen.
        
        Eine Sequenz ist zusammenhängend, wenn jeder Zustand ein Subgraph
        des nächsten ist (monotone Erweiterung).
        """
        n = comparison_matrix.shape[0]
        sequences = []
        
        # Dynamische Programmierung: dp[i] = längste Sequenz endend bei i
        dp = [1] * n
        parent = [-1] * n
        
        for i in range(1, n):
            for j in range(i):
                # Wenn j -> i (j ist Subgraph von i)
                if comparison_matrix[j][i] == 1 and j != i:
                    if dp[j] + 1 > dp[i]:
                        dp[i] = dp[j] + 1
                        parent[i] = j
        
        # Rekonstruiere längste Sequenzen
        max_length = max(dp)
        
        for i in range(n):
            if dp[i] == max_length:
                # Rekonstruiere Sequenz rückwärts
                sequence_indices = []
                current = i
                while current != -1:
                    sequence_indices.append(current)
                    current = parent[current]
                
                sequence_indices.reverse()
                
                seq = SubgraphSequence(
                    start_step=sequence_indices[0],
                    end_step=sequence_indices[-1],
                    length=len(sequence_indices),
                    states=[self.states[idx] for idx in sequence_indices]
                )
                sequences.append(seq)
        
        return sequences
    
    def _find_stable_states(self, comparison_matrix: np.ndarray) -> List[int]:
        """
        Findet stabile Zustände (Ruhelagen).
        
        Ein Zustand ist stabil, wenn er identisch zu seinem Nachfolger ist
        oder wenn eine kleine Menge von Zuständen sich zyklisch wiederholt.
        """
        n = comparison_matrix.shape[0]
        stable = []
        
        for i in range(n - 1):
            # Prüfe ob i und i+1 gleich sind
            if (comparison_matrix[i][i+1] == 1 and 
                comparison_matrix[i+1][i] == 1):
                
                # Prüfe ob die Matrizen wirklich identisch sind
                if np.array_equal(self.states[i].matrix, self.states[i+1].matrix):
                    if i not in stable:
                        stable.append(i)
        
        return stable
    
    def _find_cycles(self) -> List[Tuple[int, int]]:
        """
        Findet Zyklen im Zustandsverlauf.
        
        Ein Zyklus liegt vor, wenn ein Zustand später wieder auftritt.
        """
        cycles = []
        n = len(self.states)
        
        for i in range(n):
            for j in range(i + 1, n):
                if np.array_equal(self.states[i].matrix, self.states[j].matrix):
                    cycles.append((i, j))
        
        return cycles
    
    def print_analysis(self, analysis: Dict):
        """Gibt die Analyseergebnisse formatiert aus."""
        print(f"\nGesamtanzahl Zustände: {analysis['total_states']}")
        
        print("\n" + "-"*70)
        print("LÄNGSTE SUBGRAPH-SEQUENZEN")
        print("-"*70)
        
        sequences = analysis['longest_sequences']
        if sequences:
            for idx, seq in enumerate(sequences, 1):
                print(f"\nSequenz {idx}: {seq}")
                print(f"  Schritte: {seq.start_step} → {seq.end_step}")
                print(f"  Länge: {seq.length}")
                print(f"  Transformationen:")
                for state in seq.states:
                    print(f"    Schritt {state.step}: {state.transformation_name}")
        else:
            print("Keine zusammenhängenden Sequenzen gefunden.")
        
        print("\n" + "-"*70)
        print("STABILE ZUSTÄNDE (RUHELAGEN)")
        print("-"*70)
        
        stable = analysis['stable_states']
        if stable:
            print(f"\nGefundene stabile Zustände bei Schritten: {stable}")
            for step in stable:
                state = self.states[step]
                print(f"  Schritt {step}: {state.transformation_name}")
        else:
            print("Keine stabilen Zustände gefunden.")
        
        print("\n" + "-"*70)
        print("ZYKLEN")
        print("-"*70)
        
        cycles = analysis['cycles']
        if cycles:
            print(f"\nGefundene Zyklen: {len(cycles)}")
            for start, end in cycles:
                cycle_length = end - start
                print(f"  Zyklus: Schritt {start} wiederholt sich bei Schritt {end} (Länge: {cycle_length})")
        else:
            print("Keine Zyklen gefunden.")
        
        print("\n" + "-"*70)
        print("VERGLEICHSMATRIX")
        print("-"*70)
        print("\nZeile i, Spalte j = 1 bedeutet: Zustand i ist Subgraph von Zustand j")
        print(f"\n{analysis['comparison_matrix']}")


def create_traffic_light_system() -> Tuple[Graph, List[Transformation]]:
    """
    Erstellt das Ampelsystem mit Transformationen.
    
    Returns:
        Tuple (Anfangszustand, Liste von Transformationen)
    """
    # Anfangszustand
    initial = Graph()
    initial.add_node('ampel', type='traffic_light')
    initial.add_node('nord', type='direction')
    initial.add_node('süd', type='direction')
    initial.add_node('grün', type='signal', active='true')
    initial.add_edge('ampel', 'grün', type='shows')
    initial.add_edge('grün', 'nord', type='allows')
    initial.add_edge('grün', 'süd', type='allows')
    
    # Transformationen
    transformations = []
    
    # 1. Grün → Gelb
    left = Graph()
    left.add_node('ampel', color='black', type='traffic_light')
    left.add_node('grün', color='red', type='signal')
    left.add_node('nord', color='black', type='direction')
    left.add_node('süd', color='black', type='direction')
    left.add_edge('ampel', 'grün', color='red', type='shows')
    left.add_edge('grün', 'nord', color='red', type='allows')
    left.add_edge('grün', 'süd', color='red', type='allows')
    
    right = Graph()
    right.add_node('ampel', color='black', type='traffic_light')
    right.add_node('gelb', color='green', type='signal')
    right.add_node('nord', color='black', type='direction')
    right.add_node('süd', color='black', type='direction')
    right.add_edge('ampel', 'gelb', color='green', type='shows')
    right.add_edge('gelb', 'nord', color='green', type='warns')
    right.add_edge('gelb', 'süd', color='green', type='warns')
    
    transformations.append(Transformation('Grün → Gelb', left, right))
    
    # 2. Gelb → Rot
    left = Graph()
    left.add_node('ampel', color='black', type='traffic_light')
    left.add_node('gelb', color='red', type='signal')
    left.add_node('nord', color='black', type='direction')
    left.add_node('süd', color='black', type='direction')
    left.add_edge('ampel', 'gelb', color='red', type='shows')
    left.add_edge('gelb', 'nord', color='red', type='warns')
    left.add_edge('gelb', 'süd', color='red', type='warns')
    
    right = Graph()
    right.add_node('ampel', color='black', type='traffic_light')
    right.add_node('rot', color='green', type='signal')
    right.add_node('nord', color='black', type='direction')
    right.add_node('süd', color='black', type='direction')
    right.add_edge('ampel', 'rot', color='green', type='shows')
    right.add_edge('rot', 'nord', color='green', type='blocks')
    right.add_edge('rot', 'süd', color='green', type='blocks')
    
    transformations.append(Transformation('Gelb → Rot', left, right))
    
    # 3. Rot → Rot-Gelb
    left = Graph()
    left.add_node('ampel', color='black', type='traffic_light')
    left.add_node('rot', color='red', type='signal')
    left.add_node('nord', color='black', type='direction')
    left.add_node('süd', color='black', type='direction')
    left.add_edge('ampel', 'rot', color='red', type='shows')
    left.add_edge('rot', 'nord', color='red', type='blocks')
    left.add_edge('rot', 'süd', color='red', type='blocks')
    
    right = Graph()
    right.add_node('ampel', color='black', type='traffic_light')
    right.add_node('rot_gelb', color='green', type='signal')
    right.add_node('nord', color='black', type='direction')
    right.add_node('süd', color='black', type='direction')
    right.add_edge('ampel', 'rot_gelb', color='green', type='shows')
    right.add_edge('rot_gelb', 'nord', color='green', type='prepares')
    right.add_edge('rot_gelb', 'süd', color='green', type='prepares')
    
    transformations.append(Transformation('Rot → Rot-Gelb', left, right))
    
    # 4. Rot-Gelb → Grün
    left = Graph()
    left.add_node('ampel', color='black', type='traffic_light')
    left.add_node('rot_gelb', color='red', type='signal')
    left.add_node('nord', color='black', type='direction')
    left.add_node('süd', color='black', type='direction')
    left.add_edge('ampel', 'rot_gelb', color='red', type='shows')
    left.add_edge('rot_gelb', 'nord', color='red', type='prepares')
    left.add_edge('rot_gelb', 'süd', color='red', type='prepares')
    
    right = Graph()
    right.add_node('ampel', color='black', type='traffic_light')
    right.add_node('grün', color='green', type='signal')
    right.add_node('nord', color='black', type='direction')
    right.add_node('süd', color='black', type='direction')
    right.add_edge('ampel', 'grün', color='green', type='shows')
    right.add_edge('grün', 'nord', color='green', type='allows')
    right.add_edge('grün', 'süd', color='green', type='allows')
    
    transformations.append(Transformation('Rot-Gelb → Grün', left, right))
    
    return initial, transformations


if __name__ == "__main__":
    print("="*70)
    print("SYSTEMSTABILITÄTSANALYSE - AMPELKREUZUNG")
    print("="*70)
    
    # Erstelle System
    initial_state, transformations = create_traffic_light_system()
    
    # Initialisiere Simulation
    sim = SystemSimulation()
    for t in transformations:
        sim.add_transformation(t)
    
    # Führe Simulation aus (z.B. 3 vollständige Zyklen = 12 Schritte)
    print("\nFühre Simulation aus...")
    states = sim.run(initial_state, steps=12, delay=0.1)
    
    print(f"✓ {len(states)} Zustände generiert\n")
    
    # Analysiere Stabilität
    analysis = sim.analyze_stability()
    
    # Gebe Ergebnisse aus
    sim.print_analysis(analysis)
    
    print("\n" + "="*70)
    print("ANALYSE ABGESCHLOSSEN")
    print("="*70)
