"""
markov_chain.py - Markov-Ketten Implementierung basierend auf Graph-Klasse
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
from graph import Graph, Node, Edge


class MarkovChain(Graph):
    """
    Erweitert Graph-Klasse um Markov-Ketten Funktionalität.
    
    Eine Markov-Kette ist ein gewichteter gerichteter Graph, wobei:
    - Knoten die Zustände repräsentieren
    - Kanten die Übergänge darstellen
    - Kantengewichte die Übergangswahrscheinlichkeiten sind
    """
    
    def __init__(self, states: Optional[List[str]] = None,
                 transition_matrix: Optional[np.ndarray] = None):
        """
        Initialisiert eine Markov-Kette.
        
        Args:
            states: Liste der Zustandsnamen
            transition_matrix: Übergangsmatrix P (optional)
        """
        super().__init__()
        self._transition_matrix: Optional[np.ndarray] = None
        
        if states is not None:
            for state in states:
                self.add_node(id=state, color='black')
            
            if transition_matrix is not None:
                self.set_transition_matrix(transition_matrix)
    
    def set_transition_matrix(self, transition_matrix: np.ndarray):
        """
        Setzt die Übergangsmatrix und erstellt entsprechende Kanten.
        
        Args:
            transition_matrix: n×n Matrix mit Übergangswahrscheinlichkeiten
        """
        n = len(self._nodes)
        if transition_matrix.shape != (n, n):
            raise ValueError(f"Transition matrix must be {n}×{n}")
        
        # Validiere dass es eine stochastische Matrix ist
        if not np.allclose(transition_matrix.sum(axis=1), 1.0):
            raise ValueError("Rows of transition matrix must sum to 1")
        
        if not np.all(transition_matrix >= 0):
            raise ValueError("All transition probabilities must be non-negative")
        
        self._transition_matrix = transition_matrix.copy()
        
        # Entferne alle bestehenden Kanten
        self._edges.clear()
        
        # Erstelle Kanten für alle Übergänge mit p > 0
        node_list = sorted(self._nodes.keys())
        for i, from_state in enumerate(node_list):
            for j, to_state in enumerate(node_list):
                prob = transition_matrix[i, j]
                if prob > 0:
                    # Speichere Wahrscheinlichkeit als Attribut
                    self.add_edge(from_state, to_state, 
                                probability=float(prob))
    
    def get_transition_matrix(self) -> np.ndarray:
        """
        Gibt die Übergangsmatrix zurück.
        
        Returns:
            n×n Übergangsmatrix
        """
        if self._transition_matrix is None:
            # Rekonstruiere aus Kanten
            n = len(self._nodes)
            matrix = np.zeros((n, n))
            node_list = sorted(self._nodes.keys())
            node_to_idx = {node: i for i, node in enumerate(node_list)}
            
            for edge in self._edges:
                i = node_to_idx[edge.from_node]
                j = node_to_idx[edge.to_node]
                prob = edge.attributes.get('probability', 0.0)
                matrix[i, j] = prob
            
            self._transition_matrix = matrix
        
        return self._transition_matrix.copy()
    
    def is_irreducible(self) -> bool:
        """
        Prüft ob die Markov-Kette irreduzibel ist.
        
        Returns:
            True wenn alle Zustände voneinander erreichbar sind
        """
        matrix, _ = self.to_adjacency_matrix()
        n = matrix.shape[0]
        
        if n == 0:
            return True
        
        if n == 1:
            return True
        
        # Floyd-Warshall Algorithmus für Erreichbarkeit
        # reachability[i][j] = 1 wenn es einen Pfad von i nach j gibt
        reachability = matrix.copy()
        
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if reachability[i][k] and reachability[k][j]:
                        reachability[i][j] = 1
        
        # Alle Einträge müssen 1 sein (jeder Zustand von jedem erreichbar)
        return bool(np.all(reachability == 1))
    
    def get_period(self, state: Optional[str] = None) -> int:
        """
        Berechnet die Periode eines Zustands.
        
        Args:
            state: Zustand (wenn None, wird der erste Zustand verwendet)
        
        Returns:
            Periode des Zustands
        """
        if len(self._nodes) == 0:
            return 1
        
        if state is None:
            state = sorted(self._nodes.keys())[0]
        
        if state not in self._nodes:
            raise ValueError(f"State '{state}' not found")
        
        matrix, node_mapping = self.to_adjacency_matrix()
        idx = node_mapping[state]
        n = matrix.shape[0]
        
        # Finde alle n für die P^n[idx, idx] > 0
        powers_with_return = []
        current = matrix.copy()
        
        for power in range(1, n * n + 1):
            if current[idx, idx] > 0:
                powers_with_return.append(power)
            current = current @ matrix
            
            # Abbruch wenn wir genug Werte haben
            if len(powers_with_return) >= n:
                break
        
        if not powers_with_return:
            return 0  # Zustand ist transient
        
        # Berechne GCD aller Potenzen
        from math import gcd
        result = powers_with_return[0]
        for p in powers_with_return[1:]:
            result = gcd(result, p)
        
        return result
    
    def is_aperiodic(self) -> bool:
        """
        Prüft ob die Markov-Kette aperiodisch ist.
        
        Returns:
            True wenn alle Zustände Periode 1 haben
        """
        if len(self._nodes) == 0:
            return True
        
        for state in self._nodes.keys():
            if self.get_period(state) != 1:
                return False
        
        return True
    
    def is_ergodic(self) -> bool:
        """
        Prüft ob die Markov-Kette ergodisch ist.
        
        Returns:
            True wenn irreduzibel und aperiodisch
        """
        return self.is_irreducible() and self.is_aperiodic()
    
    def compute_stationary_distribution(self, tolerance: float = 1e-10,
                                       max_iterations: int = 10000) -> Optional[np.ndarray]:
        """
        Berechnet die stationäre Verteilung.
        
        Args:
            tolerance: Konvergenzkriterium
            max_iterations: Maximale Anzahl Iterationen
        
        Returns:
            Stationäre Verteilung π oder None wenn nicht ergodisch
        """
        if not self.is_ergodic():
            return None
        
        n = len(self._nodes)
        if n == 0:
            return np.array([])
        
        P = self.get_transition_matrix()
        
        # Eigenvalue method (more reliable than power iteration)
        eigenvalues, eigenvectors = np.linalg.eig(P.T)
        
        # Find eigenvector for eigenvalue 1
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        pi = np.real(eigenvectors[:, idx])
        
        # Normalize (ensure all values are positive)
        pi = np.abs(pi)
        pi = pi / pi.sum()
        
        return pi
    
    def add_transition(self, from_state: str, to_state: str, 
                      probability: float):
        """
        Fügt einen Übergang hinzu und renormalisiert die Zeile.
        
        Args:
            from_state: Ausgangszustand
            to_state: Zielzustand
            probability: Gewünschte neue Wahrscheinlichkeit
        """
        if from_state not in self._nodes or to_state not in self._nodes:
            raise ValueError("Both states must exist")
        
        if probability < 0 or probability > 1:
            raise ValueError("Probability must be in [0, 1]")
        
        # Hole aktuelle Matrix
        P = self.get_transition_matrix()
        node_list = sorted(self._nodes.keys())
        node_to_idx = {node: i for i, node in enumerate(node_list)}
        
        i = node_to_idx[from_state]
        j = node_to_idx[to_state]
        
        # Setze neue Wahrscheinlichkeit
        old_prob = P[i, j]
        P[i, j] = probability
        
        # Renormalisiere Zeile
        row_sum = P[i, :].sum()
        if row_sum > 0:
            P[i, :] = P[i, :] / row_sum
        
        # Aktualisiere Matrix
        self.set_transition_matrix(P)
    
    def remove_transition(self, from_state: str, to_state: str):
        """
        Entfernt einen Übergang und renormalisiert die Zeile.
        
        Args:
            from_state: Ausgangszustand
            to_state: Zielzustand
        """
        P = self.get_transition_matrix()
        node_list = sorted(self._nodes.keys())
        node_to_idx = {node: i for i, node in enumerate(node_list)}
        
        i = node_to_idx[from_state]
        j = node_to_idx[to_state]
        
        # Setze Wahrscheinlichkeit auf 0
        P[i, j] = 0
        
        # Renormalisiere Zeile
        row_sum = P[i, :].sum()
        if row_sum > 0:
            P[i, :] = P[i, :] / row_sum
        
        # Aktualisiere Matrix
        self.set_transition_matrix(P)
    
    def copy(self) -> 'MarkovChain':
        """Erstellt eine tiefe Kopie der Markov-Kette."""
        mc = MarkovChain()
        
        # Kopiere Knoten
        for node in self._nodes.values():
            mc.add_node(
                id=node.id,
                color=node.color,
                **node.attributes.copy()
            )
        
        # Kopiere Übergangsmatrix
        if self._transition_matrix is not None:
            mc.set_transition_matrix(self._transition_matrix)
        
        return mc
    
    def __repr__(self) -> str:
        irreducible = "irreducible" if self.is_irreducible() else "reducible"
        aperiodic = "aperiodic" if self.is_aperiodic() else "periodic"
        ergodic = "ergodic" if self.is_ergodic() else "not ergodic"
        
        return (f"MarkovChain(states={len(self._nodes)}, "
                f"{irreducible}, {aperiodic}, {ergodic})")