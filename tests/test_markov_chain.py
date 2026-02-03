"""
test_markov_chain.py - Tests für die MarkovChain-Klasse
"""

import pytest
import numpy as np
from src.markov_chain import MarkovChain


class TestMarkovChainCreation:
    """Tests für die Erstellung von Markov-Ketten"""
    
    def test_empty_markov_chain(self):
        """Test: Leere Markov-Kette erstellen"""
        mc = MarkovChain()
        assert len(mc) == 0
        assert mc.get_transition_matrix().shape == (0, 0)
    
    def test_markov_chain_with_states(self):
        """Test: Markov-Kette mit Zuständen erstellen"""
        states = ['A', 'B', 'C']
        mc = MarkovChain(states=states)
        
        assert len(mc) == 3
        assert 'A' in mc
        assert 'B' in mc
        assert 'C' in mc
    
    def test_markov_chain_with_transition_matrix(self):
        """Test: Markov-Kette mit Übergangsmatrix erstellen"""
        states = ['A', 'B']
        P = np.array([
            [0.7, 0.3],
            [0.4, 0.6]
        ])
        
        mc = MarkovChain(states=states, transition_matrix=P)
        
        assert len(mc) == 2
        assert np.allclose(mc.get_transition_matrix(), P)
    
    def test_invalid_transition_matrix_shape(self):
        """Test: Falsche Dimensionen der Übergangsmatrix"""
        states = ['A', 'B']
        P = np.array([[0.5, 0.5, 0.0]])  # 1x3 statt 2x2
        
        with pytest.raises(ValueError, match="must be 2×2"):
            MarkovChain(states=states, transition_matrix=P)
    
    def test_invalid_transition_matrix_not_stochastic(self):
        """Test: Zeilen summieren nicht zu 1"""
        states = ['A', 'B']
        P = np.array([
            [0.5, 0.3],  # Summe = 0.8 != 1
            [0.4, 0.6]
        ])
        
        with pytest.raises(ValueError, match="must sum to 1"):
            MarkovChain(states=states, transition_matrix=P)
    
    def test_invalid_transition_matrix_negative(self):
        """Test: Negative Wahrscheinlichkeiten"""
        states = ['A', 'B']
        P = np.array([
            [1.2, -0.2],  # Negative Wahrscheinlichkeit
            [0.4, 0.6]
        ])
        
        with pytest.raises(ValueError, match="non-negative"):
            MarkovChain(states=states, transition_matrix=P)


class TestTransitionMatrix:
    """Tests für Übergangsmatrix-Operationen"""
    
    def test_set_and_get_transition_matrix(self):
        """Test: Matrix setzen und abrufen"""
        mc = MarkovChain(states=['A', 'B'])
        P = np.array([
            [0.5, 0.5],
            [0.3, 0.7]
        ])
        
        mc.set_transition_matrix(P)
        P_retrieved = mc.get_transition_matrix()
        
        assert np.allclose(P_retrieved, P)
    
    def test_transition_matrix_creates_edges(self):
        """Test: Übergangsmatrix erstellt Kanten"""
        mc = MarkovChain(states=['A', 'B'])
        P = np.array([
            [0.5, 0.5],
            [0.0, 1.0]  # Kein Übergang B→A
        ])
        
        mc.set_transition_matrix(P)
        edges = mc.get_edges()
        
        # Sollte 3 Kanten haben (A→A, A→B, B→B)
        assert len(edges) == 3
        
        edge_pairs = {(e.from_node, e.to_node) for e in edges}
        assert ('A', 'A') in edge_pairs
        assert ('A', 'B') in edge_pairs
        assert ('B', 'B') in edge_pairs
        assert ('B', 'A') not in edge_pairs  # Kein Übergang
    
    def test_edge_probabilities(self):
        """Test: Kanten haben korrekte Wahrscheinlichkeiten"""
        mc = MarkovChain(states=['A', 'B'])
        P = np.array([
            [0.3, 0.7],
            [0.6, 0.4]
        ])
        
        mc.set_transition_matrix(P)
        
        for edge in mc.get_edges():
            if edge.from_node == 'A' and edge.to_node == 'A':
                assert edge.attributes['probability'] == 0.3
            elif edge.from_node == 'A' and edge.to_node == 'B':
                assert edge.attributes['probability'] == 0.7
            elif edge.from_node == 'B' and edge.to_node == 'A':
                assert edge.attributes['probability'] == 0.6
            elif edge.from_node == 'B' and edge.to_node == 'B':
                assert edge.attributes['probability'] == 0.4
    
    def test_transition_matrix_reconstruction(self):
        """Test: Matrix wird aus Kanten rekonstruiert"""
        mc = MarkovChain(states=['A', 'B'])
        mc._transition_matrix = None  # Setze auf None
        
        # Füge Kanten manuell hinzu
        mc.add_edge('A', 'A', probability=0.4)
        mc.add_edge('A', 'B', probability=0.6)
        mc.add_edge('B', 'A', probability=0.3)
        mc.add_edge('B', 'B', probability=0.7)
        
        P = mc.get_transition_matrix()
        
        expected = np.array([
            [0.4, 0.6],
            [0.3, 0.7]
        ])
        
        assert np.allclose(P, expected)


class TestIrreducibility:
    """Tests für Irreduzibilität"""
    
    def test_irreducible_chain(self):
        """Test: Irreduzible Kette"""
        mc = MarkovChain(states=['A', 'B', 'C'])
        P = np.array([
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0]
        ])
        mc.set_transition_matrix(P)
        
        assert mc.is_irreducible()
    
    def test_reducible_chain(self):
        """Test: Reduzible Kette mit absorbierendem Zustand"""
        mc = MarkovChain(states=['A', 'B', 'C'])
        # C ist absorbierend - kann von A,B erreicht werden, aber C kann nicht zurück
        P = np.array([
            [0.5, 0.5, 0.0],  # A kann zu A, B
            [0.0, 0.7, 0.3],  # B kann zu B, C (nicht zu A!)
            [0.0, 0.0, 1.0]   # C ist absorbierend (nur C→C)
        ])
        mc.set_transition_matrix(P)
        
        # Diese Kette ist NICHT irreduzibel weil:
        # - C ist von A,B erreichbar (A→B→C)
        # - Aber A ist NICHT von C erreichbar (kein Pfad C→A)
        assert not mc.is_irreducible()

    def test_empty_chain_is_irreducible(self):
        """Test: Leere Kette gilt als irreduzibel"""
        mc = MarkovChain()
        assert mc.is_irreducible()
    
    def test_single_state_chain(self):
        """Test: Kette mit einem Zustand"""
        mc = MarkovChain(states=['A'])
        P = np.array([[1.0]])
        mc.set_transition_matrix(P)
        
        assert mc.is_irreducible()


class TestPeriodicity:
    """Tests für Periodizität"""
    
    def test_aperiodic_chain_with_self_loop(self):
        """Test: Aperiodische Kette mit Selbstschleife"""
        mc = MarkovChain(states=['A', 'B'])
        P = np.array([
            [0.5, 0.5],  # Selbstschleife A→A
            [0.5, 0.5]
        ])
        mc.set_transition_matrix(P)
        
        assert mc.get_period('A') == 1
        assert mc.get_period('B') == 1
        assert mc.is_aperiodic()
    
    def test_periodic_chain(self):
        """Test: Periodische Kette"""
        mc = MarkovChain(states=['A', 'B'])
        P = np.array([
            [0.0, 1.0],  # Nur A→B
            [1.0, 0.0]   # Nur B→A
        ])
        mc.set_transition_matrix(P)
        
        assert mc.get_period('A') == 2
        assert mc.get_period('B') == 2
        assert not mc.is_aperiodic()
    
    def test_period_of_nonexistent_state(self):
        """Test: Periode für nicht-existierenden Zustand"""
        mc = MarkovChain(states=['A'])
        P = np.array([[1.0]])
        mc.set_transition_matrix(P)
        
        with pytest.raises(ValueError, match="not found"):
            mc.get_period('Z')
    
    def test_period_default_state(self):
        """Test: Periode ohne Zustandsangabe (nimmt ersten Zustand)"""
        mc = MarkovChain(states=['B', 'A'])  # B kommt zuerst (sortiert: A, B)
        P = np.array([
            [0.0, 1.0],
            [1.0, 0.0]
        ])
        mc.set_transition_matrix(P)
        
        # Sollte Periode des ersten sortierten Zustands ('A') zurückgeben
        period = mc.get_period()
        assert period == 2
    
    def test_empty_chain_period(self):
        """Test: Periode einer leeren Kette"""
        mc = MarkovChain()
        assert mc.get_period() == 1


class TestErgodicity:
    """Tests für Ergodizität"""
    
    def test_ergodic_chain(self):
        """Test: Ergodische Kette (irreduzibel + aperiodisch)"""
        mc = MarkovChain(states=['A', 'B', 'C'])
        P = np.array([
            [0.5, 0.3, 0.2],
            [0.2, 0.5, 0.3],
            [0.3, 0.2, 0.5]
        ])
        mc.set_transition_matrix(P)
        
        assert mc.is_irreducible()
        assert mc.is_aperiodic()
        assert mc.is_ergodic()
    
    def test_not_ergodic_reducible(self):
        """Test: Nicht ergodisch wegen Reduzibilität"""
        mc = MarkovChain(states=['A', 'B'])
        P = np.array([
            [0.5, 0.5],
            [0.0, 1.0]  # B ist absorbierend, nicht irreduzibel
        ])
        mc.set_transition_matrix(P)
        
        assert not mc.is_irreducible()
        assert not mc.is_ergodic()
    
    def test_not_ergodic_periodic(self):
        """Test: Nicht ergodisch wegen Periodizität"""
        mc = MarkovChain(states=['A', 'B'])
        P = np.array([
            [0.0, 1.0],
            [1.0, 0.0]
        ])
        mc.set_transition_matrix(P)
        
        assert mc.is_irreducible()
        assert not mc.is_aperiodic()
        assert not mc.is_ergodic()


class TestStationaryDistribution:
    """Tests für stationäre Verteilungen"""
    
    def test_stationary_distribution_ergodic_chain(self):
        """Test: Stationäre Verteilung für ergodische Kette"""
        mc = MarkovChain(states=['A', 'B'])
        P = np.array([
            [0.7, 0.3],
            [0.4, 0.6]
        ])
        mc.set_transition_matrix(P)
        
        pi = mc.compute_stationary_distribution()
        
        assert pi is not None
        assert len(pi) == 2
        assert np.isclose(pi.sum(), 1.0)  # Summe = 1
        assert np.allclose(pi @ P, pi)  # π * P = π
    
    def test_stationary_distribution_not_ergodic(self):
        """Test: Keine stationäre Verteilung für nicht-ergodische Kette"""
        mc = MarkovChain(states=['A', 'B'])
        P = np.array([
            [0.0, 1.0],
            [1.0, 0.0]
        ])
        mc.set_transition_matrix(P)
        
        pi = mc.compute_stationary_distribution()
        
        assert pi is None
    
    def test_stationary_distribution_empty_chain(self):
        """Test: Leere Kette"""
        mc = MarkovChain()
        pi = mc.compute_stationary_distribution()
        
        assert pi is not None
        assert len(pi) == 0
    
    def test_stationary_distribution_absorbing_state(self):
        """Test: Kette mit absorbierendem Zustand"""
        mc = MarkovChain(states=['A', 'B', 'C'])
        P = np.array([
            [0.0, 0.5, 0.5],
            [0.0, 0.5, 0.5],
            [0.0, 0.0, 1.0]  # C ist absorbierend
        ])
        mc.set_transition_matrix(P)
        
        # Nicht ergodisch
        assert not mc.is_ergodic()
        pi = mc.compute_stationary_distribution()
        assert pi is None


class TestTransitionModification:
    """Tests für Hinzufügen/Entfernen von Übergängen"""
    
    def test_add_transition(self):
        """Test: Übergang hinzufügen"""
        mc = MarkovChain(states=['A', 'B', 'C'])
        P = np.array([
            [0.7, 0.3, 0.0],  # Kein A→C
            [0.2, 0.5, 0.3],
            [0.1, 0.4, 0.5]
        ])
        mc.set_transition_matrix(P)
        
        # Füge A→C hinzu mit Wahrscheinlichkeit 0.1
        mc.add_transition('A', 'C', 0.1)
        
        P_new = mc.get_transition_matrix()
        
        # Zeile A sollte renormalisiert sein
        assert np.isclose(P_new.sum(axis=1)[0], 1.0)
        assert P_new[0, 2] > 0  # A→C existiert jetzt
    
    def test_remove_transition(self):
        """Test: Übergang entfernen"""
        mc = MarkovChain(states=['A', 'B'])
        P = np.array([
            [0.3, 0.7],
            [0.4, 0.6]
        ])
        mc.set_transition_matrix(P)
        
        mc.remove_transition('A', 'B')
        
        P_new = mc.get_transition_matrix()
        
        # A→B sollte 0 sein, A→A sollte 1 sein (renormalisiert)
        assert np.isclose(P_new[0, 1], 0.0)
        assert np.isclose(P_new[0, 0], 1.0)
    
    def test_add_transition_invalid_probability(self):
        """Test: Ungültige Wahrscheinlichkeit"""
        mc = MarkovChain(states=['A', 'B'])
        P = np.array([[0.5, 0.5], [0.5, 0.5]])
        mc.set_transition_matrix(P)
        
        with pytest.raises(ValueError, match="in \\[0, 1\\]"):
            mc.add_transition('A', 'B', 1.5)
    
    def test_add_transition_nonexistent_state(self):
        """Test: Übergang zu nicht-existierendem Zustand"""
        mc = MarkovChain(states=['A', 'B'])
        P = np.array([[0.5, 0.5], [0.5, 0.5]])
        mc.set_transition_matrix(P)
        
        with pytest.raises(ValueError, match="must exist"):
            mc.add_transition('A', 'Z', 0.5)


class TestMarkovChainCopy:
    """Tests für Kopieren von Markov-Ketten"""
    
    def test_copy_markov_chain(self):
        """Test: Markov-Kette kopieren"""
        mc = MarkovChain(states=['A', 'B'])
        P = np.array([
            [0.6, 0.4],
            [0.3, 0.7]
        ])
        mc.set_transition_matrix(P)
        
        mc_copy = mc.copy()
        
        assert len(mc_copy) == len(mc)
        assert np.allclose(mc_copy.get_transition_matrix(), 
                          mc.get_transition_matrix())
    
    def test_copy_is_independent(self):
        """Test: Kopie ist unabhängig vom Original"""
        mc = MarkovChain(states=['A', 'B'])
        P = np.array([[0.5, 0.5], [0.5, 0.5]])
        mc.set_transition_matrix(P)
        
        mc_copy = mc.copy()
        
        # Ändere Original
        mc.add_transition('A', 'A', 0.8)
        
        # Kopie sollte unverändert sein
        P_copy = mc_copy.get_transition_matrix()
        assert np.isclose(P_copy[0, 0], 0.5)


class TestMarkovChainRepr:
    """Tests für String-Repräsentation"""
    
    def test_repr_ergodic(self):
        """Test: Repräsentation ergodischer Kette"""
        mc = MarkovChain(states=['A', 'B'])
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        mc.set_transition_matrix(P)
        
        repr_str = repr(mc)
        
        assert 'MarkovChain' in repr_str
        assert 'states=2' in repr_str
        assert 'irreducible' in repr_str
        assert 'aperiodic' in repr_str
        assert 'ergodic' in repr_str
    
    def test_repr_not_ergodic(self):
        """Test: Repräsentation nicht-ergodischer Kette"""
        mc = MarkovChain(states=['A', 'B'])
        P = np.array([[0.0, 1.0], [1.0, 0.0]])
        mc.set_transition_matrix(P)
        
        repr_str = repr(mc)
        
        assert 'periodic' in repr_str
        assert 'not ergodic' in repr_str


class TestAdjacencyMatrixIntegration:
    """Tests für Integration mit Adjazenzmatrix (für Subgraph-Algorithmus)"""
    
    def test_to_adjacency_matrix(self):
        """Test: Konvertierung zu Adjazenzmatrix"""
        mc = MarkovChain(states=['A', 'B', 'C'])
        P = np.array([
            [0.5, 0.5, 0.0],
            [0.0, 0.7, 0.3],
            [0.2, 0.3, 0.5]
        ])
        mc.set_transition_matrix(P)
        
        adj_matrix, node_mapping = mc.to_adjacency_matrix()
        
        # Adjazenzmatrix sollte binär sein (nur 0 und 1)
        assert np.all((adj_matrix == 0) | (adj_matrix == 1))
        
        # Prüfe dass Struktur erhalten ist
        for i in range(3):
            for j in range(3):
                if P[i, j] > 0:
                    assert adj_matrix[i, j] == 1
                else:
                    assert adj_matrix[i, j] == 0
    
    def test_adjacency_preserves_transitions(self):
        """Test: Adjazenzmatrix bewahrt Übergänge"""
        mc = MarkovChain(states=['A', 'B'])
        P = np.array([
            [0.8, 0.2],
            [0.0, 1.0]  # Kein B→A Übergang
        ])
        mc.set_transition_matrix(P)
        
        adj_matrix, _ = mc.to_adjacency_matrix()
        
        # A→A: 1, A→B: 1, B→A: 0, B→B: 1
        expected = np.array([
            [1, 1],
            [0, 1]
        ])
        
        assert np.array_equal(adj_matrix, expected)