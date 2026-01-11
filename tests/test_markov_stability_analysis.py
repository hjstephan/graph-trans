"""
test_markov_stability_analysis.py - Tests für Markov-Ketten Stabilitätsanalyse
"""

import pytest
import numpy as np
from markov_chain import MarkovChain
from markov_stability_analysis import (
    MarkovState, MarkovSequence, MarkovStabilityAnalysis,
    create_weather_markov_example
)


class TestMarkovState:
    """Tests für MarkovState Dataclass"""
    
    def test_markov_state_creation(self):
        """Test: MarkovState erstellen"""
        mc = MarkovChain(states=['A', 'B'])
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        mc.set_transition_matrix(P)
        
        state = MarkovState(
            step=0,
            timestamp=123.456,
            markov_chain=mc,
            transformation_name="Test"
        )
        
        assert state.step == 0
        assert state.timestamp == 123.456
        assert state.transformation_name == "Test"
        assert state.markov_chain == mc
    
    def test_markov_state_post_init(self):
        """Test: Automatische Berechnung der Metriken"""
        mc = MarkovChain(states=['A', 'B'])
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        mc.set_transition_matrix(P)
        
        state = MarkovState(
            step=0,
            timestamp=0.0,
            markov_chain=mc
        )
        
        # Sollte automatisch berechnet werden
        assert state.transition_matrix is not None
        assert state.adjacency_matrix is not None
        assert isinstance(state.is_irreducible, bool)
        assert isinstance(state.is_aperiodic, bool)
        assert isinstance(state.is_ergodic, bool)
    
    def test_markov_state_ergodic_computes_stationary(self):
        """Test: Ergodische Kette berechnet stationäre Verteilung"""
        mc = MarkovChain(states=['A', 'B'])
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        mc.set_transition_matrix(P)
        
        state = MarkovState(step=0, timestamp=0.0, markov_chain=mc)
        
        assert state.is_ergodic
        assert state.stationary_distribution is not None
        assert len(state.stationary_distribution) == 2
        assert np.isclose(state.stationary_distribution.sum(), 1.0)
    
    def test_markov_state_not_ergodic_no_stationary(self):
        """Test: Nicht-ergodische Kette hat keine stationäre Verteilung"""
        mc = MarkovChain(states=['A', 'B'])
        P = np.array([[0.0, 1.0], [1.0, 0.0]])  # Periodisch
        mc.set_transition_matrix(P)
        
        state = MarkovState(step=0, timestamp=0.0, markov_chain=mc)
        
        assert not state.is_ergodic
        assert state.stationary_distribution is None


class TestMarkovSequence:
    """Tests für MarkovSequence Dataclass"""
    
    def test_markov_sequence_creation(self):
        """Test: MarkovSequence erstellen"""
        seq = MarkovSequence(
            start_step=0,
            end_step=5,
            length=6,
            is_monotone_irreducible=True
        )
        
        assert seq.start_step == 0
        assert seq.end_step == 5
        assert seq.length == 6
        assert seq.is_monotone_irreducible
    
    def test_markov_sequence_repr(self):
        """Test: String-Repräsentation"""
        seq = MarkovSequence(
            start_step=1,
            end_step=3,
            length=3,
            is_monotone_irreducible=False
        )
        
        repr_str = repr(seq)
        assert 'MarkovSequence' in repr_str
        assert 'steps=1-3' in repr_str
        assert 'length=3' in repr_str
        assert 'monotone_irreducible=False' in repr_str


class TestMarkovStabilityAnalysis:
    """Tests für MarkovStabilityAnalysis Klasse"""
    
    def test_analysis_creation(self):
        """Test: Analyse erstellen"""
        analysis = MarkovStabilityAnalysis()
        
        assert len(analysis.states) == 0
        assert analysis.subgraph_algo is not None
    
    def test_add_markov_chain(self):
        """Test: Markov-Kette hinzufügen"""
        analysis = MarkovStabilityAnalysis()
        
        mc = MarkovChain(states=['A', 'B'])
        P = np.array([[0.5, 0.5], [0.5, 0.5]])
        mc.set_transition_matrix(P)
        
        analysis.add_markov_chain(mc, transformation_name="Test")
        
        assert len(analysis.states) == 1
        assert analysis.states[0].transformation_name == "Test"
        assert analysis.states[0].step == 0
    
    def test_add_multiple_markov_chains(self):
        """Test: Mehrere Markov-Ketten hinzufügen"""
        analysis = MarkovStabilityAnalysis()
        
        for i in range(3):
            mc = MarkovChain(states=['A', 'B'])
            P = np.array([[0.5, 0.5], [0.5, 0.5]])
            mc.set_transition_matrix(P)
            analysis.add_markov_chain(mc, transformation_name=f"Step{i}")
        
        assert len(analysis.states) == 3
        assert analysis.states[0].step == 0
        assert analysis.states[1].step == 1
        assert analysis.states[2].step == 2
    
    def test_analyze_insufficient_states(self):
        """Test: Analyse mit zu wenigen Zuständen"""
        analysis = MarkovStabilityAnalysis()
        
        mc = MarkovChain(states=['A'])
        P = np.array([[1.0]])
        mc.set_transition_matrix(P)
        analysis.add_markov_chain(mc)
        
        result = analysis.analyze()
        
        assert "error" in result
        assert "Nicht genug" in result["error"]
    
    def test_analyze_simple_sequence(self):
        """Test: Einfache Analyse-Sequenz"""
        analysis = MarkovStabilityAnalysis()
        
        # Zwei identische Ketten
        for i in range(2):
            mc = MarkovChain(states=['A', 'B'])
            P = np.array([[0.7, 0.3], [0.4, 0.6]])
            mc.set_transition_matrix(P)
            analysis.add_markov_chain(mc, transformation_name=f"MC{i}")
        
        result = analysis.analyze()
        
        assert "total_states" in result
        assert result["total_states"] == 2
        assert "comparison_matrix" in result
        assert "longest_sequences" in result


class TestIrreducibilityEvolution:
    """Tests für Irreduzibilitäts-Evolution"""
    
    def test_irreducibility_evolution_all_irreducible(self):
        """Test: Alle Ketten sind irreduzibel"""
        analysis = MarkovStabilityAnalysis()
        
        for i in range(3):
            mc = MarkovChain(states=['A', 'B'])
            P = np.array([[0.5, 0.5], [0.5, 0.5]])
            mc.set_transition_matrix(P)
            analysis.add_markov_chain(mc)
        
        result = analysis.analyze()
        irr = result['irreducibility_evolution']
        
        assert irr['evolution'] == [True, True, True]
        assert irr['first_irreducible_step'] == 0
        assert irr['final_irreducible'] == True
        assert irr['always_irreducible'] == True
    
    def test_irreducibility_evolution_becomes_irreducible(self):
        """Test: Kette wird irreduzibel"""
        analysis = MarkovStabilityAnalysis()
        
        # MC0: Nicht irreduzibel
        mc0 = MarkovChain(states=['A', 'B'])
        P0 = np.array([[1.0, 0.0], [0.0, 1.0]])  # Zwei getrennte Komponenten
        mc0.set_transition_matrix(P0)
        analysis.add_markov_chain(mc0)
        
        # MC1: Irreduzibel
        mc1 = MarkovChain(states=['A', 'B'])
        P1 = np.array([[0.5, 0.5], [0.5, 0.5]])
        mc1.set_transition_matrix(P1)
        analysis.add_markov_chain(mc1)
        
        result = analysis.analyze()
        irr = result['irreducibility_evolution']
        
        assert irr['evolution'] == [False, True]
        assert irr['first_irreducible_step'] == 1
        assert irr['final_irreducible'] == True
        assert irr['always_irreducible'] == False


class TestErgodicityEvolution:
    """Tests für Ergodizitäts-Evolution"""
    
    def test_ergodicity_evolution_becomes_ergodic(self):
        """Test: Kette wird ergodisch"""
        analysis = MarkovStabilityAnalysis()
        
        # MC0: Nicht ergodisch (periodisch)
        mc0 = MarkovChain(states=['A', 'B'])
        P0 = np.array([[0.0, 1.0], [1.0, 0.0]])
        mc0.set_transition_matrix(P0)
        analysis.add_markov_chain(mc0)
        
        # MC1: Ergodisch
        mc1 = MarkovChain(states=['A', 'B'])
        P1 = np.array([[0.7, 0.3], [0.4, 0.6]])
        mc1.set_transition_matrix(P1)
        analysis.add_markov_chain(mc1)
        
        result = analysis.analyze()
        erg = result['ergodicity_evolution']
        
        assert erg['evolution'] == [False, True]
        assert erg['first_ergodic_step'] == 1
        assert erg['final_ergodic'] == True


class TestStationaryStability:
    """Tests für Stabilität stationärer Verteilungen"""
    
    def test_stationary_stability_identical_chains(self):
        """Test: Identische Ketten haben stabiles π"""
        analysis = MarkovStabilityAnalysis()
        
        for i in range(3):
            mc = MarkovChain(states=['A', 'B'])
            P = np.array([[0.7, 0.3], [0.4, 0.6]])
            mc.set_transition_matrix(P)
            analysis.add_markov_chain(mc)
        
        result = analysis.analyze()
        stat = result['stationary_stability']
        
        assert len(stat['distributions']) == 3
        assert len(stat['l1_changes']) == 2
        
        # Änderungen sollten nahe 0 sein
        for change in stat['l1_changes']:
            assert change < 1e-6
    
    def test_stationary_stability_changing_chains(self):
        """Test: Sich ändernde Ketten haben variierende π"""
        analysis = MarkovStabilityAnalysis()
        
        # MC0
        mc0 = MarkovChain(states=['A', 'B'])
        P0 = np.array([[0.9, 0.1], [0.1, 0.9]])
        mc0.set_transition_matrix(P0)
        analysis.add_markov_chain(mc0)
        
        # MC1 - andere Übergangswahrscheinlichkeiten
        mc1 = MarkovChain(states=['A', 'B'])
        P1 = np.array([[0.3, 0.7], [0.7, 0.3]])
        mc1.set_transition_matrix(P1)
        analysis.add_markov_chain(mc1)
        
        result = analysis.analyze()
        stat = result['stationary_stability']
        
        assert len(stat['l1_changes']) == 1
        assert stat['l1_changes'][0] > 0.1  # Signifikante Änderung
    
    def test_stationary_stability_no_ergodic_chains(self):
        """Test: Keine ergodischen Ketten - keine Verteilungen"""
        analysis = MarkovStabilityAnalysis()
        
        for i in range(2):
            mc = MarkovChain(states=['A', 'B'])
            P = np.array([[0.0, 1.0], [1.0, 0.0]])  # Periodisch
            mc.set_transition_matrix(P)
            analysis.add_markov_chain(mc)
        
        result = analysis.analyze()
        stat = result['stationary_stability']
        
        assert len(stat['distributions']) == 0
        assert len(stat['l1_changes']) == 0
        assert stat['max_change'] == 0.0


class TestLongestSequences:
    """Tests für längste Subgraph-Sequenzen"""
    
    def test_longest_sequence_monotone_growth(self):
        """Test: Monoton wachsende Sequenz"""
        analysis = MarkovStabilityAnalysis()
        
        # MC0: Keine Übergänge
        mc0 = MarkovChain(states=['A', 'B'])
        P0 = np.array([[1.0, 0.0], [0.0, 1.0]])
        mc0.set_transition_matrix(P0)
        analysis.add_markov_chain(mc0)
        
        # MC1: Ein Übergang hinzugefügt
        mc1 = MarkovChain(states=['A', 'B'])
        P1 = np.array([[0.5, 0.5], [0.0, 1.0]])
        mc1.set_transition_matrix(P1)
        analysis.add_markov_chain(mc1)
        
        # MC2: Beide Richtungen
        mc2 = MarkovChain(states=['A', 'B'])
        P2 = np.array([[0.5, 0.5], [0.5, 0.5]])
        mc2.set_transition_matrix(P2)
        analysis.add_markov_chain(mc2)
        
        result = analysis.analyze()
        sequences = result['longest_sequences']
        
        assert len(sequences) > 0
        # Sollte eine Sequenz der Länge 3 finden
        assert any(seq.length == 3 for seq in sequences)
    
    def test_sequence_monotone_irreducible_flag(self):
        """Test: Monoton irreduzibel Flag"""
        analysis = MarkovStabilityAnalysis()
        
        # Alle Ketten irreduzibel
        for i in range(3):
            mc = MarkovChain(states=['A', 'B'])
            P = np.array([[0.5, 0.5], [0.5, 0.5]])
            mc.set_transition_matrix(P)
            analysis.add_markov_chain(mc)
        
        result = analysis.analyze()
        sequences = result['longest_sequences']
        
        # Sequenz sollte monoton irreduzibel sein
        assert any(seq.is_monotone_irreducible for seq in sequences)


class TestWeatherExample:
    """Tests für das Wetter-Beispiel"""
    
    def test_create_weather_markov_example(self):
        """Test: Wetter-Beispiel erstellen"""
        chains = create_weather_markov_example()
        
        assert len(chains) == 3
        
        # MC0: Nicht irreduzibel
        assert not chains[0].is_irreducible()
        
        # MC1: Immer noch nicht irreduzibel
        assert not chains[1].is_irreducible()
        
        # MC2: Irreduzibel und ergodisch
        assert chains[2].is_irreducible()
        assert chains[2].is_ergodic()
    
    def test_weather_example_full_analysis(self):
        """Test: Vollständige Analyse des Wetter-Beispiels"""
        chains = create_weather_markov_example()
        
        analysis = MarkovStabilityAnalysis()
        for i, mc in enumerate(chains):
            analysis.add_markov_chain(mc, transformation_name=f"MC_{i}")
        
        result = analysis.analyze()
        
        assert result['total_states'] == 3
        
        # Irreduzibilität sollte sich entwickeln
        irr = result['irreducibility_evolution']
        assert irr['evolution'] == [False, False, True]
        assert irr['first_irreducible_step'] == 2
        
        # Ergodizität sollte sich entwickeln
        erg = result['ergodicity_evolution']
        assert erg['final_ergodic'] == True
        
        # Sollte eine stationäre Verteilung für MC2 haben
        stat = result['stationary_stability']
        assert len(stat['distributions']) >= 1


class TestComparisonMatrix:
    """Tests für Vergleichsmatrix"""
    
    def test_comparison_matrix_dimensions(self):
        """Test: Dimensionen der Vergleichsmatrix"""
        analysis = MarkovStabilityAnalysis()
        
        for i in range(4):
            mc = MarkovChain(states=['A', 'B'])
            P = np.array([[0.5, 0.5], [0.5, 0.5]])
            mc.set_transition_matrix(P)
            analysis.add_markov_chain(mc)
        
        result = analysis.analyze()
        
        matrix = result['comparison_matrix']
        assert matrix.shape == (4, 4)
    
    def test_comparison_matrix_diagonal_ones(self):
        """Test: Diagonale enthält nur Einsen"""
        analysis = MarkovStabilityAnalysis()
        
        for i in range(3):
            mc = MarkovChain(states=['A', 'B'])
            P = np.array([[0.5, 0.5], [0.5, 0.5]])
            mc.set_transition_matrix(P)
            analysis.add_markov_chain(mc)
        
        result = analysis.analyze()
        matrix = result['comparison_matrix']
        
        # Diagonale sollte nur 1en enthalten
        assert np.all(np.diag(matrix) == 1)
    
    def test_comparison_matrix_subgraph_relationship(self):
        """Test: Subgraph-Beziehungen in Matrix"""
        analysis = MarkovStabilityAnalysis()
        
        # MC0: Wenige Übergänge
        mc0 = MarkovChain(states=['A', 'B'])
        P0 = np.array([[1.0, 0.0], [0.0, 1.0]])
        mc0.set_transition_matrix(P0)
        analysis.add_markov_chain(mc0)
        
        # MC1: Mehr Übergänge (Subgraph von MC0)
        mc1 = MarkovChain(states=['A', 'B'])
        P1 = np.array([[0.5, 0.5], [0.5, 0.5]])
        mc1.set_transition_matrix(P1)
        analysis.add_markov_chain(mc1)
        
        result = analysis.analyze()
        matrix = result['comparison_matrix']
        
        # MC0 sollte Subgraph von MC1 sein
        assert matrix[0, 1] == 1


class TestPrintAnalysis:
    """Tests für print_analysis Methode"""
    
    def test_print_analysis_no_error(self, capsys):
        """Test: print_analysis läuft ohne Fehler"""
        analysis = MarkovStabilityAnalysis()
        
        for i in range(2):
            mc = MarkovChain(states=['A', 'B'])
            P = np.array([[0.7, 0.3], [0.4, 0.6]])
            mc.set_transition_matrix(P)
            analysis.add_markov_chain(mc)
        
        result = analysis.analyze()
        
        # Sollte ohne Exception laufen
        analysis.print_analysis(result)
        
        captured = capsys.readouterr()
        assert len(captured.out) > 0
        assert "Gesamtanzahl Markov-Ketten" in captured.out
    
    def test_print_analysis_with_sequences(self, capsys):
        """Test: Ausgabe mit Sequenzen"""
        analysis = MarkovStabilityAnalysis()
        
        for i in range(3):
            mc = MarkovChain(states=['A', 'B'])
            P = np.array([[0.5, 0.5], [0.5, 0.5]])
            mc.set_transition_matrix(P)
            analysis.add_markov_chain(mc)
        
        result = analysis.analyze()
        analysis.print_analysis(result)
        
        captured = capsys.readouterr()
        assert "LÄNGSTE SUBGRAPH-SEQUENZEN" in captured.out
        assert "VERGLEICHSMATRIX" in captured.out