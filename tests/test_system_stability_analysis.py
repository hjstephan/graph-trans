"""
test_system_stability_analysis.py - Tests für Systemstabilitätsanalyse
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from src.graph import Graph
from src.transformation import Transformation
from src.system_stability_analysis import (
    SystemState,
    SubgraphSequence,
    SystemSimulation,
    create_traffic_light_system
)


class TestSystemState:
    """Tests für SystemState Dataclass"""
    
    def test_system_state_creation(self):
        """Test: SystemState erstellen"""
        graph = Graph()
        graph.add_node('A')
        graph.add_node('B')
        graph.add_edge('A', 'B')
        
        state = SystemState(
            step=0,
            timestamp=123.456,
            graph=graph,
            transformation_name="Test"
        )
        
        assert state.step == 0
        assert state.timestamp == 123.456
        assert state.transformation_name == "Test"
        assert state.graph == graph
    
    def test_post_init_computes_matrix(self):
        """Test: __post_init__ berechnet Adjazenzmatrix"""
        graph = Graph()
        graph.add_node('A')
        graph.add_node('B')
        graph.add_edge('A', 'B')
        
        state = SystemState(
            step=0,
            timestamp=0.0,
            graph=graph
        )
        
        # Matrix und Mapping sollten automatisch berechnet werden
        assert state.matrix is not None
        assert state.node_mapping is not None
        assert state.matrix.shape == (2, 2)
    
    def test_matrix_provided(self):
        """Test: Matrix kann vorgegeben werden"""
        graph = Graph()
        graph.add_node('A')
        
        matrix = np.array([[1]])
        mapping = {'A': 0}
        
        state = SystemState(
            step=0,
            timestamp=0.0,
            graph=graph,
            matrix=matrix,
            node_mapping=mapping
        )
        
        assert np.array_equal(state.matrix, matrix)
        assert state.node_mapping == mapping


class TestSubgraphSequence:
    """Tests für SubgraphSequence Dataclass"""
    
    def test_subgraph_sequence_creation(self):
        """Test: SubgraphSequence erstellen"""
        seq = SubgraphSequence(
            start_step=0,
            end_step=5,
            length=6
        )
        
        assert seq.start_step == 0
        assert seq.end_step == 5
        assert seq.length == 6
        assert seq.states == []
    
    def test_subgraph_sequence_with_states(self):
        """Test: SubgraphSequence mit States"""
        graph = Graph()
        graph.add_node('A')
        
        state1 = SystemState(0, 0.0, graph)
        state2 = SystemState(1, 1.0, graph)
        
        seq = SubgraphSequence(
            start_step=0,
            end_step=1,
            length=2,
            states=[state1, state2]
        )
        
        assert len(seq.states) == 2
        assert seq.states[0] == state1
        assert seq.states[1] == state2
    
    def test_repr(self):
        """Test: String-Repräsentation"""
        seq = SubgraphSequence(
            start_step=2,
            end_step=7,
            length=6
        )
        
        repr_str = repr(seq)
        assert "SubgraphSequence" in repr_str
        assert "steps=2-7" in repr_str
        assert "length=6" in repr_str


class TestSystemSimulation:
    """Tests für SystemSimulation Klasse"""
    
    def test_initialization(self):
        """Test: Initialisierung"""
        sim = SystemSimulation()
        
        assert sim.states == []
        assert sim.transformations == []
        assert sim.subgraph_algo is not None
    
    def test_add_transformation(self):
        """Test: Transformation hinzufügen"""
        sim = SystemSimulation()
        
        left = Graph()
        left.add_node('A', color='black')
        right = Graph()
        right.add_node('A', color='black')
        
        t = Transformation('Test', left, right)
        sim.add_transformation(t)
        
        assert len(sim.transformations) == 1
        assert sim.transformations[0] == t
    
    def test_add_multiple_transformations(self):
        """Test: Mehrere Transformationen hinzufügen"""
        sim = SystemSimulation()
        
        left = Graph()
        left.add_node('A', color='black')
        right = Graph()
        right.add_node('A', color='black')
        
        for i in range(3):
            t = Transformation(f'T{i}', left, right)
            sim.add_transformation(t)
        
        assert len(sim.transformations) == 3


class TestSystemSimulationRun:
    """Tests für SystemSimulation.run()"""
    
    @patch('src.system_stability_analysis.time.sleep')
    def test_run_creates_initial_state(self, mock_sleep):
        """Test: run() erstellt Anfangszustand"""
        sim = SystemSimulation()
        
        initial = Graph()
        initial.add_node('A')
        
        states = sim.run(initial, steps=0, delay=0.0)
        
        assert len(states) == 1
        assert states[0].step == 0
        assert states[0].transformation_name == "Initial"
    
    @patch('src.system_stability_analysis.time.sleep')
    def test_run_applies_transformations(self, mock_sleep):
        """Test: run() wendet Transformationen an"""
        sim = SystemSimulation()
        
        # Erstelle einfache Transformation
        left = Graph()
        left.add_node('A', color='black')
        left.add_node('B', color='red')
        left.add_edge('A', 'B', color='red')
        
        right = Graph()
        right.add_node('A', color='black')
        right.add_node('C', color='green')
        right.add_edge('A', 'C', color='green')
        
        t = Transformation('A→B to A→C', left, right)
        sim.add_transformation(t)
        
        # Initialer Zustand
        initial = Graph()
        initial.add_node('A')
        initial.add_node('B')
        initial.add_edge('A', 'B')
        
        states = sim.run(initial, steps=1, delay=0.0)
        
        assert len(states) == 2  # Initial + 1 Transformation
        assert states[1].step == 1
        assert states[1].transformation_name == 'A→B to A→C'
    
    @patch('src.system_stability_analysis.time.sleep')
    def test_run_cycles_through_transformations(self, mock_sleep):
        """Test: run() zykliert durch Transformationen"""
        sim = SystemSimulation()
        
        left = Graph()
        left.add_node('A', color='black')
        right = Graph()
        right.add_node('A', color='black')
        
        t1 = Transformation('T1', left, right)
        t2 = Transformation('T2', left, right)
        
        sim.add_transformation(t1)
        sim.add_transformation(t2)
        
        initial = Graph()
        initial.add_node('A')
        
        states = sim.run(initial, steps=4, delay=0.0)
        
        # Sollte T1, T2, T1, T2 anwenden
        assert states[1].transformation_name == 'T1'
        assert states[2].transformation_name == 'T2'
        assert states[3].transformation_name == 'T1'
        assert states[4].transformation_name == 'T2'
    
    @patch('src.system_stability_analysis.time.sleep')
    def test_run_respects_delay(self, mock_sleep):
        """Test: run() berücksichtigt Verzögerung"""
        sim = SystemSimulation()
        
        left = Graph()
        left.add_node('A', color='black')
        right = Graph()
        right.add_node('A', color='black')
        
        t = Transformation('Test', left, right)
        sim.add_transformation(t)
        
        initial = Graph()
        initial.add_node('A')
        
        sim.run(initial, steps=3, delay=0.5)
        
        # sleep sollte 3 mal mit 0.5 aufgerufen werden
        assert mock_sleep.call_count == 3
        mock_sleep.assert_called_with(0.5)
    
    @patch('src.system_stability_analysis.time.sleep')
    def test_run_handles_transformation_error(self, mock_sleep, capsys):
        """Test: run() behandelt Transformationsfehler"""
        sim = SystemSimulation()
        
        left = Graph()
        left.add_node('X', color='black')  # Existiert nicht im initial state
        right = Graph()
        right.add_node('X', color='black')
        
        t = Transformation('Bad Transform', left, right)
        sim.add_transformation(t)
        
        initial = Graph()
        initial.add_node('A')
        
        states = sim.run(initial, steps=2, delay=0.0)
        
        # Sollte nach Fehler abbrechen
        captured = capsys.readouterr()
        assert "Fehler" in captured.out
        
        # Nur Initial-State wurde hinzugefügt
        assert len(states) == 1
    
    @patch('src.system_stability_analysis.time.sleep')
    def test_run_stores_graph_copies(self, mock_sleep):
        """Test: run() speichert Kopien der Graphen"""
        sim = SystemSimulation()
        
        left = Graph()
        left.add_node('A', color='black')
        right = Graph()
        right.add_node('A', color='black')
        
        t = Transformation('Test', left, right)
        sim.add_transformation(t)
        
        initial = Graph()
        initial.add_node('A')
        
        states = sim.run(initial, steps=1, delay=0.0)
        
        # Ändern des initial sollte states nicht beeinflussen
        initial.add_node('Z')
        
        assert 'Z' not in states[0].graph


class TestFindLongestSequences:
    """Tests für _find_longest_sequences()"""
    
    def test_single_monotone_sequence(self):
        """Test: Einzelne monotone Sequenz"""
        sim = SystemSimulation()
        
        # Erstelle 3 Zustände: G0 ⊆ G1 ⊆ G2
        g0 = Graph()
        g0.add_node('A')
        
        g1 = Graph()
        g1.add_node('A')
        g1.add_node('B')
        g1.add_edge('A', 'B')
        
        g2 = Graph()
        g2.add_node('A')
        g2.add_node('B')
        g2.add_node('C')
        g2.add_edge('A', 'B')
        g2.add_edge('B', 'C')
        
        sim.states = [
            SystemState(0, 0.0, g0),
            SystemState(1, 1.0, g1),
            SystemState(2, 2.0, g2)
        ]
        
        # Erstelle Vergleichsmatrix
        comparison_matrix = np.array([
            [1, 1, 1],  # G0 ⊆ G0, G1, G2
            [0, 1, 1],  # G1 ⊆ G1, G2
            [0, 0, 1]   # G2 ⊆ G2
        ])
        
        sequences = sim._find_longest_sequences(comparison_matrix)
        
        assert len(sequences) > 0
        # Längste Sequenz sollte Länge 3 haben
        assert any(seq.length == 3 for seq in sequences)
    
    def test_multiple_sequences(self):
        """Test: Mehrere getrennte Sequenzen"""
        sim = SystemSimulation()
        
        for i in range(5):
            g = Graph()
            g.add_node(f'N{i}')
            sim.states.append(SystemState(i, float(i), g))
        
        # Zwei separate Sequenzen: 0-1-2 und 3-4
        comparison_matrix = np.array([
            [1, 1, 1, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1]
        ])
        
        sequences = sim._find_longest_sequences(comparison_matrix)
        
        # Sollte mindestens eine Sequenz der Länge 3 finden
        assert any(seq.length == 3 for seq in sequences)
    
    def test_no_sequences(self):
        """Test: Keine zusammenhängenden Sequenzen"""
        sim = SystemSimulation()
        
        for i in range(3):
            g = Graph()
            g.add_node(f'N{i}')
            sim.states.append(SystemState(i, float(i), g))
        
        # Keine Subgraph-Beziehungen außer Diagonal
        comparison_matrix = np.eye(3, dtype=int)
        
        sequences = sim._find_longest_sequences(comparison_matrix)
        
        # Alle Sequenzen haben Länge 1
        assert all(seq.length == 1 for seq in sequences)


class TestFindStableStates:
    """Tests für _find_stable_states()"""
    
    def test_finds_stable_state(self):
        """Test: Findet stabilen Zustand"""
        sim = SystemSimulation()
        
        g1 = Graph()
        g1.add_node('A')
        
        g2 = Graph()
        g2.add_node('A')  # Identisch zu g1
        
        sim.states = [
            SystemState(0, 0.0, g1),
            SystemState(1, 1.0, g2)
        ]
        
        # Beide Zustände sind identisch
        comparison_matrix = np.ones((2, 2), dtype=int)
        
        stable = sim._find_stable_states(comparison_matrix)
        
        assert 0 in stable
    
    def test_no_stable_states(self):
        """Test: Keine stabilen Zustände"""
        sim = SystemSimulation()
        
        g1 = Graph()
        g1.add_node('A')
        
        g2 = Graph()
        g2.add_node('B')  # Unterschiedlich
        
        sim.states = [
            SystemState(0, 0.0, g1),
            SystemState(1, 1.0, g2)
        ]
        
        comparison_matrix = np.eye(2, dtype=int)
        
        stable = sim._find_stable_states(comparison_matrix)
        
        assert len(stable) == 0
    
    def test_multiple_stable_states(self):
        """Test: Mehrere stabile Zustände"""
        sim = SystemSimulation()
        
        g = Graph()
        g.add_node('A')
        
        # 5 identische Zustände
        for i in range(5):
            sim.states.append(SystemState(i, float(i), g.copy()))
        
        comparison_matrix = np.ones((5, 5), dtype=int)
        
        stable = sim._find_stable_states(comparison_matrix)
        
        # Sollte 0,1,2,3 enthalten (nicht 4, da es kein i+1 gibt)
        assert len(stable) >= 3


class TestFindCycles:
    """Tests für _find_cycles()"""
    
    def test_finds_simple_cycle(self):
        """Test: Findet einfachen Zyklus"""
        sim = SystemSimulation()
        
        g1 = Graph()
        g1.add_node('A')
        
        g2 = Graph()
        g2.add_node('B')
        
        g3 = Graph()
        g3.add_node('A')  # Gleich wie g1
        
        sim.states = [
            SystemState(0, 0.0, g1),
            SystemState(1, 1.0, g2),
            SystemState(2, 2.0, g3)
        ]
        
        cycles = sim._find_cycles()
        
        # Sollte Zyklus zwischen Schritt 0 und 2 finden
        assert (0, 2) in cycles
    
    def test_no_cycles(self):
        """Test: Keine Zyklen"""
        sim = SystemSimulation()
        
        # Erstelle Graphen mit unterschiedlichen Strukturen
        g0 = Graph()
        g0.add_node('A')
        
        g1 = Graph()
        g1.add_node('A')
        g1.add_node('B')
        
        g2 = Graph()
        g2.add_node('A')
        g2.add_node('B')
        g2.add_node('C')
        
        sim.states = [
            SystemState(0, 0.0, g0),
            SystemState(1, 1.0, g1),
            SystemState(2, 2.0, g2)
        ]
        
        cycles = sim._find_cycles()
        
        assert len(cycles) == 0
    
    def test_multiple_cycles(self):
        """Test: Mehrere Zyklen"""
        sim = SystemSimulation()
        
        g1 = Graph()
        g1.add_node('A')
        
        g2 = Graph()
        g2.add_node('B')
        
        # Pattern: g1, g2, g1, g2
        sim.states = [
            SystemState(0, 0.0, g1.copy()),
            SystemState(1, 1.0, g2.copy()),
            SystemState(2, 2.0, g1.copy()),
            SystemState(3, 3.0, g2.copy())
        ]
        
        cycles = sim._find_cycles()
        
        # Sollte mehrere Zyklen finden
        assert (0, 2) in cycles
        assert (1, 3) in cycles


class TestAnalyzeStability:
    """Tests für analyze_stability()"""
    
    @patch('src.system_stability_analysis.time.sleep')
    def test_analyze_with_sufficient_states(self, mock_sleep, capsys):
        """Test: Analyse mit genügend Zuständen"""
        sim = SystemSimulation()
        
        left = Graph()
        left.add_node('A', color='black')
        right = Graph()
        right.add_node('A', color='black')
        
        t = Transformation('Test', left, right)
        sim.add_transformation(t)
        
        initial = Graph()
        initial.add_node('A')
        
        sim.run(initial, steps=2, delay=0.0)
        
        result = sim.analyze_stability()
        
        assert 'total_states' in result
        assert 'comparison_matrix' in result
        assert 'longest_sequences' in result
        assert 'stable_states' in result
        assert 'cycles' in result
        assert 'states' in result
        
        captured = capsys.readouterr()
        assert "STABILITÄTSANALYSE" in captured.out
    
    def test_analyze_insufficient_states(self):
        """Test: Analyse mit zu wenig Zuständen"""
        sim = SystemSimulation()
        
        result = sim.analyze_stability()
        
        assert "error" in result
        assert "Nicht genug" in result["error"]
    
    @patch('src.system_stability_analysis.time.sleep')
    def test_analyze_creates_comparison_matrix(self, mock_sleep):
        """Test: Analyse erstellt Vergleichsmatrix"""
        sim = SystemSimulation()
        
        left = Graph()
        left.add_node('A', color='black')
        right = Graph()
        right.add_node('A', color='black')
        
        t = Transformation('Test', left, right)
        sim.add_transformation(t)
        
        initial = Graph()
        initial.add_node('A')
        
        sim.run(initial, steps=2, delay=0.0)
        result = sim.analyze_stability()
        
        matrix = result['comparison_matrix']
        assert matrix.shape == (3, 3)  # 3 Zustände
        
        # Diagonal sollte 1en sein
        assert np.all(np.diag(matrix) == 1)


class TestPrintAnalysis:
    """Tests für print_analysis()"""
    
    @patch('src.system_stability_analysis.time.sleep')
    def test_prints_all_sections(self, mock_sleep, capsys):
        """Test: Druckt alle Abschnitte"""
        sim = SystemSimulation()
        
        left = Graph()
        left.add_node('A', color='black')
        right = Graph()
        right.add_node('A', color='black')
        
        t = Transformation('Test', left, right)
        sim.add_transformation(t)
        
        initial = Graph()
        initial.add_node('A')
        
        sim.run(initial, steps=2, delay=0.0)
        result = sim.analyze_stability()
        
        sim.print_analysis(result)
        
        captured = capsys.readouterr()
        output = captured.out
        
        assert "Gesamtanzahl Zustände" in output
        assert "LÄNGSTE SUBGRAPH-SEQUENZEN" in output
        assert "STABILE ZUSTÄNDE" in output
        assert "ZYKLEN" in output
        assert "VERGLEICHSMATRIX" in output
    
    def test_prints_no_sequences(self, capsys):
        """Test: Ausgabe wenn keine Sequenzen"""
        sim = SystemSimulation()
        
        g = Graph()
        g.add_node('A')
        sim.states = [SystemState(0, 0.0, g), SystemState(1, 1.0, g)]
        
        result = {
            'total_states': 2,
            'longest_sequences': [],
            'stable_states': [],
            'cycles': [],
            'comparison_matrix': np.eye(2)
        }
        
        sim.print_analysis(result)
        
        captured = capsys.readouterr()
        assert "Keine zusammenhängenden Sequenzen gefunden" in captured.out
    
    def test_prints_stable_states(self, capsys):
        """Test: Ausgabe stabiler Zustände"""
        sim = SystemSimulation()
        
        g = Graph()
        g.add_node('A')
        sim.states = [
            SystemState(0, 0.0, g, transformation_name="T0"),
            SystemState(1, 1.0, g, transformation_name="T1")
        ]
        
        result = {
            'total_states': 2,
            'longest_sequences': [],
            'stable_states': [0],
            'cycles': [],
            'comparison_matrix': np.eye(2)
        }
        
        sim.print_analysis(result)
        
        captured = capsys.readouterr()
        output = captured.out
        
        assert "Gefundene stabile Zustände" in output
        assert "Schritt 0" in output


class TestCreateTrafficLightSystem:
    """Tests für create_traffic_light_system()"""
    
    def test_creates_initial_state(self):
        """Test: Erstellt Anfangszustand"""
        initial, transformations = create_traffic_light_system()
        
        assert len(initial) == 4  # ampel, nord, süd, grün
        assert 'ampel' in initial
        assert 'grün' in initial
    
    def test_creates_four_transformations(self):
        """Test: Erstellt vier Transformationen"""
        initial, transformations = create_traffic_light_system()
        
        assert len(transformations) == 4
    
    def test_transformation_names(self):
        """Test: Transformationen haben korrekte Namen"""
        initial, transformations = create_traffic_light_system()
        
        names = [t.name for t in transformations]
        
        assert 'Grün → Gelb' in names
        assert 'Gelb → Rot' in names
        assert 'Rot → Rot-Gelb' in names
        assert 'Rot-Gelb → Grün' in names
    
    def test_transformations_applicable(self):
        """Test: Transformationen sind anwendbar"""
        initial, transformations = create_traffic_light_system()
        
        state = initial
        
        # Sollte alle Transformationen anwenden können
        for t in transformations:
            state = t.apply(state)
            assert state is not None


class TestIntegration:
    """Integrationstests"""
    
    @patch('src.system_stability_analysis.time.sleep')
    def test_full_traffic_light_simulation(self, mock_sleep, capsys):
        """Test: Vollständige Ampelsimulation"""
        initial, transformations = create_traffic_light_system()
        
        sim = SystemSimulation()
        for t in transformations:
            sim.add_transformation(t)
        
        states = sim.run(initial, steps=8, delay=0.0)
        
        # Sollte 9 Zustände haben (Initial + 8 Schritte)
        assert len(states) == 9
        
        # Analysiere
        result = sim.analyze_stability()
        
        assert result['total_states'] == 9
        assert 'comparison_matrix' in result
    
    @patch('src.system_stability_analysis.time.sleep')
    def test_finds_cycle_in_traffic_light(self, mock_sleep):
        """Test: Findet Zyklus in Ampel"""
        initial, transformations = create_traffic_light_system()
        
        sim = SystemSimulation()
        for t in transformations:
            sim.add_transformation(t)
        
        # Führe 2 vollständige Zyklen aus
        states = sim.run(initial, steps=8, delay=0.0)
        
        result = sim.analyze_stability()
        cycles = result['cycles']
        
        # Nach 4 Schritten wiederholt sich der Zustand
        # Schritt 0 (grün) == Schritt 4 (grün) == Schritt 8 (grün)
        assert len(cycles) > 0
    
    @patch('src.system_stability_analysis.time.sleep')
    def test_print_complete_analysis(self, mock_sleep, capsys):
        """Test: Vollständige Analyse-Ausgabe"""
        initial, transformations = create_traffic_light_system()
        
        sim = SystemSimulation()
        for t in transformations:
            sim.add_transformation(t)
        
        sim.run(initial, steps=4, delay=0.0)
        result = sim.analyze_stability()
        
        sim.print_analysis(result)
        
        captured = capsys.readouterr()
        output = captured.out
        
        # Prüfe dass alle Hauptabschnitte vorhanden sind
        assert "STABILITÄTSANALYSE" in output
        assert "LÄNGSTE SUBGRAPH-SEQUENZEN" in output
        assert "VERGLEICHSMATRIX" in output
