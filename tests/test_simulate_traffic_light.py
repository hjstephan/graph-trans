"""
test_simulate_traffic_light.py - Tests für Ampelsimulation
"""

import pytest
import io
import sys
from unittest.mock import patch
from src.graph import Graph
from src.simulate_traffic_light import (
    create_initial_state,
    create_green_to_yellow,
    create_yellow_to_red,
    create_red_to_red_yellow,
    create_red_yellow_to_green,
    print_graph_state,
    run_simulation
)


class TestCreateInitialState:
    """Tests für create_initial_state"""
    
    def test_creates_correct_nodes(self):
        """Test: Initial state hat korrekte Knoten"""
        graph = create_initial_state()
        
        assert 'ampel' in graph
        assert 'nord' in graph
        assert 'süd' in graph
        assert 'grün' in graph
        
        assert len(graph) == 4
    
    def test_node_attributes(self):
        """Test: Knoten haben korrekte Attribute"""
        graph = create_initial_state()
        
        ampel = graph.get_node('ampel')
        assert ampel.attributes['type'] == 'traffic_light'
        
        grün = graph.get_node('grün')
        assert grün.attributes['type'] == 'signal'
        assert grün.attributes['active'] == 'true'
    
    def test_creates_correct_edges(self):
        """Test: Initial state hat korrekte Kanten"""
        graph = create_initial_state()
        edges = graph.get_edges()
        
        assert len(edges) == 3
        
        edge_pairs = {(e.from_node, e.to_node) for e in edges}
        assert ('ampel', 'grün') in edge_pairs
        assert ('grün', 'nord') in edge_pairs
        assert ('grün', 'süd') in edge_pairs


class TestTransformations:
    """Tests für Transformations-Erstellung"""
    
    def test_green_to_yellow_transformation(self):
        """Test: Grün→Gelb Transformation"""
        t = create_green_to_yellow()
        
        assert t.name == 'Grün → Gelb'
        assert len(t.left) == 4
        assert len(t.right) == 4
        
        # Linke Seite hat 'grün' mit red color
        assert 'grün' in t.left
        grün = t.left.get_node('grün')
        assert grün.color == 'red'
        
        # Rechte Seite hat 'gelb' mit green color
        assert 'gelb' in t.right
        gelb = t.right.get_node('gelb')
        assert gelb.color == 'green'
    
    def test_yellow_to_red_transformation(self):
        """Test: Gelb→Rot Transformation"""
        t = create_yellow_to_red()
        
        assert t.name == 'Gelb → Rot'
        assert 'gelb' in t.left
        assert 'rot' in t.right
        
        gelb = t.left.get_node('gelb')
        assert gelb.color == 'red'
        
        rot = t.right.get_node('rot')
        assert rot.color == 'green'
    
    def test_red_to_red_yellow_transformation(self):
        """Test: Rot→Rot-Gelb Transformation"""
        t = create_red_to_red_yellow()
        
        assert t.name == 'Rot → Rot-Gelb'
        assert 'rot' in t.left
        assert 'rot_gelb' in t.right
        
        rot = t.left.get_node('rot')
        assert rot.color == 'red'
        
        rot_gelb = t.right.get_node('rot_gelb')
        assert rot_gelb.color == 'green'
    
    def test_red_yellow_to_green_transformation(self):
        """Test: Rot-Gelb→Grün Transformation"""
        t = create_red_yellow_to_green()
        
        assert t.name == 'Rot-Gelb → Grün'
        assert 'rot_gelb' in t.left
        assert 'grün' in t.right
        
        rot_gelb = t.left.get_node('rot_gelb')
        assert rot_gelb.color == 'red'
        
        grün = t.right.get_node('grün')
        assert grün.color == 'green'
    
    def test_transformation_edges(self):
        """Test: Transformationen haben korrekte Kanten"""
        t = create_green_to_yellow()
        
        left_edges = t.left.get_edges()
        assert len(left_edges) == 3
        
        right_edges = t.right.get_edges()
        assert len(right_edges) == 3


class TestPrintGraphState:
    """Tests für print_graph_state"""
    
    def test_prints_graph_information(self, capsys):
        """Test: Gibt Graph-Information aus"""
        graph = create_initial_state()
        
        print_graph_state(graph, "Test Title")
        
        captured = capsys.readouterr()
        assert "Test Title" in captured.out
        assert "Knoten" in captured.out
        assert "Kanten" in captured.out
        assert "ampel" in captured.out
        assert "grün" in captured.out
    
    def test_prints_node_count(self, capsys):
        """Test: Gibt Anzahl Knoten aus"""
        graph = create_initial_state()
        
        print_graph_state(graph, "Test")
        
        captured = capsys.readouterr()
        assert "Knoten (4)" in captured.out
    
    def test_prints_edge_count(self, capsys):
        """Test: Gibt Anzahl Kanten aus"""
        graph = create_initial_state()
        
        print_graph_state(graph, "Test")
        
        captured = capsys.readouterr()
        assert "Kanten (3)" in captured.out
    
    def test_prints_node_attributes(self, capsys):
        """Test: Gibt Knoten-Attribute aus"""
        graph = Graph()
        graph.add_node('test', color='red', attr1='value1', attr2='value2')
        
        print_graph_state(graph, "Test")
        
        captured = capsys.readouterr()
        assert "attr1=value1" in captured.out
        assert "attr2=value2" in captured.out
    
    def test_prints_color_symbols(self, capsys):
        """Test: Gibt Farb-Symbole aus"""
        graph = Graph()
        graph.add_node('black_node', color='black')
        graph.add_node('red_node', color='red')
        graph.add_node('green_node', color='green')
        
        print_graph_state(graph, "Test")
        
        captured = capsys.readouterr()
        # Überprüfe dass Symbole verwendet werden
        assert "⚫" in captured.out or "🔴" in captured.out or "🟢" in captured.out


class TestRunSimulation:
    """Tests für run_simulation"""
    
    @patch('src.simulate_traffic_light.time.sleep')
    def test_runs_complete_cycle(self, mock_sleep, capsys):
        """Test: Führt vollständigen Zyklus aus"""
        run_simulation(cycles=1, delay=0.0)
        
        captured = capsys.readouterr()
        
        # Prüfe dass alle Transformationen ausgeführt wurden
        assert "Grün → Gelb" in captured.out
        assert "Gelb → Rot" in captured.out
        assert "Rot → Rot-Gelb" in captured.out
        assert "Rot-Gelb → Grün" in captured.out
        assert "Simulation abgeschlossen" in captured.out
    
    @patch('src.simulate_traffic_light.time.sleep')
    def test_runs_multiple_cycles(self, mock_sleep, capsys):
        """Test: Führt mehrere Zyklen aus"""
        run_simulation(cycles=2, delay=0.0)
        
        captured = capsys.readouterr()
        
        # Sollte ZYKLUS 1 und ZYKLUS 2 ausgeben
        assert "ZYKLUS 1" in captured.out
        assert "ZYKLUS 2" in captured.out
    
    @patch('src.simulate_traffic_light.time.sleep')
    def test_respects_delay(self, mock_sleep):
        """Test: Berücksichtigt Verzögerung"""
        run_simulation(cycles=1, delay=0.5)
        
        # time.sleep sollte 4 mal aufgerufen werden (4 Transformationen)
        assert mock_sleep.call_count == 4
        mock_sleep.assert_called_with(0.5)
    
    @patch('src.simulate_traffic_light.time.sleep')
    def test_shows_initial_state(self, mock_sleep, capsys):
        """Test: Zeigt Anfangszustand"""
        run_simulation(cycles=1, delay=0.0)
        
        captured = capsys.readouterr()
        assert "ANFANGSZUSTAND" in captured.out
        assert "Grün" in captured.out
    
    @patch('src.simulate_traffic_light.time.sleep')
    def test_shows_transformation_success(self, mock_sleep, capsys):
        """Test: Zeigt Transformations-Erfolg"""
        run_simulation(cycles=1, delay=0.0)
        
        captured = capsys.readouterr()
        assert "✅ Transformation erfolgreich angewendet" in captured.out
    
    @patch('src.simulate_traffic_light.Transformation.apply')
    @patch('src.simulate_traffic_light.time.sleep')
    def test_handles_transformation_error(self, mock_sleep, mock_apply, capsys):
        """Test: Behandelt Transformations-Fehler"""
        # Lasse die erste Anwendung einen Fehler werfen
        mock_apply.side_effect = ValueError("Test error")
        
        run_simulation(cycles=1, delay=0.0)
        
        captured = capsys.readouterr()
        assert "❌ Fehler" in captured.out
        assert "Test error" in captured.out
    
    @patch('src.simulate_traffic_light.time.sleep')
    def test_zero_cycles(self, mock_sleep, capsys):
        """Test: Null Zyklen"""
        run_simulation(cycles=0, delay=0.0)
        
        captured = capsys.readouterr()
        # Sollte nur Anfangszustand zeigen
        assert "ANFANGSZUSTAND" in captured.out
        assert "Simulation abgeschlossen" in captured.out
        # Keine Zyklen
        assert "ZYKLUS" not in captured.out


class TestTransformationApplication:
    """Tests für Transformation-Anwendung"""
    
    def test_full_cycle_transformation(self):
        """Test: Vollständiger Transformationszyklus"""
        # Starte mit Grün
        state = create_initial_state()
        assert 'grün' in state
        
        # Grün → Gelb
        t1 = create_green_to_yellow()
        state = t1.apply(state)
        assert 'gelb' in state
        assert 'grün' not in state
        
        # Gelb → Rot
        t2 = create_yellow_to_red()
        state = t2.apply(state)
        assert 'rot' in state
        assert 'gelb' not in state
        
        # Rot → Rot-Gelb
        t3 = create_red_to_red_yellow()
        state = t3.apply(state)
        assert 'rot_gelb' in state
        assert 'rot' not in state
        
        # Rot-Gelb → Grün
        t4 = create_red_yellow_to_green()
        state = t4.apply(state)
        assert 'grün' in state
        assert 'rot_gelb' not in state
    
    def test_transformation_preserves_infrastructure(self):
        """Test: Transformationen erhalten Infrastruktur"""
        state = create_initial_state()
        
        # Wende alle Transformationen an
        transformations = [
            create_green_to_yellow(),
            create_yellow_to_red(),
            create_red_to_red_yellow(),
            create_red_yellow_to_green()
        ]
        
        for t in transformations:
            state = t.apply(state)
            
            # Infrastruktur sollte immer vorhanden sein
            assert 'ampel' in state
            assert 'nord' in state
            assert 'süd' in state
    
    def test_edge_types_change_correctly(self):
        """Test: Kantentypen ändern sich korrekt"""
        state = create_initial_state()
        
        # Initial: grün 'allows' nord
        edges = state.get_edges()
        grün_nord = [e for e in edges if e.from_node == 'grün' and e.to_node == 'nord'][0]
        assert grün_nord.attributes['type'] == 'allows'
        
        # Nach Grün→Gelb: gelb 'warns' nord
        t = create_green_to_yellow()
        state = t.apply(state)
        edges = state.get_edges()
        gelb_nord = [e for e in edges if e.from_node == 'gelb' and e.to_node == 'nord'][0]
        assert gelb_nord.attributes['type'] == 'warns'


class TestEdgeCases:
    """Tests für Grenzfälle"""
    
    def test_empty_graph_print(self, capsys):
        """Test: Ausgabe eines leeren Graphen"""
        graph = Graph()
        print_graph_state(graph, "Empty Graph")
        
        captured = capsys.readouterr()
        assert "Knoten (0)" in captured.out
        assert "Kanten (0)" in captured.out
    
    def test_graph_with_no_attributes(self, capsys):
        """Test: Graph ohne Attribute"""
        graph = Graph()
        graph.add_node('node1')
        graph.add_node('node2')
        graph.add_edge('node1', 'node2')
        
        print_graph_state(graph, "Test")
        
        captured = capsys.readouterr()
        assert "node1" in captured.out
        assert "node2" in captured.out
    
    @patch('src.simulate_traffic_light.time.sleep')
    def test_large_number_of_cycles(self, mock_sleep, capsys):
        """Test: Große Anzahl von Zyklen"""
        run_simulation(cycles=10, delay=0.0)
        
        captured = capsys.readouterr()
        
        # Sollte ZYKLUS 1 bis ZYKLUS 10 enthalten
        assert "ZYKLUS 1" in captured.out
        assert "ZYKLUS 10" in captured.out


class TestIntegration:
    """Integrationstests"""
    
    @patch('src.simulate_traffic_light.time.sleep')
    def test_complete_simulation_output(self, mock_sleep, capsys):
        """Test: Vollständige Simulations-Ausgabe"""
        run_simulation(cycles=1, delay=0.0)
        
        captured = capsys.readouterr()
        output = captured.out
        
        # Prüfe Header
        assert "🚦 AMPELKREUZUNG SIMULATION" in output
        
        # Prüfe dass alle 4 Transformationen erscheinen
        assert "Grün → Gelb" in output
        assert "Gelb → Rot" in output
        assert "Rot → Rot-Gelb" in output
        assert "Rot-Gelb → Grün" in output
        
        # Prüfe Erfolgs-Marker
        assert output.count("✅ Transformation erfolgreich angewendet") == 4
        
        # Prüfe Abschluss
        assert "✨ Simulation abgeschlossen!" in output
    
    def test_transformations_are_reversible(self):
        """Test: Zyklus führt zurück zum Anfang"""
        initial = create_initial_state()
        initial_edges = len(initial.get_edges())
        initial_nodes = len(initial)
        
        # Führe vollständigen Zyklus aus
        state = initial
        transformations = [
            create_green_to_yellow(),
            create_yellow_to_red(),
            create_red_to_red_yellow(),
            create_red_yellow_to_green()
        ]
        
        for t in transformations:
            state = t.apply(state)
        
        # Nach vollständigem Zyklus: gleiche Struktur wie am Anfang
        assert len(state) == initial_nodes
        assert len(state.get_edges()) == initial_edges
        assert 'grün' in state
        assert 'ampel' in state
