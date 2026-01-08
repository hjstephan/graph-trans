"""
Tests für die Graph-Klasse
"""

import pytest
from graph import Graph, Node, Edge


class TestNode:
    """Tests für die Node-Klasse"""

    def test_node_creation(self):
        """Test: Knoten kann erstellt werden"""
        node = Node(id='n1', color='black')
        assert node.id == 'n1'
        assert node.color == 'black'
        assert node.attributes == {}

    def test_node_with_attributes(self):
        """Test: Knoten mit Attributen"""
        node = Node(id='n1', color='red', attributes={'label': 'Test'})
        assert node.attributes['label'] == 'Test'

    def test_node_equality(self):
        """Test: Knoten-Gleichheit basiert auf ID"""
        node1 = Node(id='n1', color='black')
        node2 = Node(id='n1', color='red')
        node3 = Node(id='n2', color='black')
        
        assert node1 == node2
        assert node1 != node3


class TestEdge:
    """Tests für die Edge-Klasse"""

    def test_edge_creation(self):
        """Test: Kante kann erstellt werden"""
        edge = Edge(from_node='n1', to_node='n2', color='black')
        assert edge.from_node == 'n1'
        assert edge.to_node == 'n2'
        assert edge.color == 'black'

    def test_edge_equality(self):
        """Test: Kanten-Gleichheit basiert auf from/to"""
        edge1 = Edge(from_node='n1', to_node='n2', color='black')
        edge2 = Edge(from_node='n1', to_node='n2', color='red')
        edge3 = Edge(from_node='n2', to_node='n1', color='black')
        
        assert edge1 == edge2
        assert edge1 != edge3


class TestGraph:
    """Tests für die Graph-Klasse"""

    def test_empty_graph(self):
        """Test: Leerer Graph kann erstellt werden"""
        graph = Graph()
        assert len(graph) == 0
        assert graph.get_nodes() == []
        assert graph.get_edges() == []

    def test_graph_from_dicts(self):
        """Test: Graph aus Dictionaries erstellen"""
        nodes = [
            {'id': 'n1', 'color': 'black'},
            {'id': 'n2', 'color': 'red'}
        ]
        edges = [
            {'from': 'n1', 'to': 'n2', 'color': 'black'}
        ]
        
        graph = Graph(nodes=nodes, edges=edges)
        assert len(graph) == 2
        assert len(graph.get_edges()) == 1

    def test_add_node(self):
        """Test: Knoten hinzufügen"""
        graph = Graph()
        node = graph.add_node(id='n1', color='black')
        
        assert len(graph) == 1
        assert 'n1' in graph
        assert node.id == 'n1'

    def test_add_edge(self):
        """Test: Kante hinzufügen"""
        graph = Graph()
        graph.add_node(id='n1')
        graph.add_node(id='n2')
        edge = graph.add_edge(from_node='n1', to_node='n2', color='green')
        
        assert len(graph.get_edges()) == 1
        assert edge.from_node == 'n1'
        assert edge.to_node == 'n2'

    def test_add_edge_invalid_nodes(self):
        """Test: Kante mit nicht-existierenden Knoten"""
        graph = Graph()
        graph.add_node(id='n1')
        
        with pytest.raises(ValueError):
            graph.add_edge(from_node='n1', to_node='n2')

    def test_get_node(self):
        """Test: Knoten abrufen"""
        graph = Graph()
        graph.add_node(id='n1', color='black')
        
        node = graph.get_node('n1')
        assert node is not None
        assert node.id == 'n1'
        
        missing = graph.get_node('n99')
        assert missing is None

    def test_get_nodes_by_color(self):
        """Test: Knoten nach Farbe filtern"""
        graph = Graph()
        graph.add_node(id='n1', color='black')
        graph.add_node(id='n2', color='red')
        graph.add_node(id='n3', color='green')
        
        black_nodes = graph.get_nodes(color='black')
        red_nodes = graph.get_nodes(color='red')
        
        assert len(black_nodes) == 1
        assert len(red_nodes) == 1
        assert black_nodes[0].id == 'n1'
        assert red_nodes[0].id == 'n2'

    def test_get_edges_by_color(self):
        """Test: Kanten nach Farbe filtern"""
        graph = Graph()
        graph.add_node(id='n1')
        graph.add_node(id='n2')
        graph.add_node(id='n3')
        graph.add_edge('n1', 'n2', color='black')
        graph.add_edge('n2', 'n3', color='red')
        
        black_edges = graph.get_edges(color='black')
        red_edges = graph.get_edges(color='red')
        
        assert len(black_edges) == 1
        assert len(red_edges) == 1

    def test_remove_node(self):
        """Test: Knoten entfernen"""
        graph = Graph()
        graph.add_node(id='n1')
        graph.add_node(id='n2')
        graph.add_edge('n1', 'n2')
        
        result = graph.remove_node('n1')
        
        assert result is True
        assert len(graph) == 1
        assert 'n1' not in graph
        # Kante sollte auch entfernt sein
        assert len(graph.get_edges()) == 0

    def test_remove_nonexistent_node(self):
        """Test: Nicht-existierenden Knoten entfernen"""
        graph = Graph()
        result = graph.remove_node('n99')
        assert result is False

    def test_remove_edge(self):
        """Test: Kante entfernen"""
        graph = Graph()
        graph.add_node(id='n1')
        graph.add_node(id='n2')
        graph.add_edge('n1', 'n2')
        
        result = graph.remove_edge('n1', 'n2')
        
        assert result is True
        assert len(graph.get_edges()) == 0

    def test_remove_nonexistent_edge(self):
        """Test: Nicht-existierende Kante entfernen"""
        graph = Graph()
        graph.add_node(id='n1')
        graph.add_node(id='n2')
        
        result = graph.remove_edge('n1', 'n2')
        assert result is False

    def test_copy(self):
        """Test: Graph kopieren"""
        graph = Graph()
        graph.add_node(id='n1', color='black', label='Test')
        graph.add_node(id='n2')
        graph.add_edge('n1', 'n2', color='red')
        
        copy = graph.copy()
        
        assert len(copy) == len(graph)
        assert len(copy.get_edges()) == len(graph.get_edges())
        
        # Änderungen am Original sollten Kopie nicht beeinflussen
        graph.add_node(id='n3')
        assert len(copy) == 2
        assert len(graph) == 3

    def test_contains(self):
        """Test: __contains__ Operator"""
        graph = Graph()
        graph.add_node(id='n1')
        
        assert 'n1' in graph
        assert 'n99' not in graph

    def test_repr(self):
        """Test: String-Repräsentation"""
        graph = Graph()
        graph.add_node(id='n1')
        graph.add_node(id='n2')
        graph.add_edge('n1', 'n2')
        
        repr_str = repr(graph)
        assert 'Graph' in repr_str
        assert 'nodes=2' in repr_str
        assert 'edges=1' in repr_str
