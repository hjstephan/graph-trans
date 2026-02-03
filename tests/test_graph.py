"""
Tests für die Graph-Klasse
"""

import pytest
from src.graph import Graph, Node, Edge


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


class TestGraphSubgraphInterface:
    """Tests für die Subgraph-Algorithmus Schnittstellen"""

    def test_to_adjacency_matrix_empty_graph(self):
        """Test: Leerer Graph zu Adjazenzmatrix"""
        graph = Graph()
        matrix, node_mapping = graph.to_adjacency_matrix()
        
        assert matrix.shape == (0, 0)
        assert len(node_mapping) == 0

    def test_to_adjacency_matrix_single_node(self):
        """Test: Graph mit einem Knoten"""
        graph = Graph()
        graph.add_node(id='n1')
        
        matrix, node_mapping = graph.to_adjacency_matrix()
        
        assert matrix.shape == (1, 1)
        assert matrix[0, 0] == 0
        assert 'n1' in node_mapping
        assert node_mapping['n1'] == 0

    def test_to_adjacency_matrix_with_edges(self):
        """Test: Graph mit Kanten zu Adjazenzmatrix"""
        graph = Graph()
        graph.add_node(id='n1')
        graph.add_node(id='n2')
        graph.add_node(id='n3')
        graph.add_edge('n1', 'n2')
        graph.add_edge('n2', 'n3')
        
        matrix, node_mapping = graph.to_adjacency_matrix()
        
        assert matrix.shape == (3, 3)
        # Prüfe dass Kanten korrekt abgebildet sind
        i1 = node_mapping['n1']
        i2 = node_mapping['n2']
        i3 = node_mapping['n3']
        
        assert matrix[i1, i2] == 1
        assert matrix[i2, i3] == 1
        assert matrix[i1, i3] == 0  # Keine direkte Kante

    def test_to_adjacency_matrix_sorted_nodes(self):
        """Test: Knoten werden sortiert in Matrix übertragen"""
        graph = Graph()
        graph.add_node(id='z')
        graph.add_node(id='a')
        graph.add_node(id='m')
        
        matrix, node_mapping = graph.to_adjacency_matrix()
        
        # Sortierte Reihenfolge sollte sein: a, m, z
        assert node_mapping['a'] == 0
        assert node_mapping['m'] == 1
        assert node_mapping['z'] == 2

    def test_from_adjacency_matrix_empty(self):
        """Test: Leere Matrix zu Graph"""
        import numpy as np
        matrix = np.array([]).reshape(0, 0)
        
        graph = Graph.from_adjacency_matrix(matrix)
        
        assert len(graph) == 0
        assert len(graph.get_edges()) == 0

    def test_from_adjacency_matrix_with_default_ids(self):
        """Test: Matrix zu Graph mit Standard-IDs"""
        import numpy as np
        matrix = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0]
        ])
        
        graph = Graph.from_adjacency_matrix(matrix)
        
        assert len(graph) == 3
        assert 'n0' in graph
        assert 'n1' in graph
        assert 'n2' in graph
        assert len(graph.get_edges()) == 2

    def test_from_adjacency_matrix_with_custom_ids(self):
        """Test: Matrix zu Graph mit benutzerdefinierten IDs"""
        import numpy as np
        matrix = np.array([
            [0, 1],
            [1, 0]
        ])
        node_ids = ['alice', 'bob']
        
        graph = Graph.from_adjacency_matrix(matrix, node_ids=node_ids)
        
        assert len(graph) == 2
        assert 'alice' in graph
        assert 'bob' in graph
        
        edges = graph.get_edges()
        edge_pairs = {(e.from_node, e.to_node) for e in edges}
        assert ('alice', 'bob') in edge_pairs
        assert ('bob', 'alice') in edge_pairs

    def test_from_adjacency_matrix_invalid_node_ids_length(self):
        """Test: Falsche Anzahl von node_ids führt zu Fehler"""
        import numpy as np
        matrix = np.array([
            [0, 1],
            [0, 0]
        ])
        node_ids = ['only_one']
        
        with pytest.raises(ValueError, match="muss mit Matrixgröße"):
            Graph.from_adjacency_matrix(matrix, node_ids=node_ids)

    def test_roundtrip_graph_to_matrix_to_graph(self):
        """Test: Graph → Matrix → Graph Roundtrip"""
        # Original Graph erstellen
        original = Graph()
        original.add_node(id='a')
        original.add_node(id='b')
        original.add_node(id='c')
        original.add_edge('a', 'b')
        original.add_edge('b', 'c')
        original.add_edge('a', 'c')
        
        # Zu Matrix konvertieren
        matrix, node_mapping = original.to_adjacency_matrix()
        
        # Zurück zu Graph konvertieren
        node_ids = sorted(node_mapping.keys())  # Sortiert wie in to_adjacency_matrix
        reconstructed = Graph.from_adjacency_matrix(matrix, node_ids=node_ids)
        
        # Prüfe dass die Graphen äquivalent sind
        assert len(reconstructed) == len(original)
        assert len(reconstructed.get_edges()) == len(original.get_edges())
        
        # Prüfe dass alle Knoten vorhanden sind
        for node_id in ['a', 'b', 'c']:
            assert node_id in reconstructed
        
        # Prüfe dass alle Kanten vorhanden sind
        original_edges = {(e.from_node, e.to_node) for e in original.get_edges()}
        reconstructed_edges = {(e.from_node, e.to_node) for e in reconstructed.get_edges()}
        assert original_edges == reconstructed_edges

    def test_get_node_list_empty(self):
        """Test: Leere Knotenliste für leeren Graph"""
        graph = Graph()
        node_list = graph.get_node_list()
        
        assert node_list == []

    def test_get_node_list_sorted(self):
        """Test: Knotenliste ist sortiert"""
        graph = Graph()
        graph.add_node(id='zebra')
        graph.add_node(id='alpha')
        graph.add_node(id='beta')
        
        node_list = graph.get_node_list()
        
        assert node_list == ['alpha', 'beta', 'zebra']

    def test_get_node_list_consistency(self):
        """Test: get_node_list ist konsistent mit to_adjacency_matrix"""
        graph = Graph()
        graph.add_node(id='x')
        graph.add_node(id='y')
        graph.add_node(id='z')
        
        node_list = graph.get_node_list()
        matrix, node_mapping = graph.to_adjacency_matrix()
        
        # Die Reihenfolge sollte übereinstimmen
        for idx, node_id in enumerate(node_list):
            assert node_mapping[node_id] == idx

    def test_adjacency_matrix_preserves_graph_structure(self):
        """Test: Komplexere Graphstruktur wird korrekt konvertiert"""
        graph = Graph()
        # Erstelle einen kleinen gerichteten Graphen
        for i in range(4):
            graph.add_node(id=f'n{i}')
        
        graph.add_edge('n0', 'n1')
        graph.add_edge('n0', 'n2')
        graph.add_edge('n1', 'n3')
        graph.add_edge('n2', 'n3')
        
        matrix, node_mapping = graph.to_adjacency_matrix()
        
        # Konvertiere zurück
        node_ids = sorted(node_mapping.keys())
        reconstructed = Graph.from_adjacency_matrix(matrix, node_ids=node_ids)
        
        # Prüfe alle Kanten
        original_edges = set()
        for edge in graph.get_edges():
            original_edges.add((edge.from_node, edge.to_node))
        
        reconstructed_edges = set()
        for edge in reconstructed.get_edges():
            reconstructed_edges.add((edge.from_node, edge.to_node))
        
        assert original_edges == reconstructed_edges

    def test_adjacency_matrix_dtype(self):
        """Test: Adjazenzmatrix hat korrekten Datentyp"""
        import numpy as np
        graph = Graph()
        graph.add_node(id='n1')
        graph.add_node(id='n2')
        graph.add_edge('n1', 'n2')
        
        matrix, _ = graph.to_adjacency_matrix()
        
        assert matrix.dtype == np.int_
        assert np.all((matrix == 0) | (matrix == 1))  # Nur 0 und 1
