"""
Tests für die Transformation-Klasse
"""

import pytest
from graph import Graph
from transformation import Transformation, TransformationError, apply_transformation


class TestTransformation:
    """Tests für die Transformation-Klasse"""

    def test_simple_transformation_creation(self):
        """Test: Einfache Transformation erstellen"""
        left = Graph(
            nodes=[{'id': 'n1', 'color': 'black'}],
            edges=[]
        )
        right = Graph(
            nodes=[{'id': 'n1', 'color': 'black'}],
            edges=[]
        )
        
        transformation = Transformation(
            name="IdentityTransformation",
            left=left,
            right=right
        )
        
        assert transformation.name == "IdentityTransformation"
        assert transformation.left == left
        assert transformation.right == right

    def test_transformation_with_creation(self):
        """Test: Transformation mit Knoten-Erzeugung"""
        left = Graph(
            nodes=[{'id': 'n1', 'color': 'black'}],
            edges=[]
        )
        right = Graph(
            nodes=[
                {'id': 'n1', 'color': 'black'},
                {'id': 'n2', 'color': 'green'}
            ],
            edges=[{'from': 'n1', 'to': 'n2', 'color': 'green'}]
        )
        
        transformation = Transformation(
            name="CreateNode",
            left=left,
            right=right
        )
        
        assert len(transformation.right.get_nodes(color='green')) == 1

    def test_transformation_with_deletion(self):
        """Test: Transformation mit Knoten-Löschung"""
        left = Graph(
            nodes=[
                {'id': 'n1', 'color': 'black'},
                {'id': 'n2', 'color': 'red'}
            ],
            edges=[{'from': 'n1', 'to': 'n2', 'color': 'red'}]
        )
        right = Graph(
            nodes=[{'id': 'n1', 'color': 'black'}],
            edges=[]
        )
        
        transformation = Transformation(
            name="DeleteNode",
            left=left,
            right=right
        )
        
        assert len(transformation.left.get_nodes(color='red')) == 1

    def test_invalid_transformation_black_nodes_mismatch(self):
        """Test: Ungültige Transformation - schwarze Knoten stimmen nicht überein"""
        left = Graph(
            nodes=[{'id': 'n1', 'color': 'black'}],
            edges=[]
        )
        right = Graph(
            nodes=[{'id': 'n2', 'color': 'black'}],
            edges=[]
        )
        
        with pytest.raises(TransformationError, match="schwarze Knoten"):
            Transformation(name="Invalid", left=left, right=right)

    def test_invalid_transformation_red_on_right(self):
        """Test: Ungültige Transformation - rote Elemente auf rechter Seite"""
        left = Graph(
            nodes=[{'id': 'n1', 'color': 'black'}],
            edges=[]
        )
        right = Graph(
            nodes=[
                {'id': 'n1', 'color': 'black'},
                {'id': 'n2', 'color': 'red'}
            ],
            edges=[]
        )
        
        with pytest.raises(TransformationError, match="Rote Elemente"):
            Transformation(name="Invalid", left=left, right=right)

    def test_invalid_transformation_green_on_left(self):
        """Test: Ungültige Transformation - grüne Elemente auf linker Seite"""
        left = Graph(
            nodes=[
                {'id': 'n1', 'color': 'black'},
                {'id': 'n2', 'color': 'green'}
            ],
            edges=[]
        )
        right = Graph(
            nodes=[{'id': 'n1', 'color': 'black'}],
            edges=[]
        )
        
        with pytest.raises(TransformationError, match="Grüne Elemente"):
            Transformation(name="Invalid", left=left, right=right)

    def test_apply_identity_transformation(self):
        """Test: Identitätstransformation anwenden"""
        graph = Graph(
            nodes=[{'id': 'n1', 'color': 'black'}],
            edges=[]
        )
        
        left = Graph(
            nodes=[{'id': 'n1', 'color': 'black'}],
            edges=[]
        )
        right = Graph(
            nodes=[{'id': 'n1', 'color': 'black'}],
            edges=[]
        )
        
        transformation = Transformation(name="Identity", left=left, right=right)
        result = transformation.apply(graph)
        
        assert len(result) == 1
        assert 'n1' in result

    def test_apply_node_creation(self):
        """Test: Knoten-Erzeugung anwenden"""
        graph = Graph(
            nodes=[{'id': 'n1', 'color': 'black'}],
            edges=[]
        )
        
        left = Graph(
            nodes=[{'id': 'n1', 'color': 'black'}],
            edges=[]
        )
        right = Graph(
            nodes=[
                {'id': 'n1', 'color': 'black'},
                {'id': 'n2', 'color': 'green'}
            ],
            edges=[{'from': 'n1', 'to': 'n2', 'color': 'green'}]
        )
        
        transformation = Transformation(name="CreateNode", left=left, right=right)
        result = transformation.apply(graph)
        
        assert len(result) == 2
        assert 'n1' in result
        assert 'n2' in result
        assert len(result.get_edges()) == 1

    def test_apply_node_deletion(self):
        """Test: Knoten-Löschung anwenden"""
        graph = Graph(
            nodes=[
                {'id': 'n1', 'color': 'black'},
                {'id': 'n2', 'color': 'black'}
            ],
            edges=[{'from': 'n1', 'to': 'n2', 'color': 'black'}]
        )
        
        left = Graph(
            nodes=[
                {'id': 'n1', 'color': 'black'},
                {'id': 'n2', 'color': 'red'}
            ],
            edges=[{'from': 'n1', 'to': 'n2', 'color': 'red'}]
        )
        right = Graph(
            nodes=[{'id': 'n1', 'color': 'black'}],
            edges=[]
        )
        
        transformation = Transformation(name="DeleteNode", left=left, right=right)
        result = transformation.apply(graph)
        
        assert len(result) == 1
        assert 'n1' in result
        assert 'n2' not in result
        assert len(result.get_edges()) == 0

    def test_apply_transformation_not_applicable(self):
        """Test: Transformation ist nicht anwendbar"""
        graph = Graph(
            nodes=[{'id': 'n1', 'color': 'black'}],
            edges=[]
        )
        
        left = Graph(
            nodes=[{'id': 'n2', 'color': 'black'}],
            edges=[]
        )
        right = Graph(
            nodes=[{'id': 'n2', 'color': 'black'}],
            edges=[]
        )
        
        transformation = Transformation(name="NotApplicable", left=left, right=right)
        
        with pytest.raises(TransformationError, match="kann nicht angewendet werden"):
            transformation.apply(graph)

    def test_apply_transformation_helper_function(self):
        """Test: apply_transformation Hilfsfunktion"""
        graph = Graph(
            nodes=[{'id': 'n1', 'color': 'black'}],
            edges=[]
        )
        
        left = Graph(
            nodes=[{'id': 'n1', 'color': 'black'}],
            edges=[]
        )
        right = Graph(
            nodes=[
                {'id': 'n1', 'color': 'black'},
                {'id': 'n2', 'color': 'green'}
            ],
            edges=[]
        )
        
        transformation = Transformation(name="CreateNode", left=left, right=right)
        result = apply_transformation(graph, transformation)
        
        assert len(result) == 2
        assert 'n2' in result

    def test_transformation_preserves_original_graph(self):
        """Test: Transformation verändert Original-Graph nicht"""
        graph = Graph(
            nodes=[{'id': 'n1', 'color': 'black'}],
            edges=[]
        )
        
        left = Graph(
            nodes=[{'id': 'n1', 'color': 'black'}],
            edges=[]
        )
        right = Graph(
            nodes=[
                {'id': 'n1', 'color': 'black'},
                {'id': 'n2', 'color': 'green'}
            ],
            edges=[]
        )
        
        transformation = Transformation(name="CreateNode", left=left, right=right)
        result = transformation.apply(graph)
        
        # Original sollte unverändert sein
        assert len(graph) == 1
        assert 'n2' not in graph
        
        # Ergebnis sollte neuen Knoten haben
        assert len(result) == 2
        assert 'n2' in result

    def test_complex_transformation(self):
        """Test: Komplexe Transformation mit mehreren Änderungen"""
        # Ausgangsgraph
        graph = Graph(
            nodes=[
                {'id': 'n1', 'color': 'black'},
                {'id': 'n2', 'color': 'black'},
                {'id': 'n3', 'color': 'black'}
            ],
            edges=[
                {'from': 'n1', 'to': 'n2', 'color': 'black'},
                {'from': 'n2', 'to': 'n3', 'color': 'black'}
            ]
        )
        
        # Transformation: Lösche n2, füge n4 hinzu
        left = Graph(
            nodes=[
                {'id': 'n1', 'color': 'black'},
                {'id': 'n2', 'color': 'red'},
                {'id': 'n3', 'color': 'black'}
            ],
            edges=[
                {'from': 'n1', 'to': 'n2', 'color': 'red'},
                {'from': 'n2', 'to': 'n3', 'color': 'red'}
            ]
        )
        right = Graph(
            nodes=[
                {'id': 'n1', 'color': 'black'},
                {'id': 'n3', 'color': 'black'},
                {'id': 'n4', 'color': 'green'}
            ],
            edges=[
                {'from': 'n1', 'to': 'n4', 'color': 'green'},
                {'from': 'n4', 'to': 'n3', 'color': 'green'}
            ]
        )
        
        transformation = Transformation(name="ComplexTransformation", left=left, right=right)
        result = transformation.apply(graph)
        
        assert len(result) == 3
        assert 'n1' in result
        assert 'n2' not in result
        assert 'n3' in result
        assert 'n4' in result
        assert len(result.get_edges()) == 2

    def test_repr(self):
        """Test: String-Repräsentation"""
        left = Graph(nodes=[{'id': 'n1', 'color': 'black'}], edges=[])
        right = Graph(nodes=[{'id': 'n1', 'color': 'black'}], edges=[])
        
        transformation = Transformation(name="TestTransformation", left=left, right=right)
        repr_str = repr(transformation)
        
        assert 'Transformation' in repr_str
        assert 'TestTransformation' in repr_str
