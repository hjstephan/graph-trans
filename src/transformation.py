"""
Transformation-Klasse zur Definition und Anwendung von Graphtransformationen.
"""

from typing import Optional
from graph import Graph


class TransformationError(Exception):
    """Exception für Fehler bei der Anwendung von Transformationen."""
    pass


class Transformation:
    """
    Repräsentiert eine Graphtransformation als Regel L → R.
    
    Die Transformation beschreibt, wie ein Graph (Zustand) in einen anderen
    überführt wird durch:
    - Rote Elemente: werden gelöscht
    - Grüne Elemente: werden hinzugefügt
    - Schwarze Elemente: bleiben unverändert (Kontext)
    """

    def __init__(self, name: str, left: Graph, right: Graph):
        """
        Initialisiert eine neue Transformation.
        
        Args:
            name: Name der Transformation
            left: Linke Seite der Regel (Vorbedingung)
            right: Rechte Seite der Regel (Nachbedingung)
        """
        self.name = name
        self.left = left
        self.right = right
        self._validate()

    def _validate(self):
        """
        Validiert die Transformation.
        
        Prüft, dass:
        - Schwarze Knoten in beiden Seiten vorhanden sind
        - Rote Knoten nur auf der linken Seite vorkommen
        - Grüne Knoten nur auf der rechten Seite vorkommen
        """
        left_black_ids = {n.id for n in self.left.get_nodes(color='black')}
        right_black_ids = {n.id for n in self.right.get_nodes(color='black')}
        
        # Schwarze Knoten müssen in beiden Graphen identisch sein
        if left_black_ids != right_black_ids:
            raise TransformationError(
                f"Schwarze Knoten müssen in beiden Seiten identisch sein. "
                f"Links: {left_black_ids}, Rechts: {right_black_ids}"
            )
        
        # Rote Elemente dürfen nur auf der linken Seite sein
        right_red_nodes = self.right.get_nodes(color='red')
        right_red_edges = self.right.get_edges(color='red')
        if right_red_nodes or right_red_edges:
            raise TransformationError(
                "Rote Elemente dürfen nur auf der linken Seite vorkommen"
            )
        
        # Grüne Elemente dürfen nur auf der rechten Seite sein
        left_green_nodes = self.left.get_nodes(color='green')
        left_green_edges = self.left.get_edges(color='green')
        if left_green_nodes or left_green_edges:
            raise TransformationError(
                "Grüne Elemente dürfen nur auf der rechten Seite vorkommen"
            )

    def apply(self, graph: Graph) -> Graph:
        """
        Wendet die Transformation auf einen Graphen an.
        
        Args:
            graph: Der Graph, auf den die Transformation angewendet wird
            
        Returns:
            Ein neuer Graph nach Anwendung der Transformation
            
        Raises:
            TransformationError: Wenn die Transformation nicht anwendbar ist
        """
        # Prüfe ob die linke Seite mit dem Eingabegraph übereinstimmt
        if not self._matches(graph):
            raise TransformationError(
                f"Die Transformation '{self.name}' kann nicht angewendet werden: "
                f"Linke Seite stimmt nicht mit dem Eingabegraph überein"
            )
        
        # Erstelle eine Kopie des Graphen
        result = graph.copy()
        
        # Schritt 1: Entferne rote Knoten und Kanten
        for node in self.left.get_nodes(color='red'):
            result.remove_node(node.id)
        
        for edge in self.left.get_edges(color='red'):
            result.remove_edge(edge.from_node, edge.to_node)
        
        # Schritt 2: Füge grüne Knoten und Kanten hinzu
        for node in self.right.get_nodes(color='green'):
            result.add_node(
                id=node.id,
                color='black',  # Im Ergebnisgraph sind neue Knoten schwarz
                **node.attributes
            )
        
        for edge in self.right.get_edges(color='green'):
            result.add_edge(
                from_node=edge.from_node,
                to_node=edge.to_node,
                color='black',  # Im Ergebnisgraph sind neue Kanten schwarz
                **edge.attributes
            )
        
        return result

    def _matches(self, graph: Graph) -> bool:
        """
        Prüft ob die linke Seite der Transformation mit dem Graphen übereinstimmt.
        
        Args:
            graph: Der zu prüfende Graph
            
        Returns:
            True wenn die Transformation anwendbar ist
        """
        # Prüfe schwarze Knoten
        for node in self.left.get_nodes(color='black'):
            graph_node = graph.get_node(node.id)
            if graph_node is None:
                return False
        
        # Prüfe rote Knoten (müssen im Graph existieren)
        for node in self.left.get_nodes(color='red'):
            if node.id not in graph:
                return False
        
        # Prüfe schwarze Kanten
        graph_edges = {(e.from_node, e.to_node) for e in graph.get_edges()}
        for edge in self.left.get_edges(color='black'):
            if (edge.from_node, edge.to_node) not in graph_edges:
                return False
        
        # Prüfe rote Kanten (müssen im Graph existieren)
        for edge in self.left.get_edges(color='red'):
            if (edge.from_node, edge.to_node) not in graph_edges:
                return False
        
        return True

    def __repr__(self) -> str:
        return f"Transformation(name='{self.name}')"


def apply_transformation(graph: Graph, transformation: Transformation) -> Graph:
    """
    Hilfsfunktion zum Anwenden einer Transformation auf einen Graphen.
    
    Args:
        graph: Der Graph, auf den die Transformation angewendet wird
        transformation: Die anzuwendende Transformation
        
    Returns:
        Ein neuer Graph nach Anwendung der Transformation
    """
    return transformation.apply(graph)
