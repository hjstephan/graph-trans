"""
Graph-Klasse zur Repräsentation von Graphen mit farbcodierten Knoten und Kanten.
"""

from typing import List, Dict, Set, Optional
from dataclasses import dataclass, field


@dataclass
class Node:
    """Repräsentiert einen Knoten im Graph."""
    id: str
    color: str = 'black'  # 'black', 'red', 'green'
    attributes: Dict = field(default_factory=dict)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return self.id == other.id


@dataclass
class Edge:
    """Repräsentiert eine Kante im Graph."""
    from_node: str
    to_node: str
    color: str = 'black'  # 'black', 'red', 'green'
    attributes: Dict = field(default_factory=dict)

    def __hash__(self):
        return hash((self.from_node, self.to_node))

    def __eq__(self, other):
        if not isinstance(other, Edge):
            return False
        return (self.from_node == other.from_node and 
                self.to_node == other.to_node)


class Graph:
    """
    Repräsentiert einen Graphen mit Knoten und Kanten.
    
    Unterstützt farbcodierte Elemente für Graphtransformationen:
    - black: Kontextelemente (bleiben unverändert)
    - red: Zu löschende Elemente
    - green: Neu zu erzeugende Elemente
    """

    def __init__(self, nodes: Optional[List[Dict]] = None, 
                 edges: Optional[List[Dict]] = None):
        """
        Initialisiert einen neuen Graphen.
        
        Args:
            nodes: Liste von Knoten-Dictionaries mit 'id', 'color', etc.
            edges: Liste von Kanten-Dictionaries mit 'from', 'to', 'color', etc.
        """
        self._nodes: Dict[str, Node] = {}
        self._edges: Set[Edge] = set()

        if nodes:
            for node_dict in nodes:
                self.add_node(**node_dict)
        
        if edges:
            for edge_dict in edges:
                from_node = edge_dict.pop('from', None) or edge_dict.pop('from_node')
                to_node = edge_dict.pop('to', None) or edge_dict.pop('to_node')
                self.add_edge(from_node, to_node, **edge_dict)

    def add_node(self, id: str, color: str = 'black', **attributes) -> Node:
        """Fügt einen Knoten zum Graphen hinzu."""
        node = Node(id=id, color=color, attributes=attributes)
        self._nodes[id] = node
        return node

    def add_edge(self, from_node: str, to_node: str, 
                 color: str = 'black', **attributes) -> Edge:
        """Fügt eine Kante zum Graphen hinzu."""
        if from_node not in self._nodes:
            raise ValueError(f"Knoten '{from_node}' existiert nicht")
        if to_node not in self._nodes:
            raise ValueError(f"Knoten '{to_node}' existiert nicht")
        
        edge = Edge(from_node=from_node, to_node=to_node, 
                   color=color, attributes=attributes)
        self._edges.add(edge)
        return edge

    def get_node(self, node_id: str) -> Optional[Node]:
        """Gibt einen Knoten anhand seiner ID zurück."""
        return self._nodes.get(node_id)

    def get_nodes(self, color: Optional[str] = None) -> List[Node]:
        """
        Gibt alle Knoten zurück, optional gefiltert nach Farbe.
        
        Args:
            color: Optional - filtert Knoten nach Farbe ('black', 'red', 'green')
        """
        nodes = list(self._nodes.values())
        if color:
            nodes = [n for n in nodes if n.color == color]
        return nodes

    def get_edges(self, color: Optional[str] = None) -> List[Edge]:
        """
        Gibt alle Kanten zurück, optional gefiltert nach Farbe.
        
        Args:
            color: Optional - filtert Kanten nach Farbe ('black', 'red', 'green')
        """
        edges = list(self._edges)
        if color:
            edges = [e for e in edges if e.color == color]
        return edges

    def remove_node(self, node_id: str) -> bool:
        """
        Entfernt einen Knoten und alle zugehörigen Kanten.
        
        Returns:
            True wenn der Knoten entfernt wurde, False wenn er nicht existierte
        """
        if node_id not in self._nodes:
            return False
        
        # Entferne alle Kanten, die diesen Knoten betreffen
        self._edges = {e for e in self._edges 
                      if e.from_node != node_id and e.to_node != node_id}
        
        del self._nodes[node_id]
        return True

    def remove_edge(self, from_node: str, to_node: str) -> bool:
        """
        Entfernt eine Kante.
        
        Returns:
            True wenn die Kante entfernt wurde, False wenn sie nicht existierte
        """
        edge_to_remove = Edge(from_node=from_node, to_node=to_node)
        if edge_to_remove in self._edges:
            self._edges.remove(edge_to_remove)
            return True
        return False

    def copy(self) -> 'Graph':
        """Erstellt eine tiefe Kopie des Graphen."""
        new_graph = Graph()
        
        for node in self._nodes.values():
            new_graph.add_node(
                id=node.id,
                color=node.color,
                **node.attributes.copy()
            )
        
        for edge in self._edges:
            new_graph.add_edge(
                from_node=edge.from_node,
                to_node=edge.to_node,
                color=edge.color,
                **edge.attributes.copy()
            )
        
        return new_graph

    def __len__(self) -> int:
        """Gibt die Anzahl der Knoten zurück."""
        return len(self._nodes)

    def __contains__(self, node_id: str) -> bool:
        """Prüft ob ein Knoten mit der gegebenen ID existiert."""
        return node_id in self._nodes

    def __repr__(self) -> str:
        return (f"Graph(nodes={len(self._nodes)}, "
                f"edges={len(self._edges)})")
