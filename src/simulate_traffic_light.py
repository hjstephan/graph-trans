"""
Simulation einer Ampelkreuzung mit Graphtransformationen.

Dieses Beispiel zeigt, wie eine Ampelkreuzung verschiedene Zustände
durchläuft: Grün → Gelb → Rot → Rot-Gelb → Grün
"""

import time
from graph import Graph
from transformation import Transformation


def print_separator():
    """Druckt eine Trennlinie."""
    print("\n" + "=" * 70 + "\n")


def print_graph_state(graph: Graph, title: str):
    """Zeigt den aktuellen Zustand des Graphen an."""
    print(f"📊 {title}")
    print("-" * 70)
    
    nodes = graph.get_nodes()
    edges = graph.get_edges()
    
    print(f"Knoten ({len(nodes)}):")
    for node in nodes:
        attrs = ", ".join(f"{k}={v}" for k, v in node.attributes.items())
        color_symbol = {"black": "⚫", "red": "🔴", "green": "🟢"}.get(node.color, "⚪")
        print(f"  {color_symbol} {node.id}" + (f" [{attrs}]" if attrs else ""))
    
    print(f"\nKanten ({len(edges)}):")
    for edge in edges:
        attrs = ", ".join(f"{k}={v}" for k, v in edge.attributes.items())
        color_symbol = {"black": "⚫", "red": "🔴", "green": "🟢"}.get(edge.color, "⚪")
        print(f"  {color_symbol} {edge.from_node} → {edge.to_node}" + (f" [{attrs}]" if attrs else ""))
    
    print()


def create_initial_state() -> Graph:
    """Erstellt den Anfangszustand: Ampel auf Grün."""
    graph = Graph()
    
    # Infrastruktur
    graph.add_node('ampel', type='traffic_light')
    graph.add_node('nord', type='direction')
    graph.add_node('süd', type='direction')
    
    # Aktueller Zustand: Grün
    graph.add_node('grün', type='signal', active='true')
    
    # Verbindungen
    graph.add_edge('ampel', 'grün', type='shows')
    graph.add_edge('grün', 'nord', type='allows')
    graph.add_edge('grün', 'süd', type='allows')
    
    return graph


def create_green_to_yellow() -> Transformation:
    """Transformation: Grün → Gelb."""
    
    # Linke Seite: Grün ist aktiv
    left = Graph()
    left.add_node('ampel', color='black', type='traffic_light')
    left.add_node('grün', color='red', type='signal', active='true')
    left.add_node('nord', color='black', type='direction')
    left.add_node('süd', color='black', type='direction')
    
    left.add_edge('ampel', 'grün', color='red', type='shows')
    left.add_edge('grün', 'nord', color='red', type='allows')
    left.add_edge('grün', 'süd', color='red', type='allows')
    
    # Rechte Seite: Gelb ist aktiv
    right = Graph()
    right.add_node('ampel', color='black', type='traffic_light')
    right.add_node('gelb', color='green', type='signal', active='true')
    right.add_node('nord', color='black', type='direction')
    right.add_node('süd', color='black', type='direction')
    
    right.add_edge('ampel', 'gelb', color='green', type='shows')
    right.add_edge('gelb', 'nord', color='green', type='warns')
    right.add_edge('gelb', 'süd', color='green', type='warns')
    
    return Transformation('Grün → Gelb', left, right)


def create_yellow_to_red() -> Transformation:
    """Transformation: Gelb → Rot."""
    
    # Linke Seite: Gelb ist aktiv
    left = Graph()
    left.add_node('ampel', color='black', type='traffic_light')
    left.add_node('gelb', color='red', type='signal', active='true')
    left.add_node('nord', color='black', type='direction')
    left.add_node('süd', color='black', type='direction')
    
    left.add_edge('ampel', 'gelb', color='red', type='shows')
    left.add_edge('gelb', 'nord', color='red', type='warns')
    left.add_edge('gelb', 'süd', color='red', type='warns')
    
    # Rechte Seite: Rot ist aktiv
    right = Graph()
    right.add_node('ampel', color='black', type='traffic_light')
    right.add_node('rot', color='green', type='signal', active='true')
    right.add_node('nord', color='black', type='direction')
    right.add_node('süd', color='black', type='direction')
    
    right.add_edge('ampel', 'rot', color='green', type='shows')
    right.add_edge('rot', 'nord', color='green', type='blocks')
    right.add_edge('rot', 'süd', color='green', type='blocks')
    
    return Transformation('Gelb → Rot', left, right)


def create_red_to_red_yellow() -> Transformation:
    """Transformation: Rot → Rot-Gelb."""
    
    # Linke Seite: Rot ist aktiv
    left = Graph()
    left.add_node('ampel', color='black', type='traffic_light')
    left.add_node('rot', color='red', type='signal', active='true')
    left.add_node('nord', color='black', type='direction')
    left.add_node('süd', color='black', type='direction')
    
    left.add_edge('ampel', 'rot', color='red', type='shows')
    left.add_edge('rot', 'nord', color='red', type='blocks')
    left.add_edge('rot', 'süd', color='red', type='blocks')
    
    # Rechte Seite: Rot-Gelb ist aktiv
    right = Graph()
    right.add_node('ampel', color='black', type='traffic_light')
    right.add_node('rot_gelb', color='green', type='signal', active='true')
    right.add_node('nord', color='black', type='direction')
    right.add_node('süd', color='black', type='direction')
    
    right.add_edge('ampel', 'rot_gelb', color='green', type='shows')
    right.add_edge('rot_gelb', 'nord', color='green', type='prepares')
    right.add_edge('rot_gelb', 'süd', color='green', type='prepares')
    
    return Transformation('Rot → Rot-Gelb', left, right)


def create_red_yellow_to_green() -> Transformation:
    """Transformation: Rot-Gelb → Grün."""
    
    # Linke Seite: Rot-Gelb ist aktiv
    left = Graph()
    left.add_node('ampel', color='black', type='traffic_light')
    left.add_node('rot_gelb', color='red', type='signal', active='true')
    left.add_node('nord', color='black', type='direction')
    left.add_node('süd', color='black', type='direction')
    
    left.add_edge('ampel', 'rot_gelb', color='red', type='shows')
    left.add_edge('rot_gelb', 'nord', color='red', type='prepares')
    left.add_edge('rot_gelb', 'süd', color='red', type='prepares')
    
    # Rechte Seite: Grün ist aktiv
    right = Graph()
    right.add_node('ampel', color='black', type='traffic_light')
    right.add_node('grün', color='green', type='signal', active='true')
    right.add_node('nord', color='black', type='direction')
    right.add_node('süd', color='black', type='direction')
    
    right.add_edge('ampel', 'grün', color='green', type='shows')
    right.add_edge('grün', 'nord', color='green', type='allows')
    right.add_edge('grün', 'süd', color='green', type='allows')
    
    return Transformation('Rot-Gelb → Grün', left, right)


def run_simulation(cycles: int = 2, delay: float = 2.0):
    """
    Führt die Ampel-Simulation aus.
    
    Args:
        cycles: Anzahl der vollständigen Ampelzyklen
        delay: Verzögerung in Sekunden zwischen Transformationen
    """
    print_separator()
    print("🚦 AMPELKREUZUNG SIMULATION")
    print_separator()
    
    # Erstelle Anfangszustand
    current_state = create_initial_state()
    print_graph_state(current_state, "ANFANGSZUSTAND: Grün")
    
    # Erstelle Transformationen
    transformations = [
        create_green_to_yellow(),
        create_yellow_to_red(),
        create_red_to_red_yellow(),
        create_red_yellow_to_green()
    ]
    
    # Führe Zyklen aus
    for cycle in range(cycles):
        print_separator()
        print(f"🔄 ZYKLUS {cycle + 1}")
        print_separator()
        
        for transformation in transformations:
            time.sleep(delay)
            
            print(f"\n⚙️  Wende Transformation an: {transformation.name}")
            print("-" * 70)
            
            try:
                current_state = transformation.apply(current_state)
                print("✅ Transformation erfolgreich angewendet\n")
                print_graph_state(current_state, f"NEUER ZUSTAND")
                
            except Exception as e:
                print(f"❌ Fehler: {e}")
                return
    
    print_separator()
    print("✨ Simulation abgeschlossen!")
    print_separator()


if __name__ == "__main__":
    # Starte die Simulation
    # 2 vollständige Zyklen mit 2 Sekunden Verzögerung zwischen Transformationen
    run_simulation(cycles=2, delay=2.0)
