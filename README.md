# graph-trans

Ein Framework zur Modellierung und Ausführung von Graphtransformationen für Systemzustände in der Softwaremodellierung.

## Überblick

`graph-trans` implementiert Graphtransformationen zur Modellierung von Zustandsübergängen in Softwaresystemen. Dabei werden Systemzustände als Graphen dargestellt, und Zustandsübergänge durch farbcodierte Transformationsregeln beschrieben.

## Konzept

Graphtransformationen modellieren Zustandsübergänge durch drei Kategorien von Graphelementen:

- **Rote Knoten/Kanten**: Elemente, die im nächsten Zustand entfernt werden (Löschung)
- **Grüne Knoten/Kanten**: Elemente, die im nächsten Zustand neu hinzugefügt werden (Erzeugung)
- **Schwarze Knoten/Kanten**: Elemente, die unverändert bleiben (Kontext)

Eine Transformation beschreibt den Übergang von einem Systemzustand zu einem anderen durch Anwendung dieser Regeln.

## Features

- Modellierung von Systemzuständen als Graphen
- Definition von Transformationsregeln mit Lösch-, Erzeugungs- und Kontextelementen
- Anwendung von Transformationen auf Graphzustände
- Visualisierung von Zustandsübergängen
- Validierung von Transformationsregeln
- **Neu in v1.1.0**: Beispiel-Simulation einer Ampelkreuzung

## Installation

```bash
# Repository klonen
git clone https://github.com/username/graph-trans.git
cd graph-trans

# Virtuelle Umgebung erstellen (empfohlen)
python -m venv venv
source venv/bin/activate  # Unter Windows: venv\Scripts\activate

# Abhängigkeiten installieren
pip install -r requirements.txt
```

## Verwendung

```python
from graph import Graph
from transformation import Transformation, apply_transformation

# Beispiel: Definition einer Transformation
transformation = Transformation(
    name="CreateNode",
    left=Graph(
        nodes=[{'id': 'n1', 'color': 'black'}],
        edges=[]
    ),
    right=Graph(
        nodes=[
            {'id': 'n1', 'color': 'black'},
            {'id': 'n2', 'color': 'green'}
        ],
        edges=[
            {'from': 'n1', 'to': 'n2', 'color': 'green'}
        ]
    )
)

# Transformation anwenden
new_state = apply_transformation(current_state, transformation)
```

## Ampelkreuzung Simulation (v1.1.0)

Das Projekt enthält jetzt eine vollständige Beispiel-Simulation einer Ampelkreuzung, die verschiedene Zustände durchläuft:

```bash
python simulate_traffic_light.py
```

Die Simulation demonstriert:
- **Zustandsübergänge**: Grün → Gelb → Rot → Rot-Gelb → Grün
- **Zeitverzögerungen**: Visualisierung mit konfigurierbaren Verzögerungen zwischen Transformationen
- **Graphtransformationen**: Praktische Anwendung des Frameworks auf ein reales Szenario

Die Ampel-Simulation zeigt, wie Systemzustände als Graphen modelliert und durch Transformationen verändert werden können. Jeder Ampelzustand wird als Graph dargestellt, wobei Knoten die Infrastruktur (Ampel, Richtungen) und den aktuellen Signalzustand (grün, gelb, rot, rot-gelb) repräsentieren.

## Systemanalyse

Analysiere mit Hilfe des [Subgraph Algorithmus](https://github.com/hjstephan/subgraph) die Stabilität oder die Ruhelage eines Systems, welches in jedem globalen Zustand durch einen Graphen beschrieben wird. Was wurde spezifiziert, was modelliert und wie verhält sich das System in der Realität wirklich? Damit lassen sich Fehler im System finden und Aussagen treffen zur Performance und Sicherheit des Systems.

## Subgraph Algorithmus (v2.0.0)

Ergänzt sind Schnittstellen-Methoden zur Nutzung des [Subgraph Algorithmus](https://github.com/hjstephan/subgraph).

```python
# In deinem Code kannst du jetzt:
from graph import Graph
from subgraph import Subgraph

# Graph erstellen
g1 = Graph(...)

# In Matrix konvertieren
matrix, node_mapping = g1.to_adjacency_matrix()

# Subgraph-Algorithmus anwenden
algo = Subgraph()
result = algo.compare_graphs(matrix1, matrix2)

# Zurück in Graph konvertieren
g_result = Graph.from_adjacency_matrix(matrix, node_ids=list(node_mapping.keys()))
```

## Anwendungsfälle

- UML-Zustandsdiagramme
- Objektdiagramm-Transformationen
- Model-Driven Engineering (MDE)
- Refactoring-Spezifikationen
- Petri-Netze
- Formale Verifikation von Systemübergängen
- **Ampelsteuerungen und Verkehrssysteme** (siehe `simulate_traffic_light.py`)

## Systemstabilitätsanalyse (v2.1.0)

Ergänzt wurde die vollständige Stabilitätsanalyse von Systemtransformationen mittels Subgraph-Algorithmus:

```bash
python system_stability_analysis.py
```

Die Stabilitätsanalyse:
- **Speichert vollständigen Transformationsverlauf**: Alle Systemzustände werden als Graphen mit Adjazenzmatrizen gespeichert
- **Paarweiser Subgraph-Vergleich**: Jeder Zustand wird mit jedem anderen verglichen
- **Längste Subgraph-Sequenzen**: Identifiziert monotone Erweiterungsketten im Systemverlauf
- **Stabile Zustände (Ruhelagen)**: Erkennt Zustände ohne Veränderung
- **Zyklen-Detektion**: Findet wiederkehrende Systemzustände

**Hinweis zum Ampelsystem**: Die Analyse ist für das Ampelbeispiel trivial, da es nur wenige distinkte Systemzustände gibt (Grün, Gelb, Rot, Rot-Gelb), die zyklisch durchlaufen werden. Es handelt sich um ein einfaches deterministisches System ohne gemeinsame Subgraph-Strukturen zwischen verschiedenen Ampelphasen. Für komplexere Systeme mit vielen Komponenten und Interaktionen (z.B. verteilte Systeme, Netzwerkprotokolle, Workflow-Engines) liefert die Analyse wesentlich aussagekräftigere Ergebnisse zur Systemstabilität und potentiellen Fehlerquellen.

### Eigene Systeme analysieren

```python
from system_stability_analysis import SystemSimulation
from graph import Graph
from transformation import Transformation

# Erstelle dein System
initial_state = Graph(...)
transformations = [...]

# Initialisiere Simulation
sim = SystemSimulation()
for t in transformations:
    sim.add_transformation(t)

# Führe Simulation aus
states = sim.run(initial_state, steps=100)

# Analysiere Stabilität
analysis = sim.analyze_stability()
sim.print_analysis(analysis)
```

## Markov-Ketten Analyse (v3.0.0)

### Neue Funktionalität

Version 3.0.0 erweitert den Subgraph-Algorithmus um eine vollständige Markov-Ketten Implementierung mit Stabilitätsanalyse. Diese Erweiterung ermöglicht die Analyse dynamischer Systeme durch die Evolution ihrer Übergangsmatrizen.

### Markov-Ketten Grundlagen

Eine **Markov-Kette** ist ein stochastischer Prozess, bei dem die Wahrscheinlichkeit des nächsten Zustands nur vom aktuellen Zustand abhängt (Markov-Eigenschaft).

**Repräsentation**:
- **Zustände**: Knoten im Graph
- **Übergänge**: Gerichtete Kanten mit Übergangswahrscheinlichkeiten
- **Übergangsmatrix** $P$: $P_{ij}$ = Wahrscheinlichkeit von Zustand $i$ nach $j$

**Eigenschaften**:
- **Irreduzibel**: Alle Zustände sind voneinander erreichbar (über beliebige Pfade)
- **Aperiodisch**: Alle Zustände haben Periode 1 (keine zyklischen Strukturen)
- **Ergodisch**: Irreduzibel und aperiodisch
- **Stationäre Verteilung** $\pi$: $\pi P = \pi$ (existiert für ergodische Ketten)

### Implementierung

#### MarkovChain Klasse

Erweitert die `Graph`-Klasse um Markov-Ketten spezifische Funktionalität:

```python
from markov_chain import MarkovChain
import numpy as np

# Definiere Zustände
states = ['sonnig', 'bewölkt', 'regnerisch']

# Definiere Übergangsmatrix
P = np.array([
    [0.7, 0.2, 0.1],  # von sonnig
    [0.3, 0.4, 0.3],  # von bewölkt
    [0.2, 0.3, 0.5]   # von regnerisch
])

# Erstelle Markov-Kette
mc = MarkovChain(states=states, transition_matrix=P)

# Überprüfe Eigenschaften
print(f"Irreduzibel: {mc.is_irreducible()}")
print(f"Aperiodisch: {mc.is_aperiodic()}")
print(f"Ergodisch: {mc.is_ergodic()}")

# Berechne stationäre Verteilung
if mc.is_ergodic():
    pi = mc.compute_stationary_distribution()
    print(f"Stationäre Verteilung: {pi}")
```

## Beispiele

Das Repository enthält folgende Beispiele:

- **`simulate_traffic_light.py`**: Vollständige Simulation einer Ampelkreuzung mit Zustandsübergängen
- **`system_stability_analysis.py`**: Stabilitätsanalyse mit Subgraph-Algorithmus (demonstriert am Ampelsystem)
- **`marcov_stability_analysis.py`**: Stabilitätsanalyse für Markov-Ketten

## Kontakt

Bei Fragen oder Anregungen öffnen Sie bitte ein Issue im Repository.
