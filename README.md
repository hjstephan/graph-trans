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
from graph_trans import Graph, Transformation, apply_transformation

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

## Anwendungsfälle

- UML-Zustandsdiagramme
- Objektdiagramm-Transformationen
- Model-Driven Engineering (MDE)
- Refactoring-Spezifikationen
- Petri-Netze
- Formale Verifikation von Systemübergängen

## Kontakt

Bei Fragen oder Anregungen öffnen Sie bitte ein Issue im Repository.