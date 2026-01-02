# HeteroData Visualization Redesign Plan

## Problem Statement

The current D3 visualization is **cluttered and incomplete**:
- Too many edges create visual noise
- Ball nodes (N=50-100) dominate the 21 context nodes
- Shows structure but not content (features, indices)
- No clear data flow understanding

## Cognitive Design Principles Applied

Based on cognitive design research, we should apply:

1. **Progressive Disclosure** (Shneiderman's Mantra): "Overview first, zoom and filter, then details on demand"
2. **Chunking**: Limit to 4±1 concurrent visual groups (fits working memory)
3. **Visual Hierarchy**: Size, contrast, position guide attention
4. **Small Multiples**: Repeated structure enables comparison
5. **Recognition over Recall**: Show state, don't require memory

---

## Brainstormed Visualization Approaches

### Approach 1: Multi-Panel Dashboard

**Concept**: Separate concerns into distinct panels, each focused on one aspect.

```
┌─────────────────────────────────────────────────────────────┐
│  PANEL 1: Layer Overview (always visible)                   │
│  ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐ │
│  │Global│ → │State│ → │Actor│ → │Dynam│ → │Ball │ → │Query│ │
│  │(3)   │   │(5)  │   │(7)  │   │(4)  │   │(N)  │   │(1)  │ │
│  └─────┘   └─────┘   └─────┘   └─────┘   └─────┘   └─────┘ │
│  Click any layer to expand details below                    │
├─────────────────────────────────────────────────────────────┤
│  PANEL 2: Selected Layer Detail                             │
│  (Shows nodes, features, intra-layer edges)                 │
├─────────────────────────────────────────────────────────────┤
│  PANEL 3: Edge Explorer (filterable)                        │
│  [ ] Hierarchical  [ ] Temporal  [ ] Cross-domain  [ ] Query│
└─────────────────────────────────────────────────────────────┘
```

**Pros**: Reduces clutter, progressive disclosure, each panel fits working memory
**Cons**: Loses global picture, requires interaction to see connections

---

### Approach 2: Semantic Zoom

**Concept**: Different detail levels at different zoom levels.

| Zoom Level | What's Shown |
|------------|--------------|
| **Far (Overview)** | 6 layer boxes with node counts, aggregate edge counts |
| **Medium** | Individual nodes visible, major edge types shown |
| **Close** | Node features, edge attributes, indices visible |

**Pros**: Natural interaction (zoom = more detail), maintains context
**Cons**: Complex to implement, may lose orientation when zoomed

---

### Approach 3: Separate Ball Timeline

**Concept**: Don't mix ball nodes with context graph. Show them separately.

```
┌─────────────────────────────────────────────────────────────┐
│  CONTEXT GRAPH (21 nodes)                                   │
│  ┌─────────────────────────────────────────────────────────┐│
│  │  Global → State → Actor → Dynamics → Query              ││
│  │  (clean, focused, shows hierarchical structure)         ││
│  └─────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────┤
│  BALL TIMELINE (N nodes)                                    │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ Ball 0 ── Ball 1 ── Ball 2 ── ... ── Ball N             ││
│  │ [Bu→Ro]  [Bu→Ro]   [Bu→KL]                              ││
│  │                                                          ││
│  │ Temporal edges shown as arcs above/below timeline        ││
│  │ Cross-domain connections shown when ball is selected     ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

**Pros**: Each part readable, respects different nature of ball nodes
**Cons**: Cross-domain edges harder to show (need interaction)

---

### Approach 4: Edge Matrix View

**Concept**: Use adjacency matrix for edges instead of drawn lines.

```
For temporal edges between balls:
         Ball 0  Ball 1  Ball 2  Ball 3
Ball 0     -      RP      RP      MP
Ball 1     -       -      RP      RP
Ball 2     -       -       -      RP
Ball 3     -       -       -       -

Legend: RP=recent_precedes, MP=medium_precedes, SB=same_bowler, etc.
```

**Pros**: No crossed lines, scales to many edges, compact
**Cons**: Less intuitive than visual connections, harder to trace paths

---

### Approach 5: Tabbed Documentation Pages

**Concept**: Not a single visualization but multiple focused pages.

| Tab | Content |
|-----|---------|
| **Overview** | High-level architecture diagram (your LaTeX diagram) |
| **Nodes** | Table/cards showing each node type with features |
| **Edges** | Interactive edge explorer (select type, see connections) |
| **Example** | One complete sample with real data walked through |
| **Data Flow** | Animated/stepped diagram showing message passing |

**Pros**: Complete documentation, fits any level of detail
**Cons**: Not a single visualization, requires navigation

---

### Approach 6: Layered Toggle System

**Concept**: Start minimal, add layers via toggles.

```
[x] Context Nodes (21)
[ ] Ball Nodes (50)
[ ] Hierarchical Edges
[ ] Temporal Edges
    [ ] recent_precedes
    [ ] same_bowler
    [ ] same_batsman
    [ ] same_matchup
    [ ] same_over
[ ] Cross-domain Edges
[ ] Query Edges
[ ] Show Features
[ ] Show Indices
```

**Pros**: User controls complexity, can build understanding incrementally
**Cons**: Still cluttered when all enabled, requires understanding to toggle

---

## Recommended Hybrid Approach

Combine the best ideas:

### 1. **Overview Panel** (always visible)
- Layer boxes with node counts
- Click to select layer for detail

### 2. **Context Graph View** (Panel 2)
- Shows 21 context nodes + query
- Clean hierarchical layout
- Intra-layer edges visible
- Hover for node features

### 3. **Ball Timeline View** (Panel 3)
- Horizontal timeline of balls
- Color-coded by bowler or batsman
- Temporal edges as arcs
- Click ball to see cross-domain connections

### 4. **Edge Filter Sidebar**
- Toggle edge types on/off
- Shows count of each type
- Highlights selected type

### 5. **Detail Tooltip/Panel**
- On hover: node name, type, feature dimensions
- On click: full feature vector, edge list

---

## User Requirements (Confirmed)

1. **Primary use case**: Learning architecture AND debugging samples
2. **Interactivity level**: Exploratory tool
3. **Feature display**: Both dimensions AND actual values (shown in sidebar on click)
4. **Ball nodes**: Show ALL balls from one complete game as reference

---

## Implementation Plan

### Phase 1: Layout Structure

Create the main HTML/CSS layout with 4 regions:

```
┌─────────────────────────────────────────────────────────────────────────┐
│ HEADER: Title + Match Selector (dropdown to pick match file)           │
├───────────────────────────────────────────────┬─────────────────────────┤
│                                               │                         │
│  MAIN VISUALIZATION AREA                      │  DETAIL SIDEBAR         │
│                                               │                         │
│  ┌─────────────────────────────────────────┐  │  Selected: [Node/Edge]  │
│  │ Overview Panel (Layer Boxes)            │  │  Type: striker_state    │
│  │ [Global] → [State] → [Actor] → ...      │  │  Index: 0               │
│  └─────────────────────────────────────────┘  │                         │
│                                               │  Features (8 dims):     │
│  ┌─────────────────────────────────────────┐  │  - runs: 0.45           │
│  │ Context Graph (21 nodes)                │  │  - balls: 0.12          │
│  │ Hierarchical + intra-layer edges        │  │  - SR: 0.67             │
│  └─────────────────────────────────────────┘  │  ...                    │
│                                               │                         │
│  ┌─────────────────────────────────────────┐  │  Edges:                 │
│  │ Ball Timeline (N balls)                 │  │  - conditions → state   │
│  │ Temporal edges as arcs                  │  │  - matchup ↔ bowler     │
│  └─────────────────────────────────────────┘  │                         │
│                                               │                         │
├───────────────────────────────────────────────┴─────────────────────────┤
│ EDGE FILTER BAR: [x]Hierarchical [x]Temporal [x]Cross-domain [x]Query  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Phase 2: Components (Step-by-Step)

#### Step 1: Overview Panel
- 6 clickable layer boxes (Global, State, Actor, Dynamics, Ball, Query)
- Shows node count in each layer
- Click to highlight that layer in main view
- Arrows showing hierarchical flow

#### Step 2: Context Graph View
- 21 context nodes + 1 query node
- Positioned by layer (vertical bands)
- Intra-layer edges (relates_to, matchup)
- Hierarchical edges (conditions) as vertical connectors
- Click node → populate Detail Sidebar

#### Step 3: Ball Timeline View
- Horizontal timeline (left = oldest, right = newest)
- Each ball as a node with label (bowler→batsman)
- Color-coded by over or bowler
- Temporal edges as curved arcs:
  - Above: same_bowler, same_batsman
  - Below: recent_precedes, same_matchup, same_over
- Click ball → show cross-domain edges + Detail Sidebar

#### Step 4: Edge Filter Bar
- Checkbox toggles for each edge category
- Sub-toggles for temporal edge types
- Shows edge count for each type
- Filtering updates both Context Graph and Ball Timeline

#### Step 5: Detail Sidebar
- Updates on node/edge click
- Shows:
  - Node type and index
  - Feature names with actual values
  - Connected edges (incoming/outgoing)
  - Edge attributes (temporal decay, position)

#### Step 6: Match Data Loader
- Load actual HeteroData from a processed match
- Or generate from raw JSON match file
- Populate all views with real values

### Phase 3: Data Structure

```javascript
const graphData = {
  // Context nodes (static structure, dynamic features)
  contextNodes: [
    { id: 'venue', layer: 'global', index: 0, features: { venue_id: 5 } },
    { id: 'batting_team', layer: 'global', index: 0, features: { team_id: 12 } },
    // ... 21 total
  ],

  // Ball nodes (from actual match)
  ballNodes: [
    { id: 'ball_0', index: 0, over: 0, bowler: 'Bumrah', batsman: 'Rohit',
      features: { runs: 0, is_wicket: 0, ... } },
    // ... N balls
  ],

  // Edges grouped by type
  edges: {
    hierarchical: [...],
    intraLayer: [...],
    temporal: {
      recent_precedes: [...],
      same_bowler: [...],
      // ... 7 types
    },
    crossDomain: [...],
    query: [...]
  }
};
```

### Phase 4: File Structure

```
notes/visualizations/
├── heterodata-graph.html        # Current (will be replaced)
├── heterodata-explorer/         # New explorer
│   ├── index.html               # Main page
│   ├── styles.css               # Layout & styling
│   ├── js/
│   │   ├── main.js              # Entry point
│   │   ├── data-loader.js       # Load match data
│   │   ├── overview-panel.js    # Layer boxes
│   │   ├── context-graph.js     # 21 node graph
│   │   ├── ball-timeline.js     # Ball nodes
│   │   ├── edge-filters.js      # Toggle controls
│   │   └── detail-sidebar.js    # Node/edge details
│   └── data/
│       └── sample-match.json    # Pre-processed sample
```

---

## Implementation Order

1. [ ] **Layout skeleton** - HTML/CSS for 4 regions
2. [ ] **Overview Panel** - Layer boxes with counts
3. [ ] **Context Graph** - 21 nodes, hierarchical layout
4. [ ] **Edge rendering** - Hierarchical + intra-layer edges
5. [ ] **Ball Timeline** - Horizontal ball layout
6. [ ] **Temporal edges** - Arcs for ball connections
7. [ ] **Detail Sidebar** - Click-to-show details
8. [ ] **Edge Filters** - Toggle visibility
9. [ ] **Cross-domain edges** - Ball ↔ Context connections
10. [ ] **Real data** - Load actual match HeteroData

---

## Ready to Start

Highlight the first piece of code you'd like to work on, and we'll implement step by step!
