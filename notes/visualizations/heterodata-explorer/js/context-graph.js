/**
 * Context Graph - Renders 21 context nodes with hierarchical layout
 */

const ContextGraph = {
    svg: null,
    width: 0,
    height: 0,
    margin: { top: 20, right: 20, bottom: 20, left: 20 },
    selectedNode: null,
    selectedEdge: null,
    highlightedLayer: null,

    // Node positions by layer (calculated on render)
    nodePositions: {},

    /**
     * Initialize the context graph
     */
    init() {
        this.svg = d3.select('#context-svg');
        this.resize();

        // Handle window resize
        window.addEventListener('resize', () => this.resize());
    },

    /**
     * Handle resize
     */
    resize() {
        const container = document.getElementById('context-graph');
        const rect = container.getBoundingClientRect();

        // Account for header
        const headerHeight = container.querySelector('h2').offsetHeight;
        this.width = rect.width - this.margin.left - this.margin.right;
        this.height = rect.height - headerHeight - this.margin.top - this.margin.bottom;

        this.svg
            .attr('width', rect.width)
            .attr('height', rect.height - headerHeight);

        this.render();
    },

    /**
     * Render the context graph
     */
    render() {
        const data = DataLoader.data;
        if (!data || !this.width || !this.height) return;

        this.svg.selectAll('*').remove();

        const g = this.svg.append('g')
            .attr('transform', `translate(${this.margin.left}, ${this.margin.top})`);

        // Calculate positions for each layer
        this.calculatePositions(data.contextNodes);

        // Draw edges first (below nodes)
        this.renderEdges(g, data.edges);

        // Draw nodes
        this.renderNodes(g, data.contextNodes);
    },

    /**
     * Calculate node positions using hierarchical layout
     */
    calculatePositions(nodes) {
        const layerOrder = ['global', 'state', 'actor', 'dynamics', 'query'];
        const layerX = {};
        const layerWidth = this.width / layerOrder.length;

        layerOrder.forEach((layer, i) => {
            layerX[layer] = (i + 0.5) * layerWidth;
        });

        // Group nodes by layer
        const nodesByLayer = {};
        nodes.forEach(node => {
            if (!nodesByLayer[node.layer]) {
                nodesByLayer[node.layer] = [];
            }
            nodesByLayer[node.layer].push(node);
        });

        // Position nodes within each layer
        this.nodePositions = {};
        layerOrder.forEach(layer => {
            const layerNodes = nodesByLayer[layer] || [];
            const layerHeight = this.height;
            const nodeSpacing = layerHeight / (layerNodes.length + 1);

            layerNodes.forEach((node, i) => {
                this.nodePositions[node.id] = {
                    x: layerX[layer],
                    y: (i + 1) * nodeSpacing
                };
            });
        });
    },

    /**
     * Render nodes
     */
    renderNodes(g, nodes) {
        const self = this;

        const nodeGroups = g.selectAll('.node')
            .data(nodes)
            .enter()
            .append('g')
            .attr('class', d => `node ${d.layer}`)
            .attr('transform', d => {
                const pos = this.nodePositions[d.id];
                return `translate(${pos.x}, ${pos.y})`;
            })
            .on('click', function(event, d) {
                self.selectNode(d, this);
            })
            .on('mouseenter', function(event, d) {
                self.showTooltip(event, d);
            })
            .on('mouseleave', function() {
                self.hideTooltip();
            });

        // Node circles
        nodeGroups.append('circle')
            .attr('r', d => d.layer === 'query' ? 18 : 14)
            .attr('fill', d => DataLoader.layers[d.layer].color);

        // Node labels
        nodeGroups.append('text')
            .attr('dy', d => d.layer === 'query' ? 30 : 25)
            .attr('text-anchor', 'middle')
            .text(d => this.getShortName(d.id));
    },

    /**
     * Render edges
     */
    renderEdges(g, edges) {
        const edgeGroup = g.append('g').attr('class', 'edges');

        // Hierarchical edges
        edges.hierarchical.forEach(edge => {
            this.drawEdge(edgeGroup, edge, 'hierarchical');
        });

        // Intra-layer edges
        edges.intra.forEach(edge => {
            this.drawEdge(edgeGroup, edge, 'intra');
        });

        // Query edges (only context node -> query, not ball -> query)
        edges.query.filter(e => !e.source.startsWith('ball_')).forEach(edge => {
            this.drawEdge(edgeGroup, edge, 'query-edge');
        });
    },

    /**
     * Draw a single edge
     */
    drawEdge(g, edge, className) {
        const self = this;
        const sourcePos = this.nodePositions[edge.source];
        const targetPos = this.nodePositions[edge.target];

        if (!sourcePos || !targetPos) return;

        let edgeEl;

        // Curved path for same-layer edges
        if (sourcePos.x === targetPos.x) {
            const midY = (sourcePos.y + targetPos.y) / 2;
            const curveOffset = 30;

            edgeEl = g.append('path')
                .attr('class', `edge ${className}`)
                .attr('d', `M ${sourcePos.x} ${sourcePos.y}
                           Q ${sourcePos.x + curveOffset} ${midY}
                             ${targetPos.x} ${targetPos.y}`)
                .attr('data-source', edge.source)
                .attr('data-target', edge.target)
                .attr('data-type', edge.type)
                .attr('data-category', className);
        } else {
            // Straight line for cross-layer edges
            edgeEl = g.append('line')
                .attr('class', `edge ${className}`)
                .attr('x1', sourcePos.x)
                .attr('y1', sourcePos.y)
                .attr('x2', targetPos.x)
                .attr('y2', targetPos.y)
                .attr('data-source', edge.source)
                .attr('data-target', edge.target)
                .attr('data-type', edge.type)
                .attr('data-category', className);
        }

        // Store edge data for click/hover handlers
        edgeEl.datum(edge);

        // Click handler
        edgeEl.on('click', function(event) {
            event.stopPropagation();
            self.selectEdge(edge, this);
        });

        // Hover handlers
        edgeEl.on('mouseenter', function(event) {
            self.showEdgeTooltip(event, edge);
            d3.select(this).classed('hovered', true);
        });

        edgeEl.on('mouseleave', function() {
            self.hideTooltip();
            d3.select(this).classed('hovered', false);
        });
    },

    /**
     * Get short display name for a node
     */
    getShortName(nodeId) {
        const shortNames = {
            'venue': 'Venue',
            'batting_team': 'Bat Team',
            'bowling_team': 'Bowl Team',
            'score_state': 'Score',
            'chase_state': 'Chase',
            'phase_state': 'Phase',
            'time_pressure': 'Time',
            'wicket_buffer': 'Wickets',
            'striker_identity': 'Striker ID',
            'striker_state': 'Striker',
            'nonstriker_identity': 'NS ID',
            'nonstriker_state': 'Non-Str',
            'bowler_identity': 'Bowler ID',
            'bowler_state': 'Bowler',
            'partnership': 'Partner',
            'batting_momentum': 'Bat Mom',
            'bowling_momentum': 'Bowl Mom',
            'pressure_index': 'Pressure',
            'dot_pressure': 'Dot Press',
            'query': 'Query'
        };
        return shortNames[nodeId] || nodeId;
    },

    /**
     * Select a node
     */
    selectNode(node, element) {
        // Deselect previous node and edge
        this.svg.selectAll('.node.selected').classed('selected', false);
        this.svg.selectAll('.edge.selected').classed('selected', false);
        this.selectedEdge = null;

        if (this.selectedNode === node) {
            this.selectedNode = null;
            DetailSidebar.clear();
        } else {
            this.selectedNode = node;
            d3.select(element).classed('selected', true);
            DetailSidebar.showNode(node);
        }
    },

    /**
     * Select an edge
     */
    selectEdge(edge, element) {
        // Deselect previous node and edge
        this.svg.selectAll('.node.selected').classed('selected', false);
        this.svg.selectAll('.edge.selected').classed('selected', false);
        this.selectedNode = null;

        if (this.selectedEdge === edge) {
            this.selectedEdge = null;
            DetailSidebar.clear();
        } else {
            this.selectedEdge = edge;
            d3.select(element).classed('selected', true);
            DetailSidebar.showEdge(edge);
        }
    },

    /**
     * Show edge tooltip on hover
     */
    showEdgeTooltip(event, edge) {
        const tooltip = d3.select('body').selectAll('.tooltip').data([1]);
        const tooltipEnter = tooltip.enter().append('div').attr('class', 'tooltip');

        const tooltipEl = tooltip.merge(tooltipEnter);
        const edgeType = DataLoader.edgeTypes[edge.type];

        tooltipEl
            .style('left', (event.pageX + 10) + 'px')
            .style('top', (event.pageY - 10) + 'px')
            .html(`
                <div class="title">${edge.type}</div>
                <div class="info">${edge.source} → ${edge.target}</div>
                <div class="info">${edgeType?.description || ''}</div>
            `);
    },

    /**
     * Highlight nodes in a layer
     */
    highlightLayer(layer) {
        this.highlightedLayer = layer;

        this.svg.selectAll('.node').each(function(d) {
            const isHighlighted = !layer || d.layer === layer;
            d3.select(this)
                .style('opacity', isHighlighted ? 1 : 0.3);
        });

        this.svg.selectAll('.edge').each(function() {
            const el = d3.select(this);
            const source = el.attr('data-source');
            const target = el.attr('data-target');

            // Check if edge involves highlighted layer
            const data = DataLoader.data;
            const sourceNode = data.contextNodes.find(n => n.id === source);
            const targetNode = data.contextNodes.find(n => n.id === target);

            const isHighlighted = !layer ||
                (sourceNode && sourceNode.layer === layer) ||
                (targetNode && targetNode.layer === layer);

            el.style('opacity', isHighlighted ? 0.6 : 0.1);
        });
    },

    /**
     * Show tooltip
     */
    showTooltip(event, node) {
        const tooltip = d3.select('body').selectAll('.tooltip').data([1]);
        const tooltipEnter = tooltip.enter().append('div').attr('class', 'tooltip');

        const tooltipEl = tooltip.merge(tooltipEnter);

        tooltipEl
            .style('left', (event.pageX + 10) + 'px')
            .style('top', (event.pageY - 10) + 'px')
            .html(`
                <div class="title">${node.type}</div>
                <div class="info">Layer: ${node.layer}</div>
                <div class="info">Features: ${Object.keys(node.features).length} dims</div>
            `);
    },

    /**
     * Hide tooltip
     */
    hideTooltip() {
        d3.select('.tooltip').remove();
    },

    /**
     * Filter edges by category
     */
    filterEdges(category, visible) {
        this.svg.selectAll(`.edge.${category}`)
            .classed('hidden', !visible);
    }
};
