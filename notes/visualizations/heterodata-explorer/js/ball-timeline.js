/**
 * Ball Timeline - Horizontal timeline of ball nodes with temporal edges as arcs
 */

const BallTimeline = {
    svg: null,
    width: 0,
    height: 0,
    margin: { top: 40, right: 20, bottom: 20, left: 20 },
    selectedBall: null,
    selectedEdge: null,
    ballPositions: {},

    // Visible temporal edge types
    visibleTemporalTypes: new Set([
        'recent_precedes', 'medium_precedes', 'distant_precedes',
        'same_bowler', 'same_batsman', 'same_matchup', 'same_over'
    ]),

    /**
     * Initialize the timeline
     */
    init() {
        this.svg = d3.select('#timeline-svg');
        this.resize();

        window.addEventListener('resize', () => this.resize());
    },

    /**
     * Handle resize
     */
    resize() {
        const container = document.getElementById('ball-timeline');
        const rect = container.getBoundingClientRect();
        const headerHeight = container.querySelector('h2').offsetHeight;

        this.width = rect.width - this.margin.left - this.margin.right;
        this.height = rect.height - headerHeight - this.margin.top - this.margin.bottom;

        this.svg
            .attr('width', rect.width)
            .attr('height', rect.height - headerHeight);

        this.render();
    },

    /**
     * Render the timeline
     */
    render() {
        const data = DataLoader.data;
        if (!data || !this.width || !this.height) return;

        this.svg.selectAll('*').remove();

        const g = this.svg.append('g')
            .attr('transform', `translate(${this.margin.left}, ${this.margin.top})`);

        // Calculate ball positions
        this.calculatePositions(data.ballNodes);

        // Update ball count in header
        document.getElementById('ball-count').textContent = `(${data.ballNodes.length} balls)`;

        if (data.ballNodes.length === 0) {
            g.append('text')
                .attr('x', this.width / 2)
                .attr('y', this.height / 2)
                .attr('text-anchor', 'middle')
                .attr('fill', '#aaa')
                .text('No ball history (first ball prediction)');
            return;
        }

        // Draw temporal arcs first (below balls)
        this.renderTemporalArcs(g, data.edges.temporal);

        // Draw ball nodes
        this.renderBalls(g, data.ballNodes);

        // Draw over markers
        this.renderOverMarkers(g, data.ballNodes);
    },

    /**
     * Calculate ball positions (horizontal layout)
     */
    calculatePositions(balls) {
        this.ballPositions = {};

        const ballWidth = Math.min(40, (this.width - 20) / balls.length);
        const startX = 10;
        const y = this.height / 2;

        balls.forEach((ball, i) => {
            this.ballPositions[ball.id] = {
                x: startX + i * ballWidth + ballWidth / 2,
                y: y,
                width: ballWidth - 4
            };
        });
    },

    /**
     * Render ball nodes
     */
    renderBalls(g, balls) {
        const self = this;

        const ballGroups = g.selectAll('.ball-node')
            .data(balls)
            .enter()
            .append('g')
            .attr('class', 'ball-node')
            .attr('transform', d => {
                const pos = this.ballPositions[d.id];
                return `translate(${pos.x}, ${pos.y})`;
            })
            .on('click', function(event, d) {
                self.selectBall(d, this);
            })
            .on('mouseenter', function(event, d) {
                self.showTooltip(event, d);
            })
            .on('mouseleave', function() {
                self.hideTooltip();
            });

        // Ball rectangles
        ballGroups.append('rect')
            .attr('x', d => -this.ballPositions[d.id].width / 2)
            .attr('y', -12)
            .attr('width', d => this.ballPositions[d.id].width)
            .attr('height', 24)
            .attr('fill', d => this.getBallColor(d));

        // Ball labels (runs or W for wicket)
        ballGroups.append('text')
            .attr('dy', 4)
            .attr('text-anchor', 'middle')
            .attr('fill', '#fff')
            .attr('font-size', '10px')
            .text(d => d.isWicket ? 'W' : d.runs);
    },

    /**
     * Get color based on ball outcome
     */
    getBallColor(ball) {
        if (ball.isWicket) return '#c0392b';
        if (ball.runs === 6) return '#8e44ad';
        if (ball.runs === 4) return '#27ae60';
        if (ball.runs === 0) return '#7f8c8d';
        return '#e74c3c';
    },

    /**
     * Render temporal arcs between balls
     */
    renderTemporalArcs(g, temporalEdges) {
        const arcGroup = g.append('g').attr('class', 'temporal-arcs');

        // Different arc heights for different edge types
        const arcHeights = {
            recent_precedes: 15,
            medium_precedes: 25,
            distant_precedes: 35,
            same_bowler: -20,
            same_batsman: -30,
            same_matchup: -40,
            same_over: -15
        };

        // Limit edges to render (for performance)
        const maxEdgesPerType = 50;

        for (const [type, edges] of Object.entries(temporalEdges)) {
            if (!this.visibleTemporalTypes.has(type)) continue;

            const height = arcHeights[type] || 20;
            const limitedEdges = edges.slice(0, maxEdgesPerType);

            limitedEdges.forEach(edge => {
                this.drawArc(arcGroup, edge, type, height);
            });
        }
    },

    /**
     * Draw a single arc
     */
    drawArc(g, edge, type, height) {
        const self = this;
        const sourcePos = this.ballPositions[edge.source];
        const targetPos = this.ballPositions[edge.target];

        if (!sourcePos || !targetPos) return;

        const midX = (sourcePos.x + targetPos.x) / 2;
        const arcY = sourcePos.y + height;

        const arcEl = g.append('path')
            .attr('class', `temporal-arc ${type}`)
            .attr('d', `M ${sourcePos.x} ${sourcePos.y}
                       Q ${midX} ${arcY}
                         ${targetPos.x} ${targetPos.y}`)
            .attr('data-type', type)
            .attr('data-source', edge.source)
            .attr('data-target', edge.target);

        // Store edge data
        arcEl.datum(edge);

        // Click handler
        arcEl.on('click', function(event) {
            event.stopPropagation();
            self.selectTemporalEdge(edge, this);
        });

        // Hover handlers
        arcEl.on('mouseenter', function(event) {
            self.showEdgeTooltip(event, edge);
            d3.select(this).classed('hovered', true);
        });

        arcEl.on('mouseleave', function() {
            self.hideTooltip();
            d3.select(this).classed('hovered', false);
        });
    },

    /**
     * Render over markers
     */
    renderOverMarkers(g, balls) {
        // Find over boundaries
        const overBoundaries = [];
        let currentOver = -1;

        balls.forEach(ball => {
            if (ball.over !== currentOver) {
                currentOver = ball.over;
                overBoundaries.push({ ball: ball, over: currentOver });
            }
        });

        // Draw over markers
        overBoundaries.forEach(boundary => {
            const pos = this.ballPositions[boundary.ball.id];
            if (!pos) return;

            g.append('text')
                .attr('x', pos.x)
                .attr('y', -5)
                .attr('text-anchor', 'middle')
                .attr('font-size', '9px')
                .attr('fill', '#888')
                .text(`Over ${boundary.over + 1}`);
        });
    },

    /**
     * Select a ball
     */
    selectBall(ball, element) {
        // Deselect previous ball and edge
        this.svg.selectAll('.ball-node.selected').classed('selected', false);
        this.svg.selectAll('.temporal-arc.selected').classed('selected', false);
        this.selectedEdge = null;

        if (this.selectedBall === ball) {
            this.selectedBall = null;
            DetailSidebar.clear();
            this.clearCrossDomainHighlight();
        } else {
            this.selectedBall = ball;
            d3.select(element).classed('selected', true);
            DetailSidebar.showBall(ball);
            this.highlightCrossDomainEdges(ball);
        }
    },

    /**
     * Select a temporal edge
     */
    selectTemporalEdge(edge, element) {
        // Deselect previous ball and edge
        this.svg.selectAll('.ball-node.selected').classed('selected', false);
        this.svg.selectAll('.temporal-arc.selected').classed('selected', false);
        this.selectedBall = null;
        this.clearCrossDomainHighlight();

        if (this.selectedEdge === edge) {
            this.selectedEdge = null;
            DetailSidebar.clear();
        } else {
            this.selectedEdge = edge;
            d3.select(element).classed('selected', true);
            DetailSidebar.showTemporalEdge(edge);
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
                ${edge.decay ? `<div class="info">Decay: ${edge.decay.toFixed(3)}</div>` : ''}
            `);
    },

    /**
     * Highlight cross-domain edges for selected ball
     */
    highlightCrossDomainEdges(ball) {
        const data = DataLoader.data;
        if (!data) return;

        // Find cross-domain edges involving this ball
        const relatedEdges = data.edges.crossdomain.filter(e => e.source === ball.id);

        // Highlight target nodes in context graph
        if (typeof ContextGraph !== 'undefined') {
            const targetNodes = relatedEdges.map(e => e.target);
            ContextGraph.svg.selectAll('.node').each(function(d) {
                const isTarget = targetNodes.includes(d.id);
                d3.select(this)
                    .style('opacity', isTarget ? 1 : 0.3)
                    .classed('crossdomain-target', isTarget);
            });
        }
    },

    /**
     * Clear cross-domain highlighting
     */
    clearCrossDomainHighlight() {
        if (typeof ContextGraph !== 'undefined') {
            ContextGraph.svg.selectAll('.node')
                .style('opacity', 1)
                .classed('crossdomain-target', false);
        }
    },

    /**
     * Highlight layer (called from overview panel)
     */
    highlightLayer(layer) {
        if (layer === 'ball') {
            this.svg.selectAll('.ball-node').style('opacity', 1);
        } else if (layer) {
            this.svg.selectAll('.ball-node').style('opacity', 0.3);
        } else {
            this.svg.selectAll('.ball-node').style('opacity', 1);
        }
    },

    /**
     * Toggle temporal edge type visibility
     */
    toggleTemporalType(type, visible) {
        if (visible) {
            this.visibleTemporalTypes.add(type);
        } else {
            this.visibleTemporalTypes.delete(type);
        }

        this.svg.selectAll(`.temporal-arc.${type}`)
            .style('display', visible ? 'block' : 'none');
    },

    /**
     * Show tooltip
     */
    showTooltip(event, ball) {
        const tooltip = d3.select('body').selectAll('.tooltip').data([1]);
        const tooltipEnter = tooltip.enter().append('div').attr('class', 'tooltip');

        const tooltipEl = tooltip.merge(tooltipEnter);

        tooltipEl
            .style('left', (event.pageX + 10) + 'px')
            .style('top', (event.pageY - 10) + 'px')
            .html(`
                <div class="title">Ball ${ball.index + 1}</div>
                <div class="info">Over: ${ball.over + 1}.${ball.ballInOver + 1}</div>
                <div class="info">${ball.bowler} → ${ball.batsman}</div>
                <div class="info">Runs: ${ball.runs}${ball.isWicket ? ' (WICKET)' : ''}</div>
            `);
    },

    /**
     * Hide tooltip
     */
    hideTooltip() {
        d3.select('.tooltip').remove();
    },

    /**
     * Filter all temporal edges
     */
    filterEdges(visible) {
        this.svg.selectAll('.temporal-arc')
            .style('display', visible ? 'block' : 'none');
    }
};
