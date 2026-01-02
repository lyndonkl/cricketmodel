/**
 * Detail Sidebar - Shows node/edge details on selection
 */

const DetailSidebar = {
    container: null,

    /**
     * Initialize the sidebar
     */
    init() {
        this.container = document.getElementById('detail-content');
    },

    /**
     * Show node details
     */
    showNode(node) {
        const data = DataLoader.data;

        let html = `
            <div class="detail-section">
                <h3>Node Info</h3>
                <div class="detail-row">
                    <span class="label">Type</span>
                    <span class="value">${node.type}</span>
                </div>
                <div class="detail-row">
                    <span class="label">Layer</span>
                    <span class="value">${node.layer}</span>
                </div>
                <div class="detail-row">
                    <span class="label">Description</span>
                    <span class="value" style="font-family: inherit; font-size: 12px;">${DataLoader.nodeTypes[node.type]?.description || '-'}</span>
                </div>
            </div>

            <div class="detail-section">
                <h3>Features (${node.featureNames.length} dims)</h3>
                <div class="feature-list">
        `;

        // Show feature values
        node.featureNames.forEach((name, i) => {
            const value = Object.values(node.features)[i];
            const formattedValue = typeof value === 'number' ? value.toFixed(3) : value;

            html += `
                <div class="detail-row">
                    <span class="label">${name}</span>
                    <span class="value">${formattedValue}</span>
                </div>
            `;
        });

        html += `
                </div>
            </div>
        `;

        // Show extra attributes if present (e.g., team_id, role_id for identity nodes)
        if (node.extraAttrs && Object.keys(node.extraAttrs).length > 0) {
            html += `
                <div class="detail-section">
                    <h3>Extra Attributes (hierarchical fallback)</h3>
                    <div class="feature-list">
            `;

            for (const [name, value] of Object.entries(node.extraAttrs)) {
                html += `
                    <div class="detail-row">
                        <span class="label">${name}</span>
                        <span class="value">${value}</span>
                    </div>
                `;
            }

            html += `
                    </div>
                </div>
            `;
        }

        // Find connected edges
        const incomingEdges = this.findIncomingEdges(node.id, data.edges);
        const outgoingEdges = this.findOutgoingEdges(node.id, data.edges);

        if (incomingEdges.length > 0 || outgoingEdges.length > 0) {
            html += `
                <div class="detail-section">
                    <h3>Edges</h3>
            `;

            if (incomingEdges.length > 0) {
                html += `<h4 style="font-size: 11px; color: #888; margin: 8px 0 4px;">Incoming (${incomingEdges.length})</h4>`;
                html += '<ul class="edge-list">';
                incomingEdges.slice(0, 10).forEach(edge => {
                    html += `<li>${edge.source} <span style="color: #666;">(${edge.type})</span></li>`;
                });
                if (incomingEdges.length > 10) {
                    html += `<li style="color: #666;">... and ${incomingEdges.length - 10} more</li>`;
                }
                html += '</ul>';
            }

            if (outgoingEdges.length > 0) {
                html += `<h4 style="font-size: 11px; color: #888; margin: 8px 0 4px;">Outgoing (${outgoingEdges.length})</h4>`;
                html += '<ul class="edge-list">';
                outgoingEdges.slice(0, 10).forEach(edge => {
                    html += `<li>${edge.target} <span style="color: #666;">(${edge.type})</span></li>`;
                });
                if (outgoingEdges.length > 10) {
                    html += `<li style="color: #666;">... and ${outgoingEdges.length - 10} more</li>`;
                }
                html += '</ul>';
            }

            html += '</div>';
        }

        this.container.innerHTML = html;
    },

    /**
     * Show ball details
     */
    showBall(ball) {
        const data = DataLoader.data;

        let html = `
            <div class="detail-section">
                <h3>Ball Info</h3>
                <div class="detail-row">
                    <span class="label">Index</span>
                    <span class="value">${ball.index}</span>
                </div>
                <div class="detail-row">
                    <span class="label">Over</span>
                    <span class="value">${ball.over + 1}.${ball.ballInOver + 1}</span>
                </div>
                <div class="detail-row">
                    <span class="label">Bowler</span>
                    <span class="value">${ball.bowler}</span>
                </div>
                <div class="detail-row">
                    <span class="label">Batsman</span>
                    <span class="value">${ball.batsman}</span>
                </div>
                <div class="detail-row">
                    <span class="label">Non-striker</span>
                    <span class="value">${ball.nonstriker}</span>
                </div>
                <div class="detail-row">
                    <span class="label">Runs</span>
                    <span class="value">${ball.runs}</span>
                </div>
                <div class="detail-row">
                    <span class="label">Wicket</span>
                    <span class="value">${ball.isWicket ? 'Yes' : 'No'}</span>
                </div>
                <div class="detail-row">
                    <span class="label">Boundary</span>
                    <span class="value">${ball.isBoundary ? 'Yes' : 'No'}</span>
                </div>
            </div>

            <div class="detail-section">
                <h3>Features (normalized)</h3>
                <div class="feature-list">
        `;

        // Show feature values
        for (const [name, value] of Object.entries(ball.features)) {
            const formattedValue = typeof value === 'number' ? value.toFixed(3) : value;
            html += `
                <div class="detail-row">
                    <span class="label">${name}</span>
                    <span class="value">${formattedValue}</span>
                </div>
            `;
        }

        html += `
                </div>
            </div>
        `;

        // Show extra attributes if present (player IDs for embedding lookup)
        if (ball.extraAttrs && Object.keys(ball.extraAttrs).length > 0) {
            html += `
                <div class="detail-section">
                    <h3>Player IDs (for embedding lookup)</h3>
                    <div class="feature-list">
            `;

            for (const [name, value] of Object.entries(ball.extraAttrs)) {
                html += `
                    <div class="detail-row">
                        <span class="label">${name}</span>
                        <span class="value">${value}</span>
                    </div>
                `;
            }

            html += `
                    </div>
                </div>
            `;
        }

        // Find connected edges
        const temporalEdges = this.findBallTemporalEdges(ball.id, data.edges.temporal);
        const crossDomainEdges = data.edges.crossdomain.filter(e => e.source === ball.id);

        if (temporalEdges.incoming.length > 0 || temporalEdges.outgoing.length > 0) {
            html += `
                <div class="detail-section">
                    <h3>Temporal Edges</h3>
            `;

            if (temporalEdges.incoming.length > 0) {
                html += `<h4 style="font-size: 11px; color: #888; margin: 8px 0 4px;">From earlier balls (${temporalEdges.incoming.length})</h4>`;
                html += '<ul class="edge-list">';
                temporalEdges.incoming.slice(0, 5).forEach(edge => {
                    html += `<li>${edge.source} <span style="color: #666;">(${edge.type})</span></li>`;
                });
                if (temporalEdges.incoming.length > 5) {
                    html += `<li style="color: #666;">... and ${temporalEdges.incoming.length - 5} more</li>`;
                }
                html += '</ul>';
            }

            if (temporalEdges.outgoing.length > 0) {
                html += `<h4 style="font-size: 11px; color: #888; margin: 8px 0 4px;">To later balls (${temporalEdges.outgoing.length})</h4>`;
                html += '<ul class="edge-list">';
                temporalEdges.outgoing.slice(0, 5).forEach(edge => {
                    html += `<li>${edge.target} <span style="color: #666;">(${edge.type})</span></li>`;
                });
                if (temporalEdges.outgoing.length > 5) {
                    html += `<li style="color: #666;">... and ${temporalEdges.outgoing.length - 5} more</li>`;
                }
                html += '</ul>';
            }

            html += '</div>';
        }

        if (crossDomainEdges.length > 0) {
            html += `
                <div class="detail-section">
                    <h3>Cross-Domain Edges</h3>
                    <ul class="edge-list">
            `;

            crossDomainEdges.forEach(edge => {
                html += `<li>${edge.target} <span style="color: #666;">(${edge.type})</span></li>`;
            });

            html += '</ul></div>';
        }

        this.container.innerHTML = html;
    },

    /**
     * Show edge details
     */
    showEdge(edge) {
        const edgeType = DataLoader.edgeTypes[edge.type];
        const category = edgeType?.category || 'unknown';

        // Get source and target node info
        const data = DataLoader.data;
        const sourceNode = data.contextNodes.find(n => n.id === edge.source);
        const targetNode = data.contextNodes.find(n => n.id === edge.target);

        let html = `
            <div class="detail-section">
                <h3>Edge Info</h3>
                <div class="detail-row">
                    <span class="label">Type</span>
                    <span class="value">${edge.type}</span>
                </div>
                <div class="detail-row">
                    <span class="label">Category</span>
                    <span class="value">${category}</span>
                </div>
                <div class="detail-row">
                    <span class="label">Description</span>
                    <span class="value" style="font-family: inherit; font-size: 12px;">${edgeType?.description || '-'}</span>
                </div>
            </div>

            <div class="detail-section">
                <h3>Connection</h3>
                <div class="detail-row">
                    <span class="label">Source</span>
                    <span class="value">${edge.source}</span>
                </div>
        `;

        if (sourceNode) {
            html += `
                <div class="detail-row">
                    <span class="label">Source Layer</span>
                    <span class="value">${sourceNode.layer}</span>
                </div>
            `;
        }

        html += `
                <div class="detail-row">
                    <span class="label">Target</span>
                    <span class="value">${edge.target}</span>
                </div>
        `;

        if (targetNode) {
            html += `
                <div class="detail-row">
                    <span class="label">Target Layer</span>
                    <span class="value">${targetNode.layer}</span>
                </div>
            `;
        }

        html += '</div>';

        // Show edge attributes if present (e.g., decay for temporal edges)
        const edgeAttrs = {};
        for (const [key, value] of Object.entries(edge)) {
            if (!['source', 'target', 'type'].includes(key)) {
                edgeAttrs[key] = value;
            }
        }

        if (Object.keys(edgeAttrs).length > 0) {
            html += `
                <div class="detail-section">
                    <h3>Edge Attributes</h3>
                    <div class="feature-list">
            `;

            for (const [name, value] of Object.entries(edgeAttrs)) {
                const formattedValue = typeof value === 'number' ? value.toFixed(4) : value;
                html += `
                    <div class="detail-row">
                        <span class="label">${name}</span>
                        <span class="value">${formattedValue}</span>
                    </div>
                `;
            }

            html += `
                    </div>
                </div>
            `;
        }

        // Show source node features if it's a context node
        if (sourceNode) {
            html += `
                <div class="detail-section">
                    <h3>Source Features (${sourceNode.featureNames.length} dims)</h3>
                    <div class="feature-list">
            `;

            sourceNode.featureNames.forEach((name, i) => {
                const value = Object.values(sourceNode.features)[i];
                const formattedValue = typeof value === 'number' ? value.toFixed(3) : value;
                html += `
                    <div class="detail-row">
                        <span class="label">${name}</span>
                        <span class="value">${formattedValue}</span>
                    </div>
                `;
            });

            html += `
                    </div>
                </div>
            `;
        }

        // Show target node features if it's a context node
        if (targetNode) {
            html += `
                <div class="detail-section">
                    <h3>Target Features (${targetNode.featureNames.length} dims)</h3>
                    <div class="feature-list">
            `;

            targetNode.featureNames.forEach((name, i) => {
                const value = Object.values(targetNode.features)[i];
                const formattedValue = typeof value === 'number' ? value.toFixed(3) : value;
                html += `
                    <div class="detail-row">
                        <span class="label">${name}</span>
                        <span class="value">${formattedValue}</span>
                    </div>
                `;
            });

            html += `
                    </div>
                </div>
            `;
        }

        this.container.innerHTML = html;
    },

    /**
     * Show temporal edge details (ball to ball)
     */
    showTemporalEdge(edge) {
        const edgeType = DataLoader.edgeTypes[edge.type];
        const data = DataLoader.data;

        // Find source and target balls
        const sourceBall = data.ballNodes.find(b => b.id === edge.source);
        const targetBall = data.ballNodes.find(b => b.id === edge.target);

        let html = `
            <div class="detail-section">
                <h3>Temporal Edge</h3>
                <div class="detail-row">
                    <span class="label">Type</span>
                    <span class="value">${edge.type}</span>
                </div>
                <div class="detail-row">
                    <span class="label">Category</span>
                    <span class="value">temporal</span>
                </div>
                <div class="detail-row">
                    <span class="label">Description</span>
                    <span class="value" style="font-family: inherit; font-size: 12px;">${edgeType?.description || '-'}</span>
                </div>
            </div>
        `;

        // Show edge attributes (decay, position)
        const edgeAttrs = {};
        for (const [key, value] of Object.entries(edge)) {
            if (!['source', 'target', 'type'].includes(key)) {
                edgeAttrs[key] = value;
            }
        }

        if (Object.keys(edgeAttrs).length > 0) {
            html += `
                <div class="detail-section">
                    <h3>Edge Attributes</h3>
                    <div class="feature-list">
            `;

            for (const [name, value] of Object.entries(edgeAttrs)) {
                const formattedValue = typeof value === 'number' ? value.toFixed(4) : value;
                html += `
                    <div class="detail-row">
                        <span class="label">${name}</span>
                        <span class="value">${formattedValue}</span>
                    </div>
                `;
            }

            html += `
                    </div>
                </div>
            `;
        }

        // Show source ball info
        if (sourceBall) {
            html += `
                <div class="detail-section">
                    <h3>Source Ball</h3>
                    <div class="detail-row">
                        <span class="label">Ball</span>
                        <span class="value">${sourceBall.id}</span>
                    </div>
                    <div class="detail-row">
                        <span class="label">Over</span>
                        <span class="value">${sourceBall.over + 1}.${sourceBall.ballInOver + 1}</span>
                    </div>
                    <div class="detail-row">
                        <span class="label">Bowler</span>
                        <span class="value">${sourceBall.bowler}</span>
                    </div>
                    <div class="detail-row">
                        <span class="label">Batsman</span>
                        <span class="value">${sourceBall.batsman}</span>
                    </div>
                    <div class="detail-row">
                        <span class="label">Runs</span>
                        <span class="value">${sourceBall.runs}${sourceBall.isWicket ? ' (W)' : ''}</span>
                    </div>
                </div>
            `;
        }

        // Show target ball info
        if (targetBall) {
            html += `
                <div class="detail-section">
                    <h3>Target Ball</h3>
                    <div class="detail-row">
                        <span class="label">Ball</span>
                        <span class="value">${targetBall.id}</span>
                    </div>
                    <div class="detail-row">
                        <span class="label">Over</span>
                        <span class="value">${targetBall.over + 1}.${targetBall.ballInOver + 1}</span>
                    </div>
                    <div class="detail-row">
                        <span class="label">Bowler</span>
                        <span class="value">${targetBall.bowler}</span>
                    </div>
                    <div class="detail-row">
                        <span class="label">Batsman</span>
                        <span class="value">${targetBall.batsman}</span>
                    </div>
                    <div class="detail-row">
                        <span class="label">Runs</span>
                        <span class="value">${targetBall.runs}${targetBall.isWicket ? ' (W)' : ''}</span>
                    </div>
                </div>
            `;
        }

        // Show ball gap info
        if (sourceBall && targetBall) {
            const gap = targetBall.index - sourceBall.index;
            html += `
                <div class="detail-section">
                    <h3>Connection</h3>
                    <div class="detail-row">
                        <span class="label">Ball Gap</span>
                        <span class="value">${gap} balls</span>
                    </div>
                    <div class="detail-row">
                        <span class="label">Same Bowler</span>
                        <span class="value">${sourceBall.bowler === targetBall.bowler ? 'Yes' : 'No'}</span>
                    </div>
                    <div class="detail-row">
                        <span class="label">Same Batsman</span>
                        <span class="value">${sourceBall.batsman === targetBall.batsman ? 'Yes' : 'No'}</span>
                    </div>
                    <div class="detail-row">
                        <span class="label">Same Over</span>
                        <span class="value">${sourceBall.over === targetBall.over ? 'Yes' : 'No'}</span>
                    </div>
                </div>
            `;
        }

        this.container.innerHTML = html;
    },

    /**
     * Find incoming edges for a node
     */
    findIncomingEdges(nodeId, edges) {
        const incoming = [];

        // Hierarchical
        edges.hierarchical.filter(e => e.target === nodeId).forEach(e => incoming.push(e));

        // Intra
        edges.intra.filter(e => e.target === nodeId).forEach(e => incoming.push(e));

        // Cross-domain
        edges.crossdomain.filter(e => e.target === nodeId).forEach(e => incoming.push(e));

        // Query
        edges.query.filter(e => e.target === nodeId).forEach(e => incoming.push(e));

        return incoming;
    },

    /**
     * Find outgoing edges for a node
     */
    findOutgoingEdges(nodeId, edges) {
        const outgoing = [];

        // Hierarchical
        edges.hierarchical.filter(e => e.source === nodeId).forEach(e => outgoing.push(e));

        // Intra
        edges.intra.filter(e => e.source === nodeId).forEach(e => outgoing.push(e));

        // Cross-domain
        edges.crossdomain.filter(e => e.source === nodeId).forEach(e => outgoing.push(e));

        // Query
        edges.query.filter(e => e.source === nodeId).forEach(e => outgoing.push(e));

        return outgoing;
    },

    /**
     * Find temporal edges for a ball
     */
    findBallTemporalEdges(ballId, temporalEdges) {
        const incoming = [];
        const outgoing = [];

        for (const edges of Object.values(temporalEdges)) {
            edges.forEach(edge => {
                if (edge.target === ballId) incoming.push(edge);
                if (edge.source === ballId) outgoing.push(edge);
            });
        }

        return { incoming, outgoing };
    },

    /**
     * Clear the sidebar
     */
    clear() {
        this.container.innerHTML = '<p class="placeholder">Click a node or edge to see details</p>';
    }
};
