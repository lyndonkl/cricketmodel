/**
 * Overview Panel - Layer boxes showing node counts and flow
 */

const OverviewPanel = {
    container: null,
    selectedLayer: null,

    /**
     * Initialize the overview panel
     */
    init() {
        this.container = document.getElementById('layer-boxes');
        this.render();
    },

    /**
     * Render layer boxes
     */
    render() {
        const data = DataLoader.data;
        if (!data) return;

        // Count nodes per layer
        const counts = {
            global: 3,
            state: 5,
            actor: 7,
            dynamics: 4,
            ball: data.ballNodes.length,
            query: 1
        };

        // Layer order for display
        const layerOrder = ['global', 'state', 'actor', 'dynamics', 'ball', 'query'];

        // Clear container
        this.container.innerHTML = '';

        layerOrder.forEach((layer, i) => {
            // Add layer box
            const box = document.createElement('div');
            box.className = `layer-box ${layer}`;
            box.dataset.layer = layer;
            box.innerHTML = `
                <span class="name">${DataLoader.layers[layer].name}</span>
                <span class="count">${counts[layer]}</span>
            `;

            box.addEventListener('click', () => this.selectLayer(layer));
            this.container.appendChild(box);

            // Add arrow between boxes (except after last)
            if (i < layerOrder.length - 1) {
                const arrow = document.createElement('span');
                arrow.className = 'layer-arrow';
                arrow.textContent = '→';
                this.container.appendChild(arrow);
            }
        });
    },

    /**
     * Select a layer to highlight
     */
    selectLayer(layer) {
        // Update selection state
        if (this.selectedLayer === layer) {
            this.selectedLayer = null;
        } else {
            this.selectedLayer = layer;
        }

        // Update box styling
        this.container.querySelectorAll('.layer-box').forEach(box => {
            box.classList.toggle('selected', box.dataset.layer === this.selectedLayer);
        });

        // Notify other components
        if (typeof ContextGraph !== 'undefined') {
            ContextGraph.highlightLayer(this.selectedLayer);
        }
        if (typeof BallTimeline !== 'undefined') {
            BallTimeline.highlightLayer(this.selectedLayer);
        }
    },

    /**
     * Update counts (when data changes)
     */
    updateCounts(counts) {
        this.container.querySelectorAll('.layer-box').forEach(box => {
            const layer = box.dataset.layer;
            if (counts[layer] !== undefined) {
                box.querySelector('.count').textContent = counts[layer];
            }
        });
    }
};
