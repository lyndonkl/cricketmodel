/**
 * Edge Filters - Toggle edge visibility by category and type
 */

const EdgeFilters = {
    /**
     * Initialize edge filters
     */
    init() {
        this.setupCheckboxListeners();
        this.updateCounts();
    },

    /**
     * Setup checkbox event listeners
     */
    setupCheckboxListeners() {
        // Main category filters
        document.getElementById('filter-hierarchical').addEventListener('change', (e) => {
            this.toggleCategory('hierarchical', e.target.checked);
        });

        document.getElementById('filter-intra').addEventListener('change', (e) => {
            this.toggleCategory('intra', e.target.checked);
        });

        document.getElementById('filter-temporal').addEventListener('change', (e) => {
            this.toggleCategory('temporal', e.target.checked);
            // Show/hide temporal subtypes
            document.getElementById('temporal-subtypes').classList.toggle('hidden', !e.target.checked);
        });

        document.getElementById('filter-crossdomain').addEventListener('change', (e) => {
            this.toggleCategory('crossdomain', e.target.checked);
        });

        document.getElementById('filter-query').addEventListener('change', (e) => {
            this.toggleCategory('query-edge', e.target.checked);
        });

        // Temporal subtype filters
        document.querySelectorAll('#temporal-subtypes input').forEach(checkbox => {
            checkbox.addEventListener('change', (e) => {
                const type = e.target.dataset.temporal;
                this.toggleTemporalType(type, e.target.checked);
            });
        });
    },

    /**
     * Toggle an edge category
     */
    toggleCategory(category, visible) {
        // Update context graph
        if (typeof ContextGraph !== 'undefined') {
            ContextGraph.filterEdges(category, visible);
        }

        // Update ball timeline (for temporal)
        if (category === 'temporal' && typeof BallTimeline !== 'undefined') {
            BallTimeline.filterEdges(visible);
        }
    },

    /**
     * Toggle a specific temporal edge type
     */
    toggleTemporalType(type, visible) {
        if (typeof BallTimeline !== 'undefined') {
            BallTimeline.toggleTemporalType(type, visible);
        }
    },

    /**
     * Update edge counts in the UI
     */
    updateCounts() {
        const counts = DataLoader.getEdgeCounts();

        // Update main category counts
        const countSpans = {
            'filter-hierarchical': counts.hierarchical,
            'filter-intra': counts.intra,
            'filter-temporal': counts.temporal,
            'filter-crossdomain': counts.crossdomain,
            'filter-query': counts.query
        };

        for (const [id, count] of Object.entries(countSpans)) {
            const label = document.getElementById(id).parentElement;
            const countSpan = label.querySelector('.count');
            if (countSpan) {
                countSpan.textContent = `(${count})`;
            }
        }

        // Update temporal subtype counts
        if (counts.temporalSubtypes) {
            document.querySelectorAll('#temporal-subtypes input').forEach(checkbox => {
                const type = checkbox.dataset.temporal;
                const count = counts.temporalSubtypes[type] || 0;
                const label = checkbox.parentElement;
                const text = label.textContent.split('(')[0].trim();
                label.innerHTML = `<input type="checkbox" data-temporal="${type}" ${checkbox.checked ? 'checked' : ''}> ${text} <span class="count">(${count})</span>`;
            });
        }
    }
};
