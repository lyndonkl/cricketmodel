/**
 * Main Entry Point - Initializes all components
 */

document.addEventListener('DOMContentLoaded', () => {
    console.log('HeteroData Explorer initializing...');

    // Load sample data
    DataLoader.loadSampleData();
    console.log('Sample data loaded:', DataLoader.data);

    // Update match info
    const match = DataLoader.data.match;
    document.getElementById('match-info').textContent =
        `${match.battingTeam} vs ${match.bowlingTeam} | Inn ${match.innings} | Ball ${match.ballIdx}`;

    // Initialize all components
    OverviewPanel.init();
    console.log('Overview panel initialized');

    ContextGraph.init();
    console.log('Context graph initialized');

    BallTimeline.init();
    console.log('Ball timeline initialized');

    EdgeFilters.init();
    console.log('Edge filters initialized');

    DetailSidebar.init();
    console.log('Detail sidebar initialized');

    // Setup match selector
    document.getElementById('match-selector').addEventListener('change', (e) => {
        // Future: load different match data
        console.log('Match selected:', e.target.value);
    });

    console.log('HeteroData Explorer ready!');
});
