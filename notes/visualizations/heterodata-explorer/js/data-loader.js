/**
 * Data Loader - Manages graph data structure and loading
 */

const DataLoader = {
    // Current loaded graph data
    data: null,

    // Node type metadata with layer assignments
    // IMPORTANT: Layer assignments match edge_builder.py LAYER_NODES for correct edge generation
    nodeTypes: {
        // Global layer (3 nodes) - venue and team IDs
        venue: { layer: 'global', index: 0, featureDim: 1, description: 'Match venue ID' },
        batting_team: { layer: 'global', index: 1, featureDim: 1, description: 'Batting team ID' },
        bowling_team: { layer: 'global', index: 2, featureDim: 1, description: 'Bowling team ID' },

        // State layer (5 nodes)
        score_state: { layer: 'state', index: 0, featureDim: 5, description: 'Current score, wickets, balls' },
        chase_state: { layer: 'state', index: 1, featureDim: 7, description: 'Chase target, RRR, difficulty' },
        phase_state: { layer: 'state', index: 2, featureDim: 6, description: 'Powerplay, middle, death phase' },
        time_pressure: { layer: 'state', index: 3, featureDim: 3, description: 'Balls remaining, urgency, is_final_over' },
        wicket_buffer: { layer: 'state', index: 4, featureDim: 2, description: 'Wickets in hand, is_tail indicator' },

        // Actor layer (7 nodes) - identity + state + partnership
        // Identity nodes have .x (player_id) + .team_id + .role_id for hierarchical fallback
        striker_identity: { layer: 'actor', index: 0, featureDim: 1, extraAttrs: ['team_id', 'role_id'], description: 'Striker player ID + team/role fallback' },
        striker_state: { layer: 'actor', index: 1, featureDim: 8, description: 'Striker: runs, balls, SR, dots_pct, is_set, boundaries, is_debut, balls_since' },
        nonstriker_identity: { layer: 'actor', index: 2, featureDim: 1, extraAttrs: ['team_id', 'role_id'], description: 'Non-striker player ID + team/role fallback' },
        nonstriker_state: { layer: 'actor', index: 3, featureDim: 8, description: 'Non-striker: Z2 symmetric with striker' },
        bowler_identity: { layer: 'actor', index: 4, featureDim: 1, extraAttrs: ['team_id', 'role_id'], description: 'Bowler player ID + team/role fallback' },
        bowler_state: { layer: 'actor', index: 5, featureDim: 8, description: 'Bowler: balls, runs, wickets, economy, dots_pct, threat, is_pace, is_spin' },
        partnership: { layer: 'actor', index: 6, featureDim: 4, description: 'Partnership: runs, balls, run_rate, stability' },

        // Dynamics layer (4 nodes)
        batting_momentum: { layer: 'dynamics', index: 0, featureDim: 1, description: 'Recent batting trend' },
        bowling_momentum: { layer: 'dynamics', index: 1, featureDim: 1, description: 'Recent bowling trend' },
        pressure_index: { layer: 'dynamics', index: 2, featureDim: 1, description: 'Overall pressure level' },
        dot_pressure: { layer: 'dynamics', index: 3, featureDim: 5, description: 'consecutive_dots, balls_since_boundary, balls_since_wicket, pressure_accumulated, pressure_trend' },

        // Ball layer (N nodes, 18 features each)
        // Features: runs/6, is_wicket, over/20, ball_in_over/6, is_boundary,
        //           is_wide, is_noball, is_bye, is_legbye,
        //           wicket_bowled, wicket_caught, wicket_lbw, wicket_run_out, wicket_stumped, wicket_other,
        //           striker_run_out, nonstriker_run_out, bowling_end
        // Extra attrs: bowler_ids, batsman_ids, nonstriker_ids (for embedding lookup)
        ball: { layer: 'ball', featureDim: 18, extraAttrs: ['bowler_ids', 'batsman_ids', 'nonstriker_ids'], description: 'Historical ball with 18 features + player IDs' },

        // Query layer (1 node)
        // Placeholder initialized to zeros, model learns embedding
        query: { layer: 'query', index: 0, featureDim: 1, description: 'Placeholder for learned embedding - aggregates via attention' }
    },

    // Layer metadata - matches edge_builder.py LAYER_NODES
    layers: {
        global: { name: 'Global', color: '#4a90d9', nodes: ['venue', 'batting_team', 'bowling_team'] },
        state: { name: 'State', color: '#50c878', nodes: ['score_state', 'chase_state', 'phase_state', 'time_pressure', 'wicket_buffer'] },
        actor: { name: 'Actor', color: '#f5a623', nodes: ['striker_identity', 'striker_state', 'nonstriker_identity', 'nonstriker_state', 'bowler_identity', 'bowler_state', 'partnership'] },
        dynamics: { name: 'Dynamics', color: '#9b59b6', nodes: ['batting_momentum', 'bowling_momentum', 'pressure_index', 'dot_pressure'] },
        ball: { name: 'Ball', color: '#e74c3c', nodes: [] },  // Dynamic
        query: { name: 'Query', color: '#1abc9c', nodes: ['query'] }
    },

    // Edge type metadata
    edgeTypes: {
        // Hierarchical (conditions)
        conditions: { category: 'hierarchical', description: 'Information flow from parent to child layers' },

        // Intra-layer
        relates_to: { category: 'intra', description: 'Same-layer relationships' },
        matchup: { category: 'intra', description: 'Striker-bowler matchup connection' },

        // Temporal (ball-to-ball)
        recent_precedes: { category: 'temporal', description: 'Recent predecessor (1-3 balls ago)' },
        medium_precedes: { category: 'temporal', description: 'Medium predecessor (4-10 balls ago)' },
        distant_precedes: { category: 'temporal', description: 'Distant predecessor (11+ balls ago)' },
        same_bowler: { category: 'temporal', description: 'Same bowler bowled this ball' },
        same_batsman: { category: 'temporal', description: 'Same batsman faced this ball' },
        same_matchup: { category: 'temporal', description: 'Same bowler-batsman matchup' },
        same_over: { category: 'temporal', description: 'Same over as this ball' },

        // Cross-domain (ball to context)
        faced_by: { category: 'crossdomain', description: 'Ball faced by current striker' },
        bowled_by: { category: 'crossdomain', description: 'Ball bowled by current bowler' },
        partnered_by: { category: 'crossdomain', description: 'Ball during current non-striker partnership' },
        informs: { category: 'crossdomain', description: 'Ball informs dynamics nodes' },

        // Query
        attends: { category: 'query', description: 'Query attends to context nodes' },
        drives: { category: 'query', description: 'Query drives prediction output' }
    },

    /**
     * Load sample graph data
     */
    loadSampleData() {
        // Generate sample data for a match
        const sampleBalls = this.generateSampleBalls(50);

        this.data = {
            match: {
                venue: 'Wankhede Stadium',
                battingTeam: 'Mumbai Indians',
                bowlingTeam: 'Chennai Super Kings',
                innings: 1,
                ballIdx: 50
            },

            // Context nodes with sample features
            contextNodes: this.generateContextNodes(),

            // Ball nodes
            ballNodes: sampleBalls,

            // Edges grouped by category
            edges: this.generateEdges(sampleBalls)
        };

        return this.data;
    },

    /**
     * Generate context nodes with sample feature values
     * Order matches edge_builder.py LAYER_NODES for each layer
     */
    generateContextNodes() {
        const nodes = [];

        // Global layer (3 nodes) - venue and team IDs
        nodes.push({
            id: 'venue', type: 'venue', layer: 'global',
            features: { venue_id: 5 },
            featureNames: ['venue_id']
        });
        nodes.push({
            id: 'batting_team', type: 'batting_team', layer: 'global',
            features: { team_id: 12 },
            featureNames: ['team_id']
        });
        nodes.push({
            id: 'bowling_team', type: 'bowling_team', layer: 'global',
            features: { team_id: 8 },
            featureNames: ['team_id']
        });

        // State layer
        nodes.push({
            id: 'score_state', type: 'score_state', layer: 'state',
            features: { runs: 0.45, wickets: 0.2, balls: 0.42, innings: 0.0, is_womens: 0.0 },
            featureNames: ['runs_norm', 'wickets_norm', 'balls_norm', 'innings_indicator', 'is_womens_cricket']
        });
        nodes.push({
            id: 'chase_state', type: 'chase_state', layer: 'state',
            features: { runs_needed: 0.0, rrr: 0.0, is_chase: 0.0, rrr_norm: 0.0, difficulty: 0.0, balls_rem: 0.0, wickets_rem: 0.0 },
            featureNames: ['runs_needed', 'rrr', 'is_chase', 'rrr_norm', 'difficulty', 'balls_rem', 'wickets_rem']
        });
        nodes.push({
            id: 'phase_state', type: 'phase_state', layer: 'state',
            features: { is_powerplay: 0.0, is_middle: 1.0, is_death: 0.0, over_progress: 0.67, is_first_ball: 0.0, is_super_over: 0.0 },
            featureNames: ['is_powerplay', 'is_middle', 'is_death', 'over_progress', 'is_first_ball', 'is_super_over']
        });
        nodes.push({
            id: 'time_pressure', type: 'time_pressure', layer: 'state',
            features: { balls_remaining: 0.58, urgency: 0.35, is_final_over: 0.0 },
            featureNames: ['balls_remaining_norm', 'urgency', 'is_final_over']
        });
        nodes.push({
            id: 'wicket_buffer', type: 'wicket_buffer', layer: 'state',
            features: { wickets_in_hand: 0.8, is_tail: 0.0 },
            featureNames: ['wickets_in_hand_norm', 'is_tail']
        });

        // Actor layer (7 nodes) - identity + state + partnership
        // Order matches edge_builder.py LAYER_NODES['actor']

        // striker_identity: player ID with hierarchical fallback
        nodes.push({
            id: 'striker_identity', type: 'striker_identity', layer: 'actor',
            features: { player_id: 142 },
            featureNames: ['player_id'],
            extraAttrs: { team_id: 12, role_id: 1 }
        });
        // striker_state: 7 base features + balls_since_on_strike
        nodes.push({
            id: 'striker_state', type: 'striker_state', layer: 'actor',
            features: { runs: 0.32, balls: 0.28, sr: 0.71, dots_pct: 0.3, is_set: 1.0, boundaries: 0.2, is_debut: 0.0, balls_since: 0.1 },
            featureNames: ['runs_norm', 'balls_faced_norm', 'strike_rate_norm', 'dots_pct', 'is_set', 'boundaries_norm', 'is_debut_ball', 'balls_since_on_strike']
        });
        // nonstriker_identity: player ID with hierarchical fallback
        nodes.push({
            id: 'nonstriker_identity', type: 'nonstriker_identity', layer: 'actor',
            features: { player_id: 87 },
            featureNames: ['player_id'],
            extraAttrs: { team_id: 12, role_id: 2 }
        });
        // nonstriker_state: Z2 symmetric with striker (7 base + balls_since_as_nonstriker)
        nodes.push({
            id: 'nonstriker_state', type: 'nonstriker_state', layer: 'actor',
            features: { runs: 0.45, balls: 0.38, sr: 0.78, dots_pct: 0.25, is_set: 1.0, boundaries: 0.25, is_debut: 0.0, balls_since: 0.05 },
            featureNames: ['runs_norm', 'balls_faced_norm', 'strike_rate_norm', 'dots_pct', 'is_set', 'boundaries_norm', 'is_debut_ball', 'balls_since_as_nonstriker']
        });
        // bowler_identity: player ID with hierarchical fallback
        nodes.push({
            id: 'bowler_identity', type: 'bowler_identity', layer: 'actor',
            features: { player_id: 203 },
            featureNames: ['player_id'],
            extraAttrs: { team_id: 8, role_id: 3 }
        });
        // bowler_state: bowling stats + type indicators
        nodes.push({
            id: 'bowler_state', type: 'bowler_state', layer: 'actor',
            features: { balls: 0.35, runs: 0.28, wickets: 0.2, economy: 0.42, dots_pct: 0.45, threat: 0.55, is_pace: 1.0, is_spin: 0.0 },
            featureNames: ['balls_norm', 'runs_conceded_norm', 'wickets_norm', 'economy_norm', 'dots_pct', 'threat', 'is_pace', 'is_spin']
        });
        // partnership: current batting pair stats
        nodes.push({
            id: 'partnership', type: 'partnership', layer: 'actor',
            features: { runs: 0.28, balls: 0.22, run_rate: 0.68, stability: 0.4 },
            featureNames: ['runs_norm', 'balls_norm', 'run_rate_norm', 'stability']
        });

        // Dynamics layer
        nodes.push({
            id: 'batting_momentum', type: 'batting_momentum', layer: 'dynamics',
            features: { momentum: 0.62 },
            featureNames: ['momentum']
        });
        nodes.push({
            id: 'bowling_momentum', type: 'bowling_momentum', layer: 'dynamics',
            features: { momentum: 0.38 },
            featureNames: ['momentum']
        });
        nodes.push({
            id: 'pressure_index', type: 'pressure_index', layer: 'dynamics',
            features: { pressure: 0.55 },
            featureNames: ['pressure']
        });
        nodes.push({
            id: 'dot_pressure', type: 'dot_pressure', layer: 'dynamics',
            features: { consecutive_dots: 0.3, balls_since_boundary: 0.4, balls_since_wicket: 0.25, pressure_accumulated: 0.45, pressure_trend: 0.1 },
            featureNames: ['consecutive_dots', 'balls_since_boundary', 'balls_since_wicket', 'pressure_accumulated', 'pressure_trend']
        });

        // Query layer - placeholder for learned embedding (initialized to zeros)
        nodes.push({
            id: 'query', type: 'query', layer: 'query',
            features: { learned_embedding: 0.0 },
            featureNames: ['learned_embedding_placeholder']
        });

        return nodes;
    },

    /**
     * Generate sample ball nodes
     * Each ball has 18 features + extra attributes for player IDs
     */
    generateSampleBalls(count) {
        const balls = [];
        const bowlers = ['Bumrah', 'Shami', 'Jadeja'];
        const batsmen = ['Rohit', 'SKY', 'Kishan', 'Hardik'];
        const outcomes = [0, 1, 2, 4, 6];  // Common runs

        for (let i = 0; i < count; i++) {
            const over = Math.floor(i / 6);
            const ballInOver = i % 6;
            const runs = outcomes[Math.floor(Math.random() * outcomes.length)];
            const isWicket = Math.random() < 0.05;
            const bowlerIdx = over % bowlers.length;
            const batsmanIdx = i % 2 === 0 ? 0 : 1;  // Alternating for simplicity
            const isBoundary = runs >= 4;

            balls.push({
                id: `ball_${i}`,
                index: i,
                over: over,
                ballInOver: ballInOver,
                bowler: bowlers[bowlerIdx],
                batsman: batsmen[batsmanIdx],
                nonstriker: batsmen[(batsmanIdx + 1) % 2],
                runs: runs,
                isWicket: isWicket,
                isBoundary: isBoundary,
                // 18 features matching compute_ball_features
                features: {
                    runs_norm: runs / 6,
                    is_wicket: isWicket ? 1 : 0,
                    over_norm: over / 20,
                    ball_in_over_norm: ballInOver / 6,
                    is_boundary: isBoundary ? 1 : 0,
                    // Extras (simplified - all 0 for sample)
                    is_wide: 0, is_noball: 0, is_bye: 0, is_legbye: 0,
                    // Wicket types (simplified)
                    wicket_bowled: 0, wicket_caught: 0, wicket_lbw: 0,
                    wicket_run_out: 0, wicket_stumped: 0, wicket_other: 0,
                    // Run-out attribution
                    striker_run_out: 0, nonstriker_run_out: 0,
                    // Positional
                    bowling_end: over % 2
                },
                featureNames: [
                    'runs_norm', 'is_wicket', 'over_norm', 'ball_in_over_norm', 'is_boundary',
                    'is_wide', 'is_noball', 'is_bye', 'is_legbye',
                    'wicket_bowled', 'wicket_caught', 'wicket_lbw', 'wicket_run_out', 'wicket_stumped', 'wicket_other',
                    'striker_run_out', 'nonstriker_run_out', 'bowling_end'
                ],
                // Extra attributes for embedding lookup (like identity nodes)
                extraAttrs: {
                    bowler_id: bowlerIdx * 100 + 10,
                    batsman_id: batsmanIdx * 100 + 20,
                    nonstriker_id: ((batsmanIdx + 1) % 2) * 100 + 20
                }
            });
        }

        return balls;
    },

    /**
     * Generate edges based on ball nodes
     */
    generateEdges(balls) {
        const edges = {
            hierarchical: [],
            intra: [],
            temporal: {
                recent_precedes: [],
                medium_precedes: [],
                distant_precedes: [],
                same_bowler: [],
                same_batsman: [],
                same_matchup: [],
                same_over: []
            },
            crossdomain: [],
            query: []
        };

        // Hierarchical: Global -> State -> Actor -> Dynamics
        const globalNodes = ['venue', 'batting_team', 'bowling_team'];
        const stateNodes = ['score_state', 'chase_state', 'phase_state', 'time_pressure', 'wicket_buffer'];
        const actorNodes = ['striker_identity', 'striker_state', 'nonstriker_identity', 'nonstriker_state', 'bowler_identity', 'bowler_state', 'partnership'];
        const dynamicsNodes = ['batting_momentum', 'bowling_momentum', 'pressure_index', 'dot_pressure'];

        // Global -> State
        globalNodes.forEach(g => {
            stateNodes.forEach(s => {
                edges.hierarchical.push({ source: g, target: s, type: 'conditions' });
            });
        });

        // State -> Actor
        stateNodes.forEach(s => {
            actorNodes.forEach(a => {
                edges.hierarchical.push({ source: s, target: a, type: 'conditions' });
            });
        });

        // Actor -> Dynamics
        actorNodes.forEach(a => {
            dynamicsNodes.forEach(d => {
                edges.hierarchical.push({ source: a, target: d, type: 'conditions' });
            });
        });

        // Intra-layer edges (bidirectional, matching edge_builder.py)

        // INTRA_LAYER_GLOBAL
        const intraGlobal = [
            ['venue', 'batting_team'],
            ['venue', 'bowling_team'],
            ['batting_team', 'bowling_team']
        ];
        intraGlobal.forEach(([src, tgt]) => {
            edges.intra.push({ source: src, target: tgt, type: 'relates_to' });
            edges.intra.push({ source: tgt, target: src, type: 'relates_to' });
        });

        // INTRA_LAYER_STATE
        const intraState = [
            ['score_state', 'chase_state'],
            ['score_state', 'phase_state'],
            ['score_state', 'time_pressure'],
            ['score_state', 'wicket_buffer'],
            ['chase_state', 'time_pressure'],
            ['phase_state', 'time_pressure'],
            ['time_pressure', 'wicket_buffer']
        ];
        intraState.forEach(([src, tgt]) => {
            edges.intra.push({ source: src, target: tgt, type: 'relates_to' });
            edges.intra.push({ source: tgt, target: src, type: 'relates_to' });
        });

        // INTRA_LAYER_ACTOR (uses 'matchup' relation)
        const intraActor = [
            // Identity to state connections
            ['striker_identity', 'striker_state'],
            ['nonstriker_identity', 'nonstriker_state'],
            ['bowler_identity', 'bowler_state'],
            // THE KEY MATCHUP: striker vs bowler
            ['striker_identity', 'bowler_identity'],
            // Non-striker matchups (run-out risk, strike rotation)
            ['nonstriker_identity', 'bowler_identity'],
            ['striker_identity', 'nonstriker_identity'],
            // Partnership connections
            ['striker_state', 'partnership'],
            ['nonstriker_state', 'partnership'],
            ['bowler_state', 'partnership'],
            ['striker_identity', 'partnership'],
            ['nonstriker_identity', 'partnership']
        ];
        intraActor.forEach(([src, tgt]) => {
            edges.intra.push({ source: src, target: tgt, type: 'matchup' });
            edges.intra.push({ source: tgt, target: src, type: 'matchup' });
        });

        // INTRA_LAYER_DYNAMICS
        const intraDynamics = [
            ['batting_momentum', 'bowling_momentum'],
            ['batting_momentum', 'pressure_index'],
            ['bowling_momentum', 'pressure_index'],
            ['pressure_index', 'dot_pressure']
        ];
        intraDynamics.forEach(([src, tgt]) => {
            edges.intra.push({ source: src, target: tgt, type: 'relates_to' });
            edges.intra.push({ source: tgt, target: src, type: 'relates_to' });
        });

        // Temporal edges between balls (matching build_temporal_edges in edge_builder.py)
        const numBalls = balls.length;
        const maxTemporalDistance = 120;  // T20 = 120 balls
        const spellWindow = 24.0;  // ~4 overs - typical bowler spell
        const inningsWindow = 60.0;  // ~10 overs - batsman form window

        // 1. Multi-scale precedes edges (CAUSAL: older -> newer)
        for (let i = 0; i < numBalls; i++) {
            for (let j = i + 1; j < numBalls; j++) {
                const gap = j - i;

                if (gap <= 6) {
                    // Recent: within current over context
                    edges.temporal.recent_precedes.push({
                        source: `ball_${i}`, target: `ball_${j}`,
                        type: 'recent_precedes', decay: gap / 6.0
                    });
                } else if (gap <= 24) {
                    // Medium: 4-over spell window
                    edges.temporal.medium_precedes.push({
                        source: `ball_${i}`, target: `ball_${j}`,
                        type: 'medium_precedes', decay: (gap - 6) / 18.0
                    });
                } else {
                    // Distant: sparse connections every 6 balls for efficiency
                    if (gap % 6 === 0) {
                        edges.temporal.distant_precedes.push({
                            source: `ball_${i}`, target: `ball_${j}`,
                            type: 'distant_precedes', decay: (gap - 24) / maxTemporalDistance
                        });
                    }
                }
            }
        }

        // 2. Same bowler edges (BIDIRECTIONAL with temporal decay)
        const bowlerToBalls = {};
        balls.forEach((ball, idx) => {
            const bowlerId = ball.extraAttrs?.bowler_id || ball.bowler;
            if (!bowlerToBalls[bowlerId]) bowlerToBalls[bowlerId] = [];
            bowlerToBalls[bowlerId].push(idx);
        });

        for (const ballIndices of Object.values(bowlerToBalls)) {
            if (ballIndices.length > 1) {
                for (let i = 0; i < ballIndices.length; i++) {
                    for (let j = i + 1; j < ballIndices.length; j++) {
                        const temporalDist = Math.abs(ballIndices[j] - ballIndices[i]) / spellWindow;
                        // Forward edge
                        edges.temporal.same_bowler.push({
                            source: `ball_${ballIndices[i]}`, target: `ball_${ballIndices[j]}`,
                            type: 'same_bowler', decay: temporalDist
                        });
                        // Backward edge
                        edges.temporal.same_bowler.push({
                            source: `ball_${ballIndices[j]}`, target: `ball_${ballIndices[i]}`,
                            type: 'same_bowler', decay: temporalDist
                        });
                    }
                }
            }
        }

        // 3. Same batsman edges (BIDIRECTIONAL with temporal decay)
        const batsmanToBalls = {};
        balls.forEach((ball, idx) => {
            const batsmanId = ball.extraAttrs?.batsman_id || ball.batsman;
            if (!batsmanToBalls[batsmanId]) batsmanToBalls[batsmanId] = [];
            batsmanToBalls[batsmanId].push(idx);
        });

        for (const ballIndices of Object.values(batsmanToBalls)) {
            if (ballIndices.length > 1) {
                for (let i = 0; i < ballIndices.length; i++) {
                    for (let j = i + 1; j < ballIndices.length; j++) {
                        const temporalDist = Math.abs(ballIndices[j] - ballIndices[i]) / inningsWindow;
                        // Forward edge
                        edges.temporal.same_batsman.push({
                            source: `ball_${ballIndices[i]}`, target: `ball_${ballIndices[j]}`,
                            type: 'same_batsman', decay: temporalDist
                        });
                        // Backward edge
                        edges.temporal.same_batsman.push({
                            source: `ball_${ballIndices[j]}`, target: `ball_${ballIndices[i]}`,
                            type: 'same_batsman', decay: temporalDist
                        });
                    }
                }
            }
        }

        // 4. Same matchup edges (CAUSAL: older -> newer only)
        const matchupToBalls = {};
        balls.forEach((ball, idx) => {
            const bowlerId = ball.extraAttrs?.bowler_id || ball.bowler;
            const batsmanId = ball.extraAttrs?.batsman_id || ball.batsman;
            const matchupKey = `${bowlerId}_${batsmanId}`;
            if (!matchupToBalls[matchupKey]) matchupToBalls[matchupKey] = [];
            matchupToBalls[matchupKey].push(idx);
        });

        for (const ballIndices of Object.values(matchupToBalls)) {
            if (ballIndices.length > 1) {
                const sorted = [...ballIndices].sort((a, b) => a - b);
                for (let i = 0; i < sorted.length; i++) {
                    for (let j = i + 1; j < sorted.length; j++) {
                        // Only older -> newer direction (causal)
                        edges.temporal.same_matchup.push({
                            source: `ball_${sorted[i]}`, target: `ball_${sorted[j]}`,
                            type: 'same_matchup'
                        });
                    }
                }
            }
        }

        // 5. Same over edges (CAUSAL with ball-in-over position)
        // Matches build_same_over_edges in edge_builder.py
        const overToBalls = {};
        balls.forEach((ball, idx) => {
            if (!overToBalls[ball.over]) overToBalls[ball.over] = [];
            overToBalls[ball.over].push({ idx, ballInOver: ball.ballInOver });
        });

        for (const ballsInOver of Object.values(overToBalls)) {
            if (ballsInOver.length > 1) {
                const sorted = [...ballsInOver].sort((a, b) => a.idx - b.idx);
                for (let i = 0; i < sorted.length; i++) {
                    for (let j = i + 1; j < sorted.length; j++) {
                        // Causal: older -> newer with position attribute
                        // Clamp to 1.0 max since overs can have >6 deliveries due to wides/no-balls
                        const tgtPosition = Math.min(sorted[j].ballInOver / 5.0, 1.0);
                        edges.temporal.same_over.push({
                            source: `ball_${sorted[i].idx}`, target: `ball_${sorted[j].idx}`,
                            type: 'same_over', position: tgtPosition
                        });
                    }
                }
            }
        }

        // Cross-domain: balls to context (matching build_cross_domain_edges in edge_builder.py)
        // These edges connect historical balls to CURRENT players only
        // - faced_by: balls faced by CURRENT striker -> striker_identity
        // - partnered_by: balls where CURRENT non-striker was partner OR was batting -> nonstriker_identity
        // - bowled_by: balls bowled by CURRENT bowler -> bowler_identity
        // - informs: recent_k balls -> all dynamics nodes

        // For sample data, simulate "current" players from the last ball
        const lastBall = balls[balls.length - 1];
        const currentStrikerId = lastBall?.extraAttrs?.batsman_id || lastBall?.batsman;
        const currentBowlerId = lastBall?.extraAttrs?.bowler_id || lastBall?.bowler;
        const currentNonstrikerId = lastBall?.extraAttrs?.nonstriker_id || lastBall?.nonstriker;

        // faced_by: Only balls actually faced by the CURRENT striker
        balls.forEach(ball => {
            const ballBatsmanId = ball.extraAttrs?.batsman_id || ball.batsman;
            if (ballBatsmanId === currentStrikerId) {
                edges.crossdomain.push({
                    source: ball.id, target: 'striker_identity',
                    type: 'faced_by'
                });
            }
        });

        // partnered_by: Balls where CURRENT non-striker was at non-striker end OR was batting
        balls.forEach(ball => {
            const ballNonstrikerId = ball.extraAttrs?.nonstriker_id || ball.nonstriker;
            const ballBatsmanId = ball.extraAttrs?.batsman_id || ball.batsman;
            if (ballNonstrikerId === currentNonstrikerId || ballBatsmanId === currentNonstrikerId) {
                edges.crossdomain.push({
                    source: ball.id, target: 'nonstriker_identity',
                    type: 'partnered_by'
                });
            }
        });

        // bowled_by: Only balls actually bowled by the CURRENT bowler
        balls.forEach(ball => {
            const ballBowlerId = ball.extraAttrs?.bowler_id || ball.bowler;
            if (ballBowlerId === currentBowlerId) {
                edges.crossdomain.push({
                    source: ball.id, target: 'bowler_identity',
                    type: 'bowled_by'
                });
            }
        });

        // informs: Recent balls (last recent_k=12) inform ALL dynamics nodes
        const recentK = 12;
        const recentBalls = balls.slice(-recentK);
        recentBalls.forEach(ball => {
            dynamicsNodes.forEach(d => {
                edges.crossdomain.push({
                    source: ball.id, target: d,
                    type: 'informs'
                });
            });
        });

        // Query edges (matching edge_builder.py)
        // All context nodes -> query via 'attends'
        const allContextNodes = [...globalNodes, ...stateNodes, ...actorNodes, ...dynamicsNodes];
        allContextNodes.forEach(node => {
            edges.query.push({ source: node, target: 'query', type: 'attends' });
        });

        // Balls -> query via 'attends'
        balls.forEach(ball => {
            edges.query.push({ source: ball.id, target: 'query', type: 'attends' });
        });

        // Dynamics -> query via 'drives' (separate relation for momentum influence)
        dynamicsNodes.forEach(d => {
            edges.query.push({ source: d, target: 'query', type: 'drives' });
        });

        return edges;
    },

    /**
     * Get edge counts by category
     */
    getEdgeCounts() {
        if (!this.data) return {};

        const counts = {
            hierarchical: this.data.edges.hierarchical.length,
            intra: this.data.edges.intra.length,
            temporal: Object.values(this.data.edges.temporal).reduce((sum, arr) => sum + arr.length, 0),
            crossdomain: this.data.edges.crossdomain.length,
            query: this.data.edges.query.length
        };

        // Temporal subtypes
        counts.temporalSubtypes = {};
        for (const [type, arr] of Object.entries(this.data.edges.temporal)) {
            counts.temporalSubtypes[type] = arr.length;
        }

        return counts;
    }
};
