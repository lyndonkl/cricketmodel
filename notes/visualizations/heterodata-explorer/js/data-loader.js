/**
 * Data Loader - Manages graph data structure and loading
 */

const DataLoader = {
    // Current loaded graph data
    data: null,

    // Node type metadata with layer assignments
    nodeTypes: {
        // Global/Entity layer (6 nodes) - all ID-based embedding nodes
        venue: { layer: 'global', index: 0, featureDim: 1, description: 'Match venue ID' },
        batting_team: { layer: 'global', index: 1, featureDim: 1, description: 'Batting team ID' },
        bowling_team: { layer: 'global', index: 2, featureDim: 1, description: 'Bowling team ID' },
        // Identity nodes have .x (player_id) + .team_id + .role_id for hierarchical fallback
        striker_identity: { layer: 'global', index: 3, featureDim: 1, extraAttrs: ['team_id', 'role_id'], description: 'Striker player ID + team/role fallback' },
        nonstriker_identity: { layer: 'global', index: 4, featureDim: 1, extraAttrs: ['team_id', 'role_id'], description: 'Non-striker player ID + team/role fallback' },
        bowler_identity: { layer: 'global', index: 5, featureDim: 1, extraAttrs: ['team_id', 'role_id'], description: 'Bowler player ID + team/role fallback' },

        // State layer (5 nodes)
        score_state: { layer: 'state', index: 0, featureDim: 5, description: 'Current score, wickets, balls' },
        chase_state: { layer: 'state', index: 1, featureDim: 7, description: 'Chase target, RRR, difficulty' },
        phase_state: { layer: 'state', index: 2, featureDim: 6, description: 'Powerplay, middle, death phase' },
        time_pressure: { layer: 'state', index: 3, featureDim: 3, description: 'Balls remaining, urgency, is_final_over' },
        wicket_buffer: { layer: 'state', index: 4, featureDim: 2, description: 'Wickets in hand, is_tail indicator' },

        // Actor State layer (4 nodes) - computed state features for actors
        striker_state: { layer: 'actor', index: 0, featureDim: 8, description: 'Striker: runs, balls, SR, dots_pct, is_set, boundaries, is_debut, balls_since' },
        nonstriker_state: { layer: 'actor', index: 1, featureDim: 8, description: 'Non-striker: Z2 symmetric with striker' },
        bowler_state: { layer: 'actor', index: 2, featureDim: 8, description: 'Bowler: balls, runs, wickets, economy, dots_pct, threat, is_pace, is_spin' },
        partnership: { layer: 'actor', index: 3, featureDim: 4, description: 'Partnership: runs, balls, run_rate, stability' },

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

    // Layer metadata
    layers: {
        global: { name: 'Global', color: '#4a90d9', nodes: ['venue', 'batting_team', 'bowling_team', 'striker_identity', 'nonstriker_identity', 'bowler_identity'] },
        state: { name: 'State', color: '#50c878', nodes: ['score_state', 'chase_state', 'phase_state', 'time_pressure', 'wicket_buffer'] },
        actor: { name: 'Actor', color: '#f5a623', nodes: ['striker_state', 'nonstriker_state', 'bowler_state', 'partnership'] },
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
     */
    generateContextNodes() {
        const nodes = [];

        // Global/Entity layer (6 nodes) - all ID-based embedding nodes
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
        // Identity nodes have .x (player_id) + extra attributes (team_id, role_id) for hierarchical fallback
        nodes.push({
            id: 'striker_identity', type: 'striker_identity', layer: 'global',
            features: { player_id: 142 },
            featureNames: ['player_id'],
            extraAttrs: { team_id: 12, role_id: 1 }  // Hierarchical fallback: team embedding, role embedding
        });
        nodes.push({
            id: 'nonstriker_identity', type: 'nonstriker_identity', layer: 'global',
            features: { player_id: 87 },
            featureNames: ['player_id'],
            extraAttrs: { team_id: 12, role_id: 2 }  // Hierarchical fallback: team embedding, role embedding
        });
        nodes.push({
            id: 'bowler_identity', type: 'bowler_identity', layer: 'global',
            features: { player_id: 203 },
            featureNames: ['player_id'],
            extraAttrs: { team_id: 8, role_id: 3 }  // Hierarchical fallback: team embedding, role embedding
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

        // Actor State layer (4 nodes) - computed state features for actors
        // striker_state: 7 base features + balls_since_on_strike
        nodes.push({
            id: 'striker_state', type: 'striker_state', layer: 'actor',
            features: { runs: 0.32, balls: 0.28, sr: 0.71, dots_pct: 0.3, is_set: 1.0, boundaries: 0.2, is_debut: 0.0, balls_since: 0.1 },
            featureNames: ['runs_norm', 'balls_faced_norm', 'strike_rate_norm', 'dots_pct', 'is_set', 'boundaries_norm', 'is_debut_ball', 'balls_since_on_strike']
        });
        // nonstriker_state: Z2 symmetric with striker (7 base + balls_since_as_nonstriker)
        nodes.push({
            id: 'nonstriker_state', type: 'nonstriker_state', layer: 'actor',
            features: { runs: 0.45, balls: 0.38, sr: 0.78, dots_pct: 0.25, is_set: 1.0, boundaries: 0.25, is_debut: 0.0, balls_since: 0.05 },
            featureNames: ['runs_norm', 'balls_faced_norm', 'strike_rate_norm', 'dots_pct', 'is_set', 'boundaries_norm', 'is_debut_ball', 'balls_since_as_nonstriker']
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

        // Intra-layer: matchup
        edges.intra.push({ source: 'striker_state', target: 'bowler_state', type: 'matchup' });
        edges.intra.push({ source: 'striker_state', target: 'partnership', type: 'relates_to' });
        edges.intra.push({ source: 'nonstriker_state', target: 'partnership', type: 'relates_to' });

        // Temporal edges between balls
        const numBalls = balls.length;
        for (let i = 0; i < numBalls; i++) {
            for (let j = i + 1; j < numBalls; j++) {
                const gap = j - i;

                // Precedes edges (from older to newer)
                if (gap <= 3) {
                    edges.temporal.recent_precedes.push({
                        source: `ball_${i}`, target: `ball_${j}`,
                        type: 'recent_precedes', decay: 1 / gap
                    });
                } else if (gap <= 10) {
                    edges.temporal.medium_precedes.push({
                        source: `ball_${i}`, target: `ball_${j}`,
                        type: 'medium_precedes', decay: 0.5 / gap
                    });
                } else if (gap <= 20) {
                    edges.temporal.distant_precedes.push({
                        source: `ball_${i}`, target: `ball_${j}`,
                        type: 'distant_precedes', decay: 0.25 / gap
                    });
                }

                // Same bowler/batsman/matchup (sample, not exhaustive)
                if (balls[i].bowlerId === balls[j].bowlerId && gap <= 12) {
                    edges.temporal.same_bowler.push({
                        source: `ball_${i}`, target: `ball_${j}`,
                        type: 'same_bowler'
                    });
                }
                if (balls[i].batsmanId === balls[j].batsmanId && gap <= 12) {
                    edges.temporal.same_batsman.push({
                        source: `ball_${i}`, target: `ball_${j}`,
                        type: 'same_batsman'
                    });
                }

                // Same over
                if (balls[i].over === balls[j].over) {
                    edges.temporal.same_over.push({
                        source: `ball_${i}`, target: `ball_${j}`,
                        type: 'same_over', position: balls[j].ballInOver / 5
                    });
                }
            }
        }

        // Cross-domain: balls to context (sample - last 10 balls)
        const recentBalls = balls.slice(-10);
        recentBalls.forEach(ball => {
            // faced_by: striker faced this ball
            edges.crossdomain.push({
                source: ball.id, target: 'striker_state',
                type: 'faced_by'
            });
            // bowled_by: bowler bowled this ball
            edges.crossdomain.push({
                source: ball.id, target: 'bowler_state',
                type: 'bowled_by'
            });
            // informs: ball informs all dynamics
            dynamicsNodes.forEach(d => {
                edges.crossdomain.push({
                    source: ball.id, target: d,
                    type: 'informs'
                });
            });
        });

        // Query edges
        // Attends to dynamics
        dynamicsNodes.forEach(d => {
            edges.query.push({ source: 'query', target: d, type: 'attends' });
        });
        // Attends to actor states
        ['striker_state', 'bowler_state', 'partnership'].forEach(a => {
            edges.query.push({ source: 'query', target: a, type: 'attends' });
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
