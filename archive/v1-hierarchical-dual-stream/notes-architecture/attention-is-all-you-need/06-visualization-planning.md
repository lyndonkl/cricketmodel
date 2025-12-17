# Attention Manifold Visualization - Planning Notes

## 1. Evaluation of Rough Sketch

### User's Original Concept

1. **Start**: 3D space with points everywhere (word embeddings in a room)
2. **Transformation**: As sentence is formed, space gets transformed (pulled, rotated, sheared)
3. **Hypersphere**: Bring relevant data into a small hypersphere representing context window
4. **Path**: Continuation follows a path on this hypersphere
5. **Challenge**: Constructing the best hypersphere that produces valid results

### Technical Assessment

| Concept | Technical Accuracy | Visualization Suitability |
|---------|-------------------|---------------------------|
| Points as embeddings | Accurate - tokens map to vectors | Excellent - intuitive starting point |
| Space transformation | Accurate - attention transforms manifold | Good - can animate transforms |
| Hypersphere as context | Partially accurate* | Good metaphor for audience |
| Path for generation | Metaphorically accurate | Excellent - captures sequential nature |

**\*Technical nuance**: The "hypersphere" concept conflates two related but distinct ideas:

1. **LayerNorm hypersphere**: After normalization, vectors have approximately equal magnitude (~sqrt(d_model)), placing them near a hypersphere surface. This IS accurate.

2. **Context window**: The attention context window (which tokens can attend to which) is about masking/sequence position, not spatial geometry.

**Recommendation**: The hypersphere metaphor is pedagogically powerful for non-technical audiences. We should use it while being accurate about what it represents: **the normalized representation space where context accumulates**.

### Refined Concept

**Metaphor refinement for accuracy + accessibility**:

1. **The Library** (embedding space): Vast 3D space with books (word embeddings) scattered throughout - organized by meaning but incomprehensibly large
2. **The Prompt as a Magnet**: Your prompt creates a "gravitational field" that pulls relevant books closer
3. **The Context Sphere**: As attention operates, relevant information concentrates into a "working sphere" - the context
4. **The Reading Path**: Generation traces a path through this sphere, each step consulting the accumulated context
5. **The Key Insight**: Better prompts = stronger, more focused magnetic field = more relevant context sphere

---

## 2. Narrative Design (Communication-Storytelling Framework)

### Audience Profile

- **Who**: Non-technical stakeholders (executives, product managers, marketing, customers)
- **Expertise level**: No ML background, may know "AI" and "prompts" conceptually
- **Core question**: "Why does how I phrase my prompt matter so much?"
- **Time constraints**: 3-5 minute attention span for explanation
- **Decision authority**: May be deciding on AI adoption, prompt training, or understanding capabilities

### Narrative Structure: Before-After-Bridge

**BEFORE (The Problem)**
> You ask the AI a question. Sometimes it gives brilliant answers. Sometimes it completely misses the point. Why? The AI has access to vast knowledge, but HOW you ask determines WHAT it can access.

**AFTER (The Desired State)**
> When you understand how the AI "sees" your prompt, you can craft prompts that reliably pull in exactly the right knowledge - like tuning a radio to the right frequency.

**BRIDGE (The Explanation)**
> The visualization will show how attention transforms a vast space of knowledge into a focused context sphere.

### Scene-by-Scene Narrative Arc

#### Scene 1: "The Library of Everything" (10 seconds)
**What user sees**: Thousands of points floating in 3D space, different colors for different topics
**Narration**: "Imagine all human knowledge as points in space. Similar concepts cluster together. This is what the AI 'knows' - but it's overwhelming."
**Emotion**: Awe, slight overwhelm

#### Scene 2: "Your Words Enter" (15 seconds)
**What user sees**: Prompt words appear one by one, each creating ripples in the space
**Narration**: "When you type a prompt, each word creates a 'pull' - attracting related concepts."
**Emotion**: Curiosity, engagement

#### Scene 3: "The Transformation Begins" (20 seconds)
**What user sees**: Space warps, rotates, relevant points move closer, irrelevant fade
**Narration**: "This is ATTENTION - the space literally transforms to bring relevant information together. Different words attend to different parts of knowledge."
**Emotion**: Understanding dawning

#### Scene 4: "The Context Sphere Forms" (15 seconds)
**What user sees**: Points coalesce into a tight sphere, normalized and organized
**Narration**: "Your entire prompt creates a 'context sphere' - a concentrated region of relevant knowledge that the AI will use to respond."
**Emotion**: Clarity, insight

#### Scene 5: "The Path of Generation" (20 seconds)
**What user sees**: A path traces through the sphere, each point illuminating nearby context
**Narration**: "Now generation begins. Each new word is predicted by consulting this context sphere. The path follows what makes sense given everything gathered."
**Emotion**: Satisfaction, comprehension

#### Scene 6: "The Prompt Engineering Challenge" (10 seconds)
**What user sees**: Split view - weak prompt (sparse sphere) vs strong prompt (dense sphere)
**Narration**: "THIS is why prompt engineering matters. Better prompts create better context spheres. Same AI, same knowledge - but you control what it can access."
**Emotion**: Empowerment, motivation

### Key Message (Headline)

> **"Your prompt is a lens - it focuses the AI's vast knowledge into a usable context sphere. Better prompts create sharper focus."**

---

## 3. Visual Design Decisions

### Color Palette

| Element | Color | Meaning |
|---------|-------|---------|
| Embedding points (inactive) | Gray (#888) | Raw, unfocused knowledge |
| Embedding points (active) | Gold (#FFD700) | Relevant, attended information |
| Query/Prompt words | Cyan (#00CED1) | User input |
| Attention beams | Gradient (cyan → gold) | Attention flow |
| Context sphere boundary | White, semi-transparent | The focused context |
| Generation path | Bright green (#00FF00) | Output trajectory |
| Background | Dark (#1a1a2e) | Focus on content |

### Animation Principles

1. **Easing**: Use `power2.inOut` for smooth, natural-feeling transforms
2. **Duration**: 1-2 seconds per major transition (slow enough to follow)
3. **Staging**: Never move more than one concept at a time
4. **Anticipation**: Slight pause before major transforms

### Interaction Design

| Control | Function |
|---------|----------|
| Play/Pause button | Start/stop automatic progression |
| Step buttons (< >) | Move between scenes manually |
| Scene indicator | Show current scene (dots) |
| Reset button | Return to beginning |
| Speed slider | Adjust animation speed |

### 3D Considerations

- Use THREE.js via D3's 3D capabilities OR pure D3 with perspective transforms
- Keep camera movements slow and predictable
- Provide visual anchors (grid lines, axis indicators)
- Avoid nauseating rotations

---

## 4. Technical Implementation Plan

### Technology Stack

- **D3.js v7**: Data binding, SVG manipulation, force simulations
- **GSAP (GreenSock)**: Timeline animations, complex easing, morphing
- **Three.js** (optional): True 3D rendering if needed
- **HTML/CSS**: UI controls, layout

### File Structure

```
notes/architecture/attention-is-all-you-need/
  06-visualization-planning.md  (this file)
  attention-manifold-visualization.html  (the visualization)
```

### D3 + GSAP Integration Pattern

```javascript
// D3 handles data binding and DOM
const points = svg.selectAll('.point')
  .data(embeddings)
  .enter()
  .append('circle')
  .attr('class', 'point');

// GSAP handles animations
gsap.to('.point', {
  duration: 2,
  attr: { cx: d => newX(d), cy: d => newY(d) },
  ease: 'power2.inOut',
  stagger: 0.01
});
```

### Scene Implementation

Each scene will be a function that:
1. Sets up target state for all elements
2. Creates GSAP timeline for transitions
3. Returns timeline for sequencing

```javascript
const scenes = {
  library: createLibraryScene,
  wordsEnter: createWordsEnterScene,
  transformation: createTransformationScene,
  sphereForm: createSphereFormScene,
  generation: createGenerationScene,
  comparison: createComparisonScene
};
```

---

## 5. Implementation Checklist

- [ ] Create base HTML structure with controls
- [ ] Set up D3 SVG canvas with responsive sizing
- [ ] Generate initial embedding point cloud
- [ ] Implement Scene 1: Library visualization
- [ ] Implement Scene 2: Word entry animation
- [ ] Implement Scene 3: Attention transformation
- [ ] Implement Scene 4: Sphere formation (LayerNorm metaphor)
- [ ] Implement Scene 5: Generation path
- [ ] Implement Scene 6: Comparison view
- [ ] Add play/pause/step controls
- [ ] Add scene indicators
- [ ] Test performance with 500+ points
- [ ] Add explanatory text overlays
- [ ] Mobile responsiveness

---

## 6. Notes for Reference During Implementation

### Key Technical Concepts to Visualize

1. **Attention as weighted averaging**: Points move toward weighted centroid
2. **Multi-head attention**: Show multiple simultaneous "pulls" (different colors)
3. **LayerNorm**: All points normalize to similar distance from origin (sphere surface)
4. **Softmax**: Attention weights sum to 1 (show as percentage/brightness)
5. **Causal masking**: Generation only "sees" previous tokens (show as cone of visibility)

### Simplifications for Clarity

1. Reduce 512 dimensions to 3D (use PCA-like projection metaphor)
2. Reduce vocabulary to ~100 visible points
3. Use categorical colors instead of continuous gradients where possible
4. Exaggerate transformations for visibility

### User Control Philosophy

The visualization should work in two modes:
1. **Presentation mode**: Auto-plays with narration timing
2. **Exploration mode**: User controls every step, can pause and examine

---

## 7. Success Criteria

After viewing the visualization, stakeholders should be able to:

1. Explain why "how you phrase a prompt matters" in their own words
2. Use the "context sphere" metaphor in conversations
3. Appreciate that prompt engineering is about "focusing the AI's attention"
4. Feel confident discussing AI capabilities without technical jargon

---

## 8. Skill-Informed Design Decisions

### From Cognitive Design Skill

**Applied Principles:**

| Principle | Application |
|-----------|-------------|
| **Working Memory (4±1)** | Max 6 scenes, each focused on ONE concept |
| **Progressive Disclosure** | Overview first (all points), then zoom (transformation), then details (path) |
| **Preattentive Salience** | Use color/motion ONLY for key moments (prompt entry, sphere formation) |
| **Visual Hierarchy** | Primary: current scene visualization; Secondary: controls; Tertiary: narration |
| **Recognition over Recall** | Scene indicators show current position; controls always visible |
| **Immediate Feedback** | Button presses have instant visual response |
| **Spatial Contiguity** | Narration text positioned near relevant visual elements |

**3-Question Quick Check:**
1. ✓ **Attention**: Clear focal point (center sphere), predictable scanning
2. ✓ **Memory**: Scene indicators show state, no hidden modes
3. ✓ **Clarity**: 5-second test passes - each scene has one clear message

### From D3 Visualization Skill

**Applied Patterns:**

```javascript
// Easing: Natural feel with slow finish
.ease(d3.easeCubicOut)

// Staggered entry: Points appear in sequence
.delay((d, i) => i * 10)

// Object constancy: Track points through transforms
.data(points, d => d.id)

// Named transitions: Prevent conflicts
.transition('sphere-form')
```

**Interaction Model:**
- Zoom disabled (we control the view for storytelling)
- Click/tap for play/pause only
- Keyboard support (Space = play/pause, Arrow keys = step)

### From Communication-Storytelling Skill

**Narrative Structure: Before-After-Bridge**

| Stage | Scene | Emotion Target |
|-------|-------|----------------|
| BEFORE | Scene 1-2 | Overwhelm → Curiosity |
| BRIDGE | Scene 3-4 | Understanding → Insight |
| AFTER | Scene 5-6 | Clarity → Empowerment |

**Headline (to reinforce throughout):**
> "Your prompt shapes the context sphere - better prompts create sharper focus"

---

## 9. Final Implementation Notes

### Animation Timing

| Scene | Duration | Transition Type |
|-------|----------|-----------------|
| 1. Library | 3s | Fade in points (staggered 2ms each) |
| 2. Words Enter | 4s | Sequential word appearance + ripple |
| 3. Transform | 5s | Smooth point repositioning |
| 4. Sphere Form | 4s | Points converge + normalize |
| 5. Path | 5s | Line traces through sphere |
| 6. Compare | 3s | Split view morph |

### Performance Targets

- 500 points maximum
- 60fps during animations
- < 100ms initial load
- Mobile-friendly (touch support)

---

---

## 10. Cognitive Design Validation (Post-Implementation)

### 3-Question Quick Check

#### Question 1: Attention - "Is it obvious what to look at first?"
- [x] **Visual hierarchy is clear**: 3D point cloud is primary, controls secondary, narration tertiary
- [x] **Most important element is preattentively salient**: Gold/bright points vs dim gray creates immediate pop-out
- [x] **Predictable scanning**: Center focus (sphere), then bottom (narration), then corners (controls/stats)

**Result: PASS**

#### Question 2: Memory - "Is user required to remember anything?"
- [x] **Current state visible**: Scene dots show position, prompt text shows current prompt, stats panel shows metrics
- [x] **Options presented**: All controls always visible (prev/next/play/reset)
- [x] **4±1 chunks**: 6 scenes, 3 stats, 5 legend items - all within working memory limits

**Result: PASS**

#### Question 3: Clarity - "Can someone unfamiliar understand in 5 seconds?"
- [x] **Purpose graspable**: Title + subtitle + narration make concept clear
- [x] **No unnecessary decoration**: Grid serves purpose (shows warping), not decorative
- [x] **Familiar terminology**: "Knowledge space", "context sphere", "attention" - accessible metaphors

**Result: PASS**

### Applied Cognitive Principles

| Principle | Implementation |
|-----------|----------------|
| **Preattentive Salience** | Gold points vs gray; warping grid changes color when active |
| **Progressive Disclosure** | 6 scenes reveal concept step by step |
| **Dual Coding** | Visual (3D animation) + Verbal (narration text) |
| **Comparison** | Plain vs Structured shown sequentially for direct comparison |
| **Spatial Contiguity** | Stats panel updates alongside visualization |
| **Immediate Feedback** | Scene changes trigger instant visual transformation |

### Visual Elements Added for Space Transformation

1. **Warp Grid**: 3D grid lines that deform toward center during attention
2. **Flow Lines**: Curved paths showing knowledge being "pulled" into context sphere
3. **Point Movement**: Relevant points physically move toward center
4. **Color Intensity**: Grid brightens as warp strength increases
5. **Size Changes**: Relevant points grow, irrelevant points shrink

---

*Implementation complete. Refresh browser to see changes.*
