### High‑level goal

You don’t need a perfect generative model of the field; you need something that:
- **Respects projected CPT / FLEX ownership and salary caps**
- **Roughly matches stack / position tendencies of the real field**
- **Is cheap and easy to calibrate with flashbacks**

Below are a few **simple, incremental approaches** you can mix and match, ordered from easiest to more sophisticated.

---

### 1. Ownership‑driven “independent” sampler with sanity constraints

This is the minimum viable synthetic field.

- **Step 1 – Sample CPT by projected CPT ownership**
  - Build \(p_\text{CPT}(i) \propto \text{cpt\_own}_i\) from your Sabersim ownership.
  - For each field lineup: draw a CPT index \(i\) from that categorical.

- **Step 2 – Sample FLEX using flex ownership, with basic rules**
  - Build \(p_\text{FLEX}(i) \propto \text{flex\_own}_i\).
  - For each of the 5 flex slots, sample with:
    - **No duplicates** of same player across slots.
    - **Salary cap**: reject + resample if total salary > 50k.
    - **Roster rules**: at least 1 player from each team, max 4 from a single team, etc.

- **Step 3 – Use rejection sampling to enforce “field‑ish” patterns**
  - Reject lineups that violate super‑basic heuristics you see in flashbacks:
    - Too many QBs (e.g. 2 QBs without their receivers),
    - No correlation with CPT team (e.g. QB CPT with 0 receivers from his team),
    - Extreme punt overuse that almost never shows up in real fields.

**Pros:** very easy to wire up; directly uses your existing ownership.  
**Cons:** still somewhat “independent”, doesn’t capture structure like “if QB CPT then more opposing WRs”.

---

### 2. Conditional sampler with simple “archetype” templates

You can get a lot of realism by conditioning on **CPT archetype** and using a few hand‑built templates.

- **Define CPT archetypes**
  - `QB_CPT`, `WR_CPT`, `RB_CPT`, `TE_CPT`, `DST_CPT`, etc.
  - From ownership, compute \(p(\text{archetype})\) for the field (e.g. share of CPT proj ownership going to each position).

- **For each archetype, define a small set of templates**
  - Example for `WR_CPT`:
    - **Template A:** `WR CPT + own QB + 2 opp pass + 2 misc`
    - **Template B:** `WR CPT + own QB + 1 opp pass + 3 misc`
  - Example for `RB_CPT`:
    - **Template A:** `RB CPT + own QB fade + 2 opp pass + 3 misc`
    - **Template B:** `RB CPT + own QB + 1 opp pass + 3 misc`
  - Calibrate these with your flashback `Stack` label and player‑summary sheets.

- **Sampling loop**
  1. Sample a CPT player \(i\) from \(p_\text{CPT}\).
  2. Look up CPT position → archetype.
  3. Sample a template conditioned on that archetype.
  4. Fill slots from position‑filtered ownership distributions (e.g. “opp WRs with high proj” for “opp pass” slot).
  5. Enforce salary cap / DK rules; resample if needed.

This gives you Gibbs‑CPT lineups that look like the kinds of Gibbs CPT lineups the actual field plays, not just “Gibbs + 5 random dudes.”

---

### 3. Mixture of “rec field” and “sharp sub‑field”

You already see in flashbacks that sharp entrants:
- Overweight RB/WR at CPT vs the average field
- Use particular stack patterns more aggressively

So explicitly model **two populations**:

- **Population A – Mass field**
  - CPT / FLEX ownership = raw Sabersim projections.
  - Templates closer to “balanced / common” structures.

- **Population B – Sharps**
  - Build an **adjusted ownership** profile using flashback stats:
    - Increase RB/WR CPT shares to match what your top entrants actually do.
    - Reduce QB CPT / chalky but low‑top1 combos accordingly.
  - Use templates that match sharp behavior (e.g. more 4–2 onslaughts, more leverage plays).

- **Sampling**
  - For each lineup, first sample `population ~ Bernoulli(π)` (e.g. π = 0.15 sharps).
  - Then use the CPT & template sampler (from above) for that population.

This lets you ask: “How does my lineup perform *vs the overall field*?” and “vs just the sharp sub‑field?” without a fancy model.

---

### 4. Use external sims (e.g. SaberSim) as a field lineup source

If you already export SaberSim’s lineups or can cheaply generate them, you can:

- **Pre‑contest:**
  - Pull a big pool of SaberSim lineups (tuned to their ownership).
  - Randomly sample from that set with probability proportional to some weight:
    - Weight by their `My Own` ownership,
    - Or by a smoothed function of projection and ownership.
  - Optionally **add noise** by randomly swapping one or two players according to ownership to simulate “user tweaks”.

- **Advantage**
  - You skip hand‑building a generator and instead just re‑sample from a realistic lineup pool that already encodes correlations and sane constructions.

---

### 5. Keep it lightweight and calibrate with flashbacks

Whatever simple generator you choose, use flashbacks to **tune a few knobs**:

- **Compare distributions:**
  - CPT position mix, CPT player shares
  - Stack patterns (`Stack` column in flashback Simulation)
  - Salary distributions, total ownership / sum ownership

- **Iterate with tiny changes:**
  - Increase or decrease weight on certain templates,
  - Adjust CPT position shares to better match real contests,
  - Add one or two rejection rules if your synthetic field allows weird stuff you never see in reality.

You don’t need ML if you:
- Start with **ownership‑driven sampling**,  
- Add **a few hand‑crafted templates**, and  
- **Calibrate those** against your existing flashback outputs.

If you tell me your current field size and how many synthetic field lineups you’re comfortable simulating (e.g. 50k, 100k), I can sketch a very concrete algorithm (with pseudo‑code) that plugs into your existing projections + ownership tables.