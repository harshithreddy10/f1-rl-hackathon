# Clarify — Resolved Questions & Decisions

## Q1: Which race(s) should we use?
**Answer:** Monaco 2023 as primary training race. Monza 2023 as validation (different strategy profile — high-speed, 1-stop vs Monaco's 2-stop). Both available via FastF1.

## Q2: How many laps in the simulation?
**Answer:** 15-lap simplified episodes during training (faster convergence). Full 57-lap Monaco race for the final demo comparison.

## Q3: Single driver or full grid?
**Answer:** Single driver only. Focus on Carlos Sainz (Ferrari, Monaco 2023) — he had an interesting 2-stop strategy that the agent should be able to beat.

## Q4: What counts as the "real strategy" baseline?
**Answer:** The actual pit laps and compounds used by Sainz in Monaco 2023, loaded directly from FastF1. This is the ground truth we compare against.

## Q5: How do we handle safety car periods?
**Answer:** Include safety car as a binary state variable (from FastF1 `TrackStatus`). The agent observes it but we don't need to simulate safety car deployment — just read from historical data.

## Q6: What if the RL agent doesn't converge in time?
**Answer:** Pre-train overnight before the hackathon using the same environment. Bring saved weights. If live training fails, load saved weights and present training curves as evidence of learning.

## Q7: How many pit stops should the agent be allowed?
**Answer:** Unconstrained — the agent can pit every lap if it wants. The pit penalty (-22s) naturally discourages excessive stops. FIA mandates at least 2 compounds used — encode this as a hard constraint in the episode end check.

## Q8: What's the minimum viable demo if everything goes wrong?
**Answer:** Show the tyre degradation model + a rule-based agent (pit at lap 18, 36) + the visualization. This alone is a complete, working, explainable AI system.
