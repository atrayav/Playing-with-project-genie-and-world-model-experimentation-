---
layout: default
title: Two Agents, One World
---

# Two agents, one world: what breaks when AI characters share a generated environment?

*November 4, 2026 · @raj · World models / multi-agent RL*

I recently got invited to a call with someone at Google Labs, so I started reading more about what the team was working on. That's how I ended up learning about Project Genie and spending a lot more time looking into world models than I expected.

One thing I noticed was that both Genie 3 and the new Genie 4 mention interactions between independent agents as an open problem. That's pretty close to the research I've been doing in multi-agent RL, except I usually look at it from the agent's side. I work on how an agent can learn about an unfamiliar partner from its behavior. With Genie, I started wondering about something a bit different: can the *world model itself* keep track of two people acting in the same environment?

I wanted to try it rather than just read about it, so I built a small experiment with an 8×8 grid, two agents, and a learned model that predicts what happens next. I later added an AI character that tries to reach a goal while the other character gets in its way. It's obviously nowhere near the scale of Genie, but it was enough to make the failure cases pretty clear.

The funniest result was that the AI could sometimes reach its goal **by walking through a wall**. If I'd just looked at the success rate, I would've missed that completely.

## Part 1: Learning separate actions from unlabeled transitions

### Why learn actions from video?

The [original Genie paper](https://proceedings.mlr.press/v235/bruce24a.html) starts with a problem I hadn't thought much about before. If you train on gameplay videos from the internet, you see what happened on the screen, but you don't know which buttons the player pressed. Genie handles this with a latent action model that learns a small set of codes to help predict the next frame. Ideally, those codes pick up something like the player's actions without being given the action labels.

If there's only one player, it's easy to imagine a code learning something like "move left." With two players moving at once, though, what should one code mean? A moves left? B moves right? Both do something?

So my first test was pretty simple: what changes if each agent gets its own latent action instead of sharing one code for the entire step?

### The setup

The world is an 8×8 grid with random walls and two agents. Each agent chooses **stay, up, down, left, or right** on every step. They move simultaneously, except that moves fail if an agent hits a wall, two agents try to occupy the same square, or they try to swap places.

I generated **48,000 transitions** from **3,000 episodes of 16 steps**. Their movements were mostly random, although I made them chase each other sometimes so they would actually collide. I trained four versions:

| Model | What it sees during training | Action representation |
|---|---|---|
| Joint 8-code | State transitions, no action labels | One code for both agents |
| Per-agent | State transitions, no action labels | A separate code for each agent, using a shared 8-code vocabulary |
| Per-agent + 5% labels | Same pretraining, then 2,400 labeled steps for dynamics fine-tuning | Two agent-specific codes |
| Fully supervised baseline | True actions throughout training | Ground-truth actions |

I skipped the video tokenizer because the observations are already tiny grids. The learned model predicts the next grid state directly. So this is inspired by Genie's *latent-action question*, not an implementation of Genie 1 or Genie 4.

One thing I need to be careful about here: the single-code model only has eight codes, while the per-agent model has two slots of eight, or up to 64 combinations. There are 25 possible pairs of actions. So part of the difference could just be that the first model doesn't have enough codes. I'd need a larger joint codebook to tell whether separating the agents actually helps beyond giving the model more capacity.

### First result: two code slots made independent steering easier

The one-code model struggled to give each code a clear meaning. Its codes ended up mixing different pairs of actions, and the most common action pair only accounted for around **52%** of a code's uses.

The per-agent codes were easier to interpret. They matched an individual action about **92%** of the time, and in 2,000 sampled states, changing A's code while B was told to stay didn't move B.

<!-- FIG_ACTION_CODES -->

At first I thought that was most of the problem solved. Then I looked at collisions, and the results were much worse.

### The weird part: a blocked move looks exactly like doing nothing

When both moves succeeded, the per-agent model steered correctly around **90%** of the time, compared with **45%** for the joint eight-code baseline.

Then I separated out the cases where a move *couldn't* happen. The per-agent video-only model got roughly **1%** accuracy on wall hits and **0%** on agent collisions in that initial evaluation. It was terrible at precisely the situations where two characters have to share a world.

It took me a bit to figure out why. Imagine an agent standing next to a wall. It tries to move into the wall, so it stays in the same square. Now imagine it chooses "stay" instead. The frames look exactly the same, even though the agent tried to do two different things.

That's the problem with only seeing the frames. You know where the agent ended up, but you don't necessarily know what it tried to do. There isn't enough information in that one transition to tell the two cases apart. Longer histories might give you clues, but a single blocked move is ambiguous.

That distinction matters a lot more when two independent agents can block each other.

### What changed with a little labeled data?

I kept the per-agent encoder and its codes fixed, then fine-tuned the dynamics on **2,400 labeled transitions**, just 5% of the original dataset.

In the initial run, accuracy on wall hits and collisions climbed to about **82%** for each, and overall steering accuracy reached **88%**. The fully supervised baseline reached 100% in this tiny setting.

What I found useful was that the model didn't need to relearn all the movement codes. It mainly needed examples of what happens when an agent *tries* to move and can't.

Rollouts made the gap more obvious. After eight steps feeding each prediction back into the model, **55%** of the fine-tuned rollouts were still exactly correct. The per-agent video-only model was at **4%**, and the joint eight-code model at **0%**. By step 16, the fine-tuned model was down to **32%**. Even the better model drifted; it just drifted much more slowly.

## Part 2: The model is the world

After that, I wanted to try something closer to how someone would actually use an interactive world model.

I kept the same grid, but now one character is an AI trying to reach a goal square. The other character represents a user moving around and sometimes getting in its way. The AI plans a route, issues an action, and sees the next state produced by the learned model.

Here **the learned model is the world**. There's no normal game engine quietly fixing mistakes behind the scenes. If the model says the character walked through a wall, the planner just sees it on the other side.

I tested three ways to run it over **300 episodes**:

| Runtime setup | What happens |
|---|---|
| Open loop | Take the model's next state and keep going |
| Check + replan | Flag an impossible state or failed step and plan again, but don't replace the generated state |
| Check + recover | Reject invalid generated candidates, pick a candidate that passes the checks, and replan when needed |

The checker is deliberately simple and **hand-written**. It looks for agents overlapping, walking through walls, teleporting, or violating the movement rules. It can also tell the planner whether the intended move appears to have happened.

This is a toy analogy to the kind of feedback loop Google describes with Gemini observing Genie 4's generated frames. It is **not** a reproduction of Gemini, and I don't know whether Google's internal system uses candidate rejection like this.

### A 77% success rate that was actually bad news

In the open-loop experiment, the video-only model's AI reached its goal in **77%** of episodes. That was higher than the **64%** goal-reaching rate reported for the better-trained comparison model.

At first that sounds like the video-only model is better. But roughly **two-thirds** of its episodes contained at least one impossible state. It was sometimes reaching the goal by walking through a wall or straight through the user.

That was the part I found pretty surprising. **The AI looked successful partly because the world was letting it cheat.**

I'd rather count an episode as successful only if the AI actually reaches the destination *without* breaking the game's rules. That's a metric I still want to add.

### A checker that only complains doesn't solve much

I thought checking for mistakes and making the AI replan would help a lot. It didn't. Episodes with a broken frame went from about **67%** to **66%**.

The planner was changing its route, but the world model was still making the same kinds of mistakes. A better route doesn't help much if the next frame can still put someone inside a wall.

So I added recovery. Instead of sampling another whole frame, this toy decoder scores the allowed next positions under the model’s existing prediction and takes the highest-scoring valid one. That matters for the latency comparison later: there’s no new model forward pass here.

With that change, accepted checker violations dropped to **0%**, and agreement with the actual gridworld's transition rules improved from **69% to 93%** in the reported evaluation.

<!-- FIG_CLOSED_LOOP -->

The zero needs some context: the checker literally rejects frames that break its rules, so getting zero *accepted* violations is expected. The more useful number is the **93% agreement with a separate simulator**, since the checker doesn't catch every possible wrong outcome.

### The tradeoff I wasn't expecting to care about: latency

Recovery meant the system had to intervene on about **25%** of frames for the video-only model. The model trained with 5% action labels needed intervention on about **8%**.

<!-- FIG_CORRECTION_RATE -->

This is where the engineering side gets interesting. In a real video model, fixing a bad frame could require another expensive generation or more candidates up front. You could make the world more consistent but also slower to respond, which isn't great if someone is trying to play in it.

To be precise, **25% corrected frames does not mean 25% extra latency**. I haven't measured actual end-to-end latency here. That depends on the number of candidates, whether generation is batched, and how expensive the check is. I'd want to measure candidate attempts per accepted frame and action-to-frame delay next.

## What this does (and doesn't) say about Genie

[Google's Genie 4 announcement](https://deepmind.google/blog/genie-4-a-new-generation-of-world-simulation/) describes a system where Gemini observes the generated simulation, checks whether steps happened, and changes its plan when a user intervenes. Google also lists interactions between independent agents and maintaining a shared world as ongoing research problems.

That's what I was trying to understand with this project. In my gridworld, being able to control both agents, making the world behave correctly, and reaching the goal turned out to be three different problems. Doing well on one didn't mean the others worked.

But the differences matter. Genie generates video (and now audio); my model predicts symbolic grids. The checker reads structured state with explicit wall and collision rules. Real video doesn't hand you a neat object position or tell you whether a frame is physically possible. I also used one primary training seed, an artificial behavior policy, and agent identities that were given to the model for free.

I wouldn't use these numbers to say anything about how well Genie works. The model and the environment here are way too simple for that. What the experiment did show me is how easy it is to get a good-looking task success rate when the generated environment is making mistakes.

## What I'd test next

The first thing I'd do is train a **capacity-matched joint-action model**. My eight-code baseline is too small to separate whether per-agent factorization helps from whether it simply has more room to represent actions.

Then I'd rerun everything across multiple seeds and unseen wall layouts, and report **valid task success** alongside raw goal-reaching. I'd also log how many candidate frames recovery needs and what it costs in actual runtime. That would make the latency story something I could measure instead of speculate about.

The longer-term question is how much of this checking can be learned. Hand-written rules are easy for an 8×8 grid. They won't cover all the ways a generated visual world can drift, and a learned checker will have its own mistakes.

## What I took away

I spend a lot of time in my research asking what an agent can learn about another agent from a trajectory. This was a fun way to look at the same problem from the world model's perspective instead.

The thing I'll remember is watching the AI walk through a wall and still get counted as successful! It made me much more careful about what we're actually measuring when the environment is generated too.

## Code and reproducibility

The [code and figures are on GitHub](https://github.com/atrayav/Playing-with-project-genie-and-world-model-experimentation-). The experiment consists of the gridworld, latent-action models, rollout evaluation, and a second stage with verification, replanning, and recovery. Both parts were designed to run on a laptop CPU. The numbers above are from the initial reported runs; they haven't yet been replicated across multiple seeds, so please treat the precise percentages as provisional.

I used an AI coding assistant while implementing it. I chose the questions and experimental comparisons, debugged the behavior, and worked through what the results might mean. The implementation and metrics still deserve an independent reproducibility pass before anyone treats this as a research claim.

If you've worked on world models or multi-agent RL and see something I missed, feel free to reach out. I'd be interested in your take.

---

**References:** [Genie: Generative Interactive Environments (ICML 2024)](https://proceedings.mlr.press/v235/bruce24a.html); [Google DeepMind's Genie 4 announcement (Oct 27, 2026)](https://deepmind.google/blog/genie-4-a-new-generation-of-world-simulation/); [Project Genie launch (Jan 29, 2026)](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/project-genie/).
