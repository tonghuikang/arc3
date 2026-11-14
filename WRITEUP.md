Public LB 41st, Private LB xx, likely Silver

(Work in progress, will add more links and content later)

Some fun facts

- In milestone 1, I was in 4th place.
- In milestone 2, I was in 7th place, with 3 teams ahead of me deciding not to open-source.

Unfortunately, I missed both milestone prizes by one place.

Key ideas

- Cache eviction
- fp4 cache
- No MTP
- Lightly tuned model.
    On the same harness, the stock model scored [33.99, 27.93] and my fine-tune scored [31.25, 37.92, 36.27, 35.77] on the public leaderboard.

I will organize my writeup into four sections: harness, inference, post-training, and AI coding.

Key URLs

- TODO


# Harness

Implementation

- I built my harness from scratch.
- The harness has to store the results token-in token-out.
- I raise Python exceptions on GameOver and level clear.
- Each `execute()` call performs only one action.
- I built a visualization so I can mouse over and check the tokens.
- The image I gave is slightly different.
    It has grid numbers, which I think also makes it easier for me to review.

I do not think my current harness is one standard deviation better than the winning Milestone 2 solution, but I do get token-in token-out generation, which has been useful for post-training.

Some things I learnt

- I should have started from the Duck harness.
    I should have adopted many of the Duck harness's features earlier, such as showing the difference between the current frame and the next frame.
- Tokens-in tokens-out is the correct format to save the data, even if you are not planning to post-train.
- Just give the agent what it wants.
    The agent frequently prints ASCII of the board instead of rendering an image.
    This does not make sense to me if you look at which tokens the ASCII is composed of.
    For a long time, I instructed the agent to avoid printing ASCII and gave it only tools to render images, only to have the agent print the ASCII.
- Tabulate how many tokens are wasted.
    For example, if the agent needs to write code to render ASCII, you should give the agent a function to call that prints the ASCII it wants.
- Focus on iterating on what matters.
    If you cannot prove that a prompt change is an improvement, there is no need to spend a lot of time on it.
- Improve the prompt engineering process.
    For example, at GameOver you have a reminder of what the agent should be doing.
    What is the best prompt to get the agent to understand why it died?
    Again, focus only on places where you can prove an improvement.


# Inference

Key ideas

- KV cache eviction
- No speculative decoding, which allows for one Mamba state per session.
- fp4 cache

Key numbers and results

- KV pool: TBC
- Decode width: TBC
- Throughput: TBC

KV cache eviction is an idea I think not many people will discover.
Instead of keeping all the tokens in the context in the KV cache, the idea is to drop some of the KV cache.
This is a visualization of the cache being evicted (TODO).
I had this idea at Milestone 2, and it is the key idea I hoped to have popularized.

Cache eviction allows for:

- Packing more decode sequences together to achieve higher throughput
- Giving each session a fixed KV cache budget
- No need to bin-pack or predict how many tokens you need to decode

Agentic models should be trained with the ability to decide which parts of their cache to evict, instead of running compression.
With this, you can effectively get models with unlimited context.
There is currently some research on cache engineering, and I hope this makes it into mainstream models someday.


Some things I learnt

- Do get a coding agent to implement from scratch without looking at your current solution.
- One great idea (cache eviction) is not sufficient to win a gold medal; you still need great execution.
- I only discovered that I could use fp4 KV cache when I had only one submission left.
    It increased my throughput from 760 to 1300 without an apparent loss in points per token.
    Even though none of the prize-winning solutions used an fp4 cache, the 4th-highest published solution did manage to get above 20 points with an fp4 cache.
- After my final submission, I looked at the comparisons I had made.
    I also missed out on offloading the input embedding to the KV cache, which could have allowed me to decode even more games in parallel.
- It is quite amazing that you can take a 4-point solution up to probably 55 points with harness and inference engineering.


# Post-training

This is my research question: what is the minimum number of tokens needed to teach the agent one thing in the weights?

This is how I approached my initial fine-tuning process:

- Take the (already tokenized) traces from the Kaggle runs.
- Select spans where I think the agent could have improved.
    In my toy example, I am training the model NOT to click on all green objects in Game bp35 level 1, and not to do extensive meter reasoning, which wastes tokens.
- Run inference to reroll from the start of the spans.
- Get another agent to select the better spans.
- Train to produce the better spans (and maybe train to avoid the worse spans).
- In real rollouts, check whether the model still shows those behaviors.

In theory, this should work.
I also hope that this is how we will post-train models: take traces from production and learn what could have been done better.
I hope this is far more token-efficient than training on full rollouts, which have mistakes mixed in.
However, I was not able to make this work.

I have also tried another trajectory generation method.
You generate the chain of thought paragraph by paragraph, and another agent decides which paragraph is better.
The main agent should then be better at solving the level, with more efficient actions and fewer tokens.
However, I discovered that this is bad: you are training an agent that believes it never makes mistakes.
In any case, this still shows a feasible number of tokens with which you can train the agent to solve the problem.

When I post-train, I think these metrics are interesting to look at:

- The entropy.
    Are you merely training the model to make predictions with a different confidence?
    If that is the case, you should just change the temperature parameter.
- Compare the training loss with the reference loss.
    Training loss alone is very noisy; it depends greatly on the tokens.
    You need to obtain the reference loss.
    At scale you might not want to do this for every token, but it is an indicator of whether you are training your model to predict the tokens with higher likelihood.
    Naively, when you train on a set of better sequences, you should get a better difference on a held-out set of sequences.
    After thinking about this more, I do not think this should necessarily be the case.

I also tried the following algorithms, unintentionally popularized by Thinking Machines:

- Score centering.
    This brought into mainstream discussion whether you are training your model to fit the sampler.
    With this, I realized that I might have been fitting my training to the top-k sampler.
    I tried training on the same dataset again, but it still did not achieve my objectives.
- DiMuon.
    Someone also reverse-engineered DiMuon, and I tried it on my setup.
    I do not really see a difference, and I am not sure where to look.
    I still need to learn the key idea behind why DiMuon is better than AdamW.

Money spent

- Around $1600 of my own money, along with another $1100 of compute credits.

What I learnt

- Think in terms of token counts: rollout tokens, rollout throughput, training tokens, and training throughput.
- You do want to fix your throughput before training.
    If you have access to 24 GPU-hours, it is worth using 4 hours to double your throughput for the remaining 20 hours.
- My approach has been to start with a big model and small improvements.
    If the big model can learn the small improvements, you have the confidence to throw more GPU resources at the problem.
    The issue with starting with small models is that they cannot achieve even the small improvements.
- I think it is good to have hands-on experience with how difficult post-training is, instead of theorizing about what post-training should be.
- I think post-training publications should show the data like I do.
    You can host 1GB of tokens and numbers on GitHub for free.
- I have never done a tutorial on post-training, not even one where you GRPO a small model on GSM8K.
    I do not have a good idea of what successful post-training looks like.
    I will be doing one soon.
- In 2026, you should bet that most Kagglers still do not know how to RL models.
    If your objective is to get a gold medal, you should focus your efforts and money on inference engineering.
    I still hope that there will be Kaggle competitions that focus on post-training.
    I look forward to the Gemma Developer Agent Competition, where you should need to post-train the adapter to get a gold medal.


# AI coding

I do not look at the code anymore.

There are some things I believe the agent will not do automatically, unless you give it a lot of autonomy and GPU resources.

- Use of the sessions API instead of chat completions
- Use of a fully token-in token-out format

There were multiple times I needed to remind the AI of the following:

- Individual latency does not matter.
    Only throughput matters.
- (need to think of more)

I think I need a better system for enforcing these reminders.

AI is currently quite bad at calculating the upper bound given the GPU and environment specifications.
The answer is usually very different from the actual throughput achieved.

I think I should also optimize for my own learning.
One thing I should have done for inference engineering is to break down how much time each part takes, and how much memory each part takes.
They should add up to the total.
Otherwise, something is missing and the agent is expected to find it.
I would have learnt more this way, instead of asking to "increase overall throughput".

