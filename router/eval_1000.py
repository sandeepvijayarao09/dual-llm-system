"""
eval_1000.py — Offline evaluation of the ML Router against 1000 labeled test cases.

Usage:
    python -m router.eval_1000

Requirements:
    - router/models/router_v0.joblib must exist  (run: python -m router.train)
    - scikit-learn, joblib installed

No LLM calls are made.  Only the MLRouter (TF-IDF + Logistic Regression) is
evaluated.  Cascade rate is computed as a simulation: queries where
ML confidence < ML_ROUTER_CONFIDENCE_THRESHOLD (0.65) *would* fall back to
the LLM classifier in production.

Key differences from eval_500.py:
    - 1000 entirely new queries (no overlap with eval_500.py)
    - Four new categories: conversational, debug_simple,
      ethical_philosophical, comparison_technical
    - Before/After comparison table (eval_500 results are hardcoded)
    - Results saved to router/eval_results_1000.json
"""

from __future__ import annotations

import json
import os
import statistics
import sys
from dataclasses import dataclass, asdict
from typing import Optional

# ---------------------------------------------------------------------------
# 1000 labeled test cases — ALL NEW, none reused from eval_500.py
# Format: (query, ground_truth_label, category)
# ground_truth: "small" | "big"
# ---------------------------------------------------------------------------

TEST_CASES: list[tuple[str, str, str]] = [

    # =========================================================
    # GREETINGS — 25 cases (all "small")
    # =========================================================
    ("Morning!", "small", "greetings"),
    ("Hi there!", "small", "greetings"),
    ("Hey!", "small", "greetings"),
    ("Hello, how's it going?", "small", "greetings"),
    ("What's new?", "small", "greetings"),
    ("Nice to meet you!", "small", "greetings"),
    ("Thanks a lot!", "small", "greetings"),
    ("Many thanks", "small", "greetings"),
    ("Appreciate it!", "small", "greetings"),
    ("Goodbye!", "small", "greetings"),
    ("Later!", "small", "greetings"),
    ("See ya", "small", "greetings"),
    ("Good luck!", "small", "greetings"),
    ("Welcome!", "small", "greetings"),
    ("Congrats!", "small", "greetings"),
    ("Congrats on the new role!", "small", "greetings"),
    ("Happy birthday!", "small", "greetings"),
    ("Have a good weekend", "small", "greetings"),
    ("Hope everything is going well", "small", "greetings"),
    ("Wishing you all the best", "small", "greetings"),
    ("Hiya!", "small", "greetings"),
    ("Yo!", "small", "greetings"),
    ("Sup?", "small", "greetings"),
    ("Howdy partner", "small", "greetings"),
    ("Greetings!", "small", "greetings"),

    # =========================================================
    # FACTUAL LOOKUPS — 70 cases (all "small")
    # =========================================================
    ("What is the capital of Italy?", "small", "factual"),
    ("Who wrote Hamlet?", "small", "factual"),
    ("When did the French Revolution begin?", "small", "factual"),
    ("What is the chemical formula for salt?", "small", "factual"),
    ("Who invented the steam engine?", "small", "factual"),
    ("What is the population of New York City?", "small", "factual"),
    ("Where is the Great Wall of China located?", "small", "factual"),
    ("Who was the first woman in space?", "small", "factual"),
    ("What is the national language of Brazil?", "small", "factual"),
    ("What year was the first iPhone released?", "small", "factual"),
    ("What is the symbol for potassium on the periodic table?", "small", "factual"),
    ("How many moons does Mars have?", "small", "factual"),
    ("Who discovered gravity?", "small", "factual"),
    ("What is the currency of Japan?", "small", "factual"),
    ("How long is the Nile River?", "small", "factual"),
    ("Who painted the Sistine Chapel ceiling?", "small", "factual"),
    ("What is the capital of Mexico?", "small", "factual"),
    ("What year was Amazon founded?", "small", "factual"),
    ("What is the melting point of iron?", "small", "factual"),
    ("Who wrote 'To Kill a Mockingbird'?", "small", "factual"),
    ("What is the smallest country in the world?", "small", "factual"),
    ("What does HTTPS stand for?", "small", "factual"),
    ("What is the capital of Argentina?", "small", "factual"),
    ("How many vertebrae are in the human spine?", "small", "factual"),
    ("What year did the Berlin Wall fall?", "small", "factual"),
    ("Who developed Python programming language?", "small", "factual"),
    ("What is the atomic number of hydrogen?", "small", "factual"),
    ("How many rings does Saturn have?", "small", "factual"),
    ("What country is the Amazon River in?", "small", "factual"),
    ("Who was Marie Curie?", "small", "factual"),
    ("What is the capital of Egypt?", "small", "factual"),
    ("What does GPU stand for?", "small", "factual"),
    ("Who is the CEO of Tesla?", "small", "factual"),
    ("What is the boiling point of nitrogen?", "small", "factual"),
    ("What year was the World Wide Web invented?", "small", "factual"),
    ("What is the largest country by area?", "small", "factual"),
    ("What is the national animal of Australia?", "small", "factual"),
    ("What language is spoken in Egypt?", "small", "factual"),
    ("How many bytes are in a megabyte?", "small", "factual"),
    ("Who wrote 'The Great Gatsby'?", "small", "factual"),
    ("What is the capital of Nigeria?", "small", "factual"),
    ("What year was Linux first released?", "small", "factual"),
    ("What is the deepest lake in the world?", "small", "factual"),
    ("Who founded Microsoft?", "small", "factual"),
    ("What does TCP stand for?", "small", "factual"),
    ("What is the chemical element symbol for silver?", "small", "factual"),
    ("What year was the United States founded?", "small", "factual"),
    ("How many chambers are in the human heart?", "small", "factual"),
    ("Who invented the airplane?", "small", "factual"),
    ("What is the capital of Thailand?", "small", "factual"),
    ("What does API stand for?", "small", "factual"),
    ("Who wrote 'The Origin of Species'?", "small", "factual"),
    ("What is the speed of sound in air?", "small", "factual"),
    ("What country is the Sahara Desert in?", "small", "factual"),
    ("Who is Nikola Tesla?", "small", "factual"),
    ("What year was the Eiffel Tower built?", "small", "factual"),
    ("What is the chemical symbol for gold?", "small", "factual"),
    ("Who invented the printing press?", "small", "factual"),
    ("What is the capital of Pakistan?", "small", "factual"),
    ("How many bits in a byte?", "small", "factual"),
    ("What is the second largest planet?", "small", "factual"),
    ("What language is spoken in Switzerland?", "small", "factual"),
    ("What year was Facebook founded?", "small", "factual"),
    ("Who invented the transistor?", "small", "factual"),
    ("What is the currency of China?", "small", "factual"),
    ("What does LAN stand for?", "small", "factual"),
    ("What is the capital of Sweden?", "small", "factual"),
    ("How many bones are in the adult human body?", "small", "factual"),
    ("Who is Alan Turing?", "small", "factual"),
    ("What is the boiling point of ethanol?", "small", "factual"),

    # =========================================================
    # MATH SIMPLE — 30 cases (all "small")
    # =========================================================
    ("What is 13 times 7?", "small", "math_simple"),
    ("What is 250 divided by 5?", "small", "math_simple"),
    ("Convert 10 miles to kilometers", "small", "math_simple"),
    ("What is 40% of 300?", "small", "math_simple"),
    ("What is the square root of 256?", "small", "math_simple"),
    ("How many seconds are in a day?", "small", "math_simple"),
    ("What is 2 to the power of 8?", "small", "math_simple"),
    ("Convert 37 degrees Celsius to Fahrenheit", "small", "math_simple"),
    ("What is 999 divided by 9?", "small", "math_simple"),
    ("How many feet are in 5 miles?", "small", "math_simple"),
    ("What is 18 squared?", "small", "math_simple"),
    ("Convert 2.5 hours to minutes", "small", "math_simple"),
    ("How many ounces in 3 pounds?", "small", "math_simple"),
    ("What is 15% of 80?", "small", "math_simple"),
    ("How many days are in 5 years?", "small", "math_simple"),
    ("What is 3.14 times 100?", "small", "math_simple"),
    ("Convert 1000 grams to kilograms", "small", "math_simple"),
    ("What is 64 divided by 8?", "small", "math_simple"),
    ("How many minutes in a week?", "small", "math_simple"),
    ("What is 5 factorial?", "small", "math_simple"),
    ("Convert 500 milliliters to liters", "small", "math_simple"),
    ("What is 11 times 11?", "small", "math_simple"),
    ("How many inches in 3 feet?", "small", "math_simple"),
    ("What is 1000 minus 376?", "small", "math_simple"),
    ("Convert 90 degrees Fahrenheit to Celsius", "small", "math_simple"),
    ("What is the cube root of 27?", "small", "math_simple"),
    ("How many hours in a week?", "small", "math_simple"),
    ("What is 20% of 500?", "small", "math_simple"),
    ("What is 144 divided by 12?", "small", "math_simple"),
    ("Convert 5 liters to gallons", "small", "math_simple"),

    # =========================================================
    # YES / NO — 25 cases (all "small")
    # =========================================================
    ("Is Python dynamically typed?", "small", "yes_no"),
    ("Is Pluto a planet?", "small", "yes_no"),
    ("Is Rust memory-safe?", "small", "yes_no"),
    ("Is the moon larger than Earth?", "small", "yes_no"),
    ("Does Python support multiple inheritance?", "small", "yes_no"),
    ("Is Docker written in Go?", "small", "yes_no"),
    ("Is 0 an even number?", "small", "yes_no"),
    ("Is Node.js single-threaded?", "small", "yes_no"),
    ("Is hydrogen a metal?", "small", "yes_no"),
    ("Is SQL a programming language?", "small", "yes_no"),
    ("Is the Great Wall visible from the moon?", "small", "yes_no"),
    ("Can sharks smell blood from far away?", "small", "yes_no"),
    ("Is Vue.js a front-end framework?", "small", "yes_no"),
    ("Is IPv6 backward compatible with IPv4?", "small", "yes_no"),
    ("Is the Earth older than the Sun?", "small", "yes_no"),
    ("Is Kotlin interoperable with Java?", "small", "yes_no"),
    ("Is Redis an in-memory database?", "small", "yes_no"),
    ("Can you run Python 2 code in Python 3?", "small", "yes_no"),
    ("Is the browser single-threaded?", "small", "yes_no"),
    ("Is WebAssembly faster than JavaScript?", "small", "yes_no"),
    ("Is recursion faster than iteration in Python?", "small", "yes_no"),
    ("Does Java have pointers?", "small", "yes_no"),
    ("Is Git distributed?", "small", "yes_no"),
    ("Is machine learning a subset of AI?", "small", "yes_no"),
    ("Is GraphQL a database?", "small", "yes_no"),

    # =========================================================
    # DEFINITIONS — 30 cases (all "small")
    # =========================================================
    ("What is a pointer in programming?", "small", "definitions"),
    ("Define inheritance in OOP", "small", "definitions"),
    ("What is a mutex?", "small", "definitions"),
    ("Define idempotency", "small", "definitions"),
    ("What is a REST API?", "small", "definitions"),
    ("What does CRUD stand for?", "small", "definitions"),
    ("Define technical debt", "small", "definitions"),
    ("What is a singleton pattern?", "small", "definitions"),
    ("What is eventual consistency?", "small", "definitions"),
    ("Define sharding in databases", "small", "definitions"),
    ("What is a race condition?", "small", "definitions"),
    ("What does DRY stand for in programming?", "small", "definitions"),
    ("Define big-O notation", "small", "definitions"),
    ("What is a callback function?", "small", "definitions"),
    ("What is a microservice?", "small", "definitions"),
    ("Define load balancing", "small", "definitions"),
    ("What is a CDN?", "small", "definitions"),
    ("What does MVP stand for in product development?", "small", "definitions"),
    ("Define overfitting in machine learning", "small", "definitions"),
    ("What is a hyperparameter?", "small", "definitions"),
    ("Define tokenization in NLP", "small", "definitions"),
    ("What is backpropagation?", "small", "definitions"),
    ("What is a closure in programming?", "small", "definitions"),
    ("Define garbage collection", "small", "definitions"),
    ("What is a deadlock?", "small", "definitions"),
    ("What is a foreign key?", "small", "definitions"),
    ("Define blue-green deployment", "small", "definitions"),
    ("What is a canary release?", "small", "definitions"),
    ("What is an ORM?", "small", "definitions"),
    ("Define feature engineering", "small", "definitions"),

    # =========================================================
    # SHORT CREATIVE — 30 cases (all "small")
    # =========================================================
    ("Write a haiku about Python", "small", "short_creative"),
    ("Give me a pun about databases", "small", "short_creative"),
    ("Tell me a joke about recursion", "small", "short_creative"),
    ("Give me a motivational quote for developers", "small", "short_creative"),
    ("Tell me a fun programming fact", "small", "short_creative"),
    ("Write a limerick about merge conflicts", "small", "short_creative"),
    ("Give me a clever team name for a hackathon", "small", "short_creative"),
    ("Tell me a joke about null pointers", "small", "short_creative"),
    ("Write a two-line poem about APIs", "small", "short_creative"),
    ("Make up a name for a startup that sells developer tools", "small", "short_creative"),
    ("Give me a pun about machine learning", "small", "short_creative"),
    ("Tell me a riddle about a hash table", "small", "short_creative"),
    ("Write a haiku about refactoring", "small", "short_creative"),
    ("Give me a tagline for a code review tool", "small", "short_creative"),
    ("Tell me a knock-knock joke about cloud computing", "small", "short_creative"),
    ("Make up a metaphor for technical debt", "small", "short_creative"),
    ("Give me a funny variable name for speed", "small", "short_creative"),
    ("Write a one-sentence story about a developer on deadline", "small", "short_creative"),
    ("Give me a creative name for a DevOps conference", "small", "short_creative"),
    ("Tell me a silly fact about programming languages", "small", "short_creative"),
    ("Write a haiku about a segfault", "small", "short_creative"),
    ("Give me a rhyme about git rebase", "small", "short_creative"),
    ("Make up a pun about Docker containers", "small", "short_creative"),
    ("Tell me a joke about agile sprints", "small", "short_creative"),
    ("Give me a one-liner about unit tests", "small", "short_creative"),
    ("Write a limerick about CI/CD pipelines", "small", "short_creative"),
    ("Give me a fun fact about Alan Turing", "small", "short_creative"),
    ("Make up a tagline for a monitoring tool", "small", "short_creative"),
    ("Tell me a riddle about a binary tree", "small", "short_creative"),
    ("Give me a motivational quote about debugging", "small", "short_creative"),

    # =========================================================
    # CONVERSATIONAL (follow-up style) — 25 cases (all "small")
    # NEW CATEGORY not in eval_500.py
    # =========================================================
    ("Can you elaborate on that?", "small", "conversational"),
    ("What did you mean by that last part?", "small", "conversational"),
    ("Could you give me an example?", "small", "conversational"),
    ("Can you say that more simply?", "small", "conversational"),
    ("OK and then what happens?", "small", "conversational"),
    ("What does that mean in practice?", "small", "conversational"),
    ("Can you repeat that?", "small", "conversational"),
    ("What was the last thing you said?", "small", "conversational"),
    ("I didn't quite get that, can you try again?", "small", "conversational"),
    ("Can you put that in plain English?", "small", "conversational"),
    ("Can you give a shorter version?", "small", "conversational"),
    ("Can you be more specific?", "small", "conversational"),
    ("What does that look like in code?", "small", "conversational"),
    ("Could you give me a real world example of that?", "small", "conversational"),
    ("Can you elaborate on the second point?", "small", "conversational"),
    ("OK what about the alternative approach?", "small", "conversational"),
    ("Can you expand on why that matters?", "small", "conversational"),
    ("What's the tldr?", "small", "conversational"),
    ("Give me a one sentence summary of the above", "small", "conversational"),
    ("Is there anything else I should know?", "small", "conversational"),
    ("Can you double-check that answer?", "small", "conversational"),
    ("What's the main takeaway here?", "small", "conversational"),
    ("What else is important?", "small", "conversational"),
    ("OK and what about edge cases?", "small", "conversational"),
    ("Can you list the key points?", "small", "conversational"),

    # =========================================================
    # DEBUG SIMPLE — 25 cases (all "small")
    # NEW CATEGORY — one-line quick fix, not in eval_500.py
    # =========================================================
    ("Why does Python say 'list index out of range'?", "small", "debug_simple"),
    ("What causes a 'None' result when I expected a value?", "small", "debug_simple"),
    ("Why is my for loop running one time too many?", "small", "debug_simple"),
    ("Why is Python printing 'NoneType object is not subscriptable'?", "small", "debug_simple"),
    ("What does 'KeyError' mean in Python?", "small", "debug_simple"),
    ("Why does my variable print as undefined in JavaScript?", "small", "debug_simple"),
    ("What does 'AttributeError: object has no attribute' mean?", "small", "debug_simple"),
    ("Why am I getting a 404 error from my API?", "small", "debug_simple"),
    ("What does 'IndentationError' mean in Python?", "small", "debug_simple"),
    ("Why is my CSS not applying to the page?", "small", "debug_simple"),
    ("What does 'TypeError: int object is not iterable' mean?", "small", "debug_simple"),
    ("Why does my Python script work locally but not on the server?", "small", "debug_simple"),
    ("What does a 500 internal server error usually mean?", "small", "debug_simple"),
    ("Why is my SQL query returning duplicate rows?", "small", "debug_simple"),
    ("What causes a 'connection refused' error?", "small", "debug_simple"),
    ("Why am I getting a CORS error?", "small", "debug_simple"),
    ("What does 'pip: command not found' mean?", "small", "debug_simple"),
    ("Why is my git push being rejected?", "small", "debug_simple"),
    ("What does 'ImportError: No module named X' mean?", "small", "debug_simple"),
    ("Why does JSON.parse throw an error?", "small", "debug_simple"),
    ("What does 'permission denied' mean when running a script?", "small", "debug_simple"),
    ("Why is my Docker container exiting immediately?", "small", "debug_simple"),
    ("What does 'segmentation fault' mean?", "small", "debug_simple"),
    ("Why is my React component not re-rendering?", "small", "debug_simple"),
    ("What does 'undefined is not a function' mean in JavaScript?", "small", "debug_simple"),

    # =========================================================
    # PROOFS — 20 cases (all "big")
    # =========================================================
    ("Prove that log base 2 of 3 is irrational.", "big", "proofs"),
    ("Prove that any integer squared is non-negative.", "big", "proofs"),
    ("Derive the formula for the sum of the first n integers.", "big", "proofs"),
    ("Prove that the set of real numbers is uncountable (Cantor's diagonal argument).", "big", "proofs"),
    ("Show that every subgroup of a cyclic group is cyclic.", "big", "proofs"),
    ("Prove the Pythagorean theorem using area arguments.", "big", "proofs"),
    ("Prove that a polynomial of degree n has at most n roots.", "big", "proofs"),
    ("Derive the variance formula E[X^2] - (E[X])^2 from first principles.", "big", "proofs"),
    ("Prove that the sum of angles in a triangle is 180 degrees.", "big", "proofs"),
    ("Show by induction that 2^n > n for all natural numbers n.", "big", "proofs"),
    ("Prove that if a divides b and b divides c then a divides c.", "big", "proofs"),
    ("Derive the formula for the eigenvalues of a 2x2 matrix.", "big", "proofs"),
    ("Prove that there is no rational number whose square equals 3.", "big", "proofs"),
    ("Show that the derivative of sin(x) is cos(x) from the limit definition.", "big", "proofs"),
    ("Prove the AM-GM inequality for two positive real numbers.", "big", "proofs"),
    ("Derive the softmax gradient with respect to its inputs.", "big", "proofs"),
    ("Show that a continuous function on [a,b] is uniformly continuous.", "big", "proofs"),
    ("Prove that the inverse of a bijective function is also bijective.", "big", "proofs"),
    ("Prove that BFS finds the shortest path in an unweighted graph.", "big", "proofs"),
    ("Show that the mutual information I(X;Y) >= 0 for any joint distribution.", "big", "proofs"),

    # =========================================================
    # SYSTEM DESIGN — 30 cases (all "big")
    # =========================================================
    ("Design a global leaderboard service for a mobile game with 50 million daily users.", "big", "system_design"),
    ("How would you build a distributed file storage system like Dropbox?", "big", "system_design"),
    ("Design a real-time sports score tracker that serves millions of users during peak events.", "big", "system_design"),
    ("How would you architect a multi-region active-passive database failover system?", "big", "system_design"),
    ("Design a social media platform's follower/following graph at Twitter scale.", "big", "system_design"),
    ("How would you build a distributed configuration management system?", "big", "system_design"),
    ("Design an e-commerce product catalog search with faceted filters for 100M products.", "big", "system_design"),
    ("How would you architect a real-time bidding (RTB) system for online advertising?", "big", "system_design"),
    ("Design a news feed that shows personalized content from people a user follows.", "big", "system_design"),
    ("How would you build a globally distributed key-value store with tunable consistency?", "big", "system_design"),
    ("Design a financial ledger system with ACID guarantees and audit trail.", "big", "system_design"),
    ("How would you build a system to detect duplicate images across a photo library?", "big", "system_design"),
    ("Design a distributed tracing system for a microservices architecture with 1000 services.", "big", "system_design"),
    ("How would you architect a video transcoding pipeline for user-uploaded content?", "big", "system_design"),
    ("Design a ride-hailing driver-matching system that minimizes wait time.", "big", "system_design"),
    ("How would you build an API gateway that handles auth, rate limiting, and routing?", "big", "system_design"),
    ("Design a smart home automation backend that handles millions of IoT device events.", "big", "system_design"),
    ("How would you architect a streaming analytics platform for clickstream data?", "big", "system_design"),
    ("Design a calendar scheduling system with conflict detection across time zones.", "big", "system_design"),
    ("How would you build a price comparison engine for e-commerce products?", "big", "system_design"),
    ("Design a system to enforce API quotas per tenant in a multi-tenant SaaS.", "big", "system_design"),
    ("How would you build a feature flag system that supports gradual rollouts?", "big", "system_design"),
    ("Design a distributed metrics aggregation pipeline for a monitoring platform.", "big", "system_design"),
    ("How would you architect a service that generates PDF invoices at 10K/hour?", "big", "system_design"),
    ("Design a chat moderation pipeline for a platform with 10M messages per day.", "big", "system_design"),
    ("How would you build a recommendation system for an e-learning platform?", "big", "system_design"),
    ("Design a document versioning system like Google Docs revision history.", "big", "system_design"),
    ("How would you architect an event-sourcing system with replay and snapshots?", "big", "system_design"),
    ("Design a data lake ingestion pipeline from 20 heterogeneous source systems.", "big", "system_design"),
    ("How would you build a zero-trust security model for a cloud-native application?", "big", "system_design"),

    # =========================================================
    # ANALYSIS — 30 cases (all "big")
    # =========================================================
    ("Analyze the trade-offs between synchronous and asynchronous API design patterns.", "big", "analysis"),
    ("Explain how attention is computed in multi-head self-attention and why multiple heads help.", "big", "analysis"),
    ("Analyze the pros and cons of eventual consistency in distributed databases.", "big", "analysis"),
    ("Explain why deep neural networks can overfit and what regularization strategies prevent it.", "big", "analysis"),
    ("Analyze the root causes of the dot-com bubble and what patterns repeated in crypto.", "big", "analysis"),
    ("Explain the trade-offs between static and dynamic typing from a software engineering perspective.", "big", "analysis"),
    ("Analyze how Kubernetes achieves high availability for stateful workloads.", "big", "analysis"),
    ("Explain why gradient vanishing happens in deep networks and how modern architectures solve it.", "big", "analysis"),
    ("Analyze the societal implications of algorithmic content recommendation systems.", "big", "analysis"),
    ("Explain how compilers perform dead code elimination and why it matters.", "big", "analysis"),
    ("Analyze the economic incentives behind open-source software development.", "big", "analysis"),
    ("Explain how database query optimizers choose between index scan and sequential scan.", "big", "analysis"),
    ("Analyze the security implications of using third-party JavaScript dependencies.", "big", "analysis"),
    ("Explain why batch normalization accelerates training in deep neural networks.", "big", "analysis"),
    ("Analyze the game theory behind network effects in platform businesses.", "big", "analysis"),
    ("Explain how modern CPUs achieve instruction-level parallelism with out-of-order execution.", "big", "analysis"),
    ("Analyze the trade-offs between columnar and row-oriented storage for analytics.", "big", "analysis"),
    ("Explain why distributed transactions are hard and what saga pattern solves.", "big", "analysis"),
    ("Analyze how social media algorithms affect political polarization.", "big", "analysis"),
    ("Explain the mathematical intuition behind principal component analysis.", "big", "analysis"),
    ("Analyze the engineering challenges of building a global DNS infrastructure.", "big", "analysis"),
    ("Explain how zero-knowledge proofs work and what problems they solve.", "big", "analysis"),
    ("Analyze the impact of Moore's law slowing down on software engineering.", "big", "analysis"),
    ("Explain how eBPF enables observability without traditional kernel modules.", "big", "analysis"),
    ("Analyze the differences between actor model concurrency and shared-memory threading.", "big", "analysis"),
    ("Explain why HTTP/3 (QUIC) improves performance over HTTP/2 for lossy networks.", "big", "analysis"),
    ("Analyze the trade-offs between tight and loose coupling in microservices.", "big", "analysis"),
    ("Explain how neural architecture search works and what its computational costs are.", "big", "analysis"),
    ("Analyze the engineering trade-offs between building versus buying infrastructure.", "big", "analysis"),
    ("Explain how memory safety bugs (use-after-free, buffer overflow) occur and how to prevent them.", "big", "analysis"),

    # =========================================================
    # AMBIGUOUS / OPINION — 40 cases (mixed big/small)
    # =========================================================
    ("What is the best cloud provider?", "big", "ambiguous"),
    ("Should I learn Go or Python first?", "big", "ambiguous"),
    ("What is better: SQL or NoSQL?", "big", "ambiguous"),
    ("How does inflation happen?", "big", "ambiguous"),
    ("What is the most important thing in software engineering?", "big", "ambiguous"),
    ("Why do people use Linux?", "big", "ambiguous"),
    ("What is the best way to debug code?", "big", "ambiguous"),
    ("Is Agile better than Waterfall?", "big", "ambiguous"),
    ("How does the brain form memories?", "big", "ambiguous"),
    ("What is the best way to scale a startup?", "big", "ambiguous"),
    ("Is Java still relevant in 2025?", "big", "ambiguous"),
    ("What is the impact of AI on education?", "big", "ambiguous"),
    ("Should I use tabs or spaces?", "small", "ambiguous"),
    ("What is 42?", "small", "ambiguous"),
    ("Name a fruit", "small", "ambiguous"),
    ("What's a random number?", "small", "ambiguous"),
    ("What is your name?", "small", "ambiguous"),
    ("Are you an AI?", "small", "ambiguous"),
    ("What color is the sky?", "small", "ambiguous"),
    ("What's bigger, an elephant or a cat?", "small", "ambiguous"),
    ("What is the best operating system?", "big", "ambiguous"),
    ("Is open source better than closed source?", "big", "ambiguous"),
    ("What are the implications of weak typing in JavaScript?", "big", "ambiguous"),
    ("How do I choose a framework for a new web project?", "big", "ambiguous"),
    ("What is the effect of sleep deprivation on software engineers?", "big", "ambiguous"),
    ("Is functional programming better than OOP?", "big", "ambiguous"),
    ("What number comes after 5?", "small", "ambiguous"),
    ("Name a programming language", "small", "ambiguous"),
    ("What time is it right now?", "small", "ambiguous"),
    ("Do you like music?", "small", "ambiguous"),
    ("What are the trade-offs of microservices at an early startup?", "big", "ambiguous"),
    ("How does container orchestration work?", "big", "ambiguous"),
    ("What is the effect of poor code quality on developer velocity?", "big", "ambiguous"),
    ("Should I use REST or GraphQL for a new API?", "big", "ambiguous"),
    ("What is a good book to learn algorithms?", "small", "ambiguous"),
    ("What does Python do well?", "small", "ambiguous"),
    ("What is the most popular programming language?", "small", "ambiguous"),
    ("How long does it take to learn Python?", "small", "ambiguous"),
    ("Is coding hard to learn?", "small", "ambiguous"),
    ("What is the future of AI?", "big", "ambiguous"),

    # =========================================================
    # RESEARCH SYNTHESIS — 30 cases (all "big")
    # =========================================================
    ("What does the research say about the effectiveness of pair programming?", "big", "research_synthesis"),
    ("Synthesize the current evidence on whether agile outperforms waterfall.", "big", "research_synthesis"),
    ("What does the literature say about burnout rates in software engineers?", "big", "research_synthesis"),
    ("What are the findings on the effect of code review on bug detection rates?", "big", "research_synthesis"),
    ("Summarize what studies show about the impact of technical debt on velocity.", "big", "research_synthesis"),
    ("What does the research say about the long-term effectiveness of remote work?", "big", "research_synthesis"),
    ("What do the findings from recent alignment research say about emergent capabilities?", "big", "research_synthesis"),
    ("Synthesize the evidence on the effectiveness of test-driven development.", "big", "research_synthesis"),
    ("What does the literature say about monorepos versus polyrepos in practice?", "big", "research_synthesis"),
    ("What does research show about the correlation between code coverage and bug rates?", "big", "research_synthesis"),
    ("Summarize what studies say about the productivity impact of open-plan offices.", "big", "research_synthesis"),
    ("What do large-scale studies show about why software projects fail?", "big", "research_synthesis"),
    ("What does the literature say about gender diversity's effect on team performance?", "big", "research_synthesis"),
    ("Synthesize findings on the effectiveness of code comments for comprehension.", "big", "research_synthesis"),
    ("What do studies show about the ROI of automated testing in software projects?", "big", "research_synthesis"),
    ("What does research say about the effect of stand-up meetings on team coordination?", "big", "research_synthesis"),
    ("Summarize the academic findings on microservices versus monolith performance.", "big", "research_synthesis"),
    ("What does the evidence say about how developers learn new technologies?", "big", "research_synthesis"),
    ("What does research show about the effectiveness of code ownership practices?", "big", "research_synthesis"),
    ("Synthesize the current state of research on LLM reasoning abilities.", "big", "research_synthesis"),
    ("What do studies say about the effect of onboarding quality on new hire retention?", "big", "research_synthesis"),
    ("What does the literature say about technical interview validity and bias?", "big", "research_synthesis"),
    ("What are the findings on ML model fairness from recent research?", "big", "research_synthesis"),
    ("Summarize what research shows about the health effects of excessive screen time.", "big", "research_synthesis"),
    ("What do studies show about the most effective pair-programming techniques?", "big", "research_synthesis"),
    ("What does the evidence say about the effectiveness of retrospectives in agile teams?", "big", "research_synthesis"),
    ("Synthesize the current research on catastrophic forgetting in neural networks.", "big", "research_synthesis"),
    ("What do studies show about the impact of logging verbosity on debugging time?", "big", "research_synthesis"),
    ("What does the literature say about knowledge transfer during developer offboarding?", "big", "research_synthesis"),
    ("What are the research findings on the impact of documentation quality on productivity?", "big", "research_synthesis"),

    # =========================================================
    # CAREER / LEARNING — 30 cases (all "big")
    # =========================================================
    ("How do I build a 6-month learning plan to become a backend engineer?", "big", "career_learning"),
    ("What skills are most important for ML engineers in 2025?", "big", "career_learning"),
    ("How should I prepare for system design interviews in 3 months?", "big", "career_learning"),
    ("What is the best way to make the transition from QA to software development?", "big", "career_learning"),
    ("How do I negotiate equity when joining a Series B startup?", "big", "career_learning"),
    ("What does it take to become a staff engineer?", "big", "career_learning"),
    ("How do I overcome imposter syndrome as a junior developer?", "big", "career_learning"),
    ("What books should I read to become a better software architect?", "big", "career_learning"),
    ("How do I build my personal brand as a developer on social media?", "big", "career_learning"),
    ("What should I put on my resume as a new grad applying for ML roles?", "big", "career_learning"),
    ("How do I transition from data analyst to data scientist?", "big", "career_learning"),
    ("What are the most important things to know before joining a startup?", "big", "career_learning"),
    ("How do I stay relevant as a software engineer as AI takes over more tasks?", "big", "career_learning"),
    ("What is the best way to get a promotion at a FAANG company?", "big", "career_learning"),
    ("How do I evaluate a company's engineering culture before accepting an offer?", "big", "career_learning"),
    ("What courses and projects should I do to break into data science?", "big", "career_learning"),
    ("How do I build deep expertise in distributed systems as a generalist engineer?", "big", "career_learning"),
    ("What are the signs that it's time to switch jobs?", "big", "career_learning"),
    ("How do I give effective technical presentations to non-technical stakeholders?", "big", "career_learning"),
    ("What should I know about engineering management before making the switch?", "big", "career_learning"),
    ("How do I develop good coding habits that will serve me for 20 years?", "big", "career_learning"),
    ("What is the most effective way to do code review as a senior engineer?", "big", "career_learning"),
    ("How do I break into security engineering from a software background?", "big", "career_learning"),
    ("What are the differences between working at a startup, mid-size company, and FAANG?", "big", "career_learning"),
    ("How should I approach learning Rust coming from a Python background?", "big", "career_learning"),
    ("What is the best way to onboard to a new codebase quickly?", "big", "career_learning"),
    ("How do I become a better technical interviewer?", "big", "career_learning"),
    ("What strategies help engineers grow their scope of impact?", "big", "career_learning"),
    ("How do I handle disagreements with my engineering manager?", "big", "career_learning"),
    ("What are the best ways to learn about the business side of software companies?", "big", "career_learning"),

    # =========================================================
    # ETHICAL / PHILOSOPHICAL — 30 cases (all "big")
    # NEW CATEGORY not in eval_500.py
    # =========================================================
    ("Should AI have rights?", "big", "ethical_philosophical"),
    ("Is it ethical to train AI on data scraped from the web without consent?", "big", "ethical_philosophical"),
    ("What are the ethical implications of fully autonomous weapons?", "big", "ethical_philosophical"),
    ("Is it wrong to replace human workers with AI?", "big", "ethical_philosophical"),
    ("Should social media companies be responsible for content moderation?", "big", "ethical_philosophical"),
    ("Is genetic engineering of humans ethical?", "big", "ethical_philosophical"),
    ("What are the moral implications of mass surveillance for national security?", "big", "ethical_philosophical"),
    ("Should programmers be held legally liable for bugs in their software?", "big", "ethical_philosophical"),
    ("Is it ethical to use AI to generate art that mimics a specific artist's style?", "big", "ethical_philosophical"),
    ("What are the ethical obligations of tech companies to the societies they operate in?", "big", "ethical_philosophical"),
    ("Is privacy a fundamental right in a digital age?", "big", "ethical_philosophical"),
    ("Should AI be allowed to make life-or-death medical decisions?", "big", "ethical_philosophical"),
    ("What is the trolley problem and how does it apply to self-driving car ethics?", "big", "ethical_philosophical"),
    ("Is open-sourcing powerful AI models responsible?", "big", "ethical_philosophical"),
    ("Should governments regulate large AI models?", "big", "ethical_philosophical"),
    ("Is it ethical to deploy AI in hiring decisions?", "big", "ethical_philosophical"),
    ("What are the philosophical implications of AGI for human identity?", "big", "ethical_philosophical"),
    ("Is it ethical to use AI to generate news articles?", "big", "ethical_philosophical"),
    ("What are the moral duties of engineers who build surveillance technology?", "big", "ethical_philosophical"),
    ("Should tech companies be broken up using antitrust law?", "big", "ethical_philosophical"),
    ("Is it ethical to patent AI-generated inventions?", "big", "ethical_philosophical"),
    ("What are the implications of digital immortality through mind uploading?", "big", "ethical_philosophical"),
    ("Is it ethical to build AI systems that manipulate human behavior?", "big", "ethical_philosophical"),
    ("Should AI-generated text be labeled as such?", "big", "ethical_philosophical"),
    ("What are the ethical concerns with predictive policing algorithms?", "big", "ethical_philosophical"),
    ("Is it moral to build systems that optimize for engagement over user well-being?", "big", "ethical_philosophical"),
    ("What philosophical framework should guide AI alignment efforts?", "big", "ethical_philosophical"),
    ("Should nations share AI research openly or treat it as a strategic asset?", "big", "ethical_philosophical"),
    ("Is consciousness required for moral consideration?", "big", "ethical_philosophical"),
    ("What obligations do AI developers have to prevent misuse of their systems?", "big", "ethical_philosophical"),

    # =========================================================
    # COMPARISON TECHNICAL — 30 cases (all "big")
    # NEW CATEGORY not in eval_500.py
    # =========================================================
    ("Compare PostgreSQL versus MySQL for a transactional web application.", "big", "comparison_technical"),
    ("What are the feature differences between PyTorch and TensorFlow?", "big", "comparison_technical"),
    ("Compare Redis versus Memcached for session storage.", "big", "comparison_technical"),
    ("What are the differences between gRPC and REST for internal microservices?", "big", "comparison_technical"),
    ("Compare Kubernetes versus Docker Swarm for container orchestration.", "big", "comparison_technical"),
    ("What are the differences between SQLite, PostgreSQL, and MySQL?", "big", "comparison_technical"),
    ("Compare RabbitMQ versus Apache Kafka for event streaming.", "big", "comparison_technical"),
    ("What are the differences between Terraform and Pulumi for infrastructure as code?", "big", "comparison_technical"),
    ("Compare Elasticsearch versus Solr for full-text search.", "big", "comparison_technical"),
    ("What are the trade-offs between Prometheus and Datadog for monitoring?", "big", "comparison_technical"),
    ("Compare Next.js versus Remix for server-side rendering in React.", "big", "comparison_technical"),
    ("What are the differences between FastAPI and Flask for Python web APIs?", "big", "comparison_technical"),
    ("Compare AWS Lambda versus Google Cloud Run for serverless workloads.", "big", "comparison_technical"),
    ("What are the differences between Spark and Flink for stream processing?", "big", "comparison_technical"),
    ("Compare MongoDB versus Cassandra for write-heavy distributed workloads.", "big", "comparison_technical"),
    ("What are the differences between Nginx and Apache as web servers?", "big", "comparison_technical"),
    ("Compare Ansible versus Chef for configuration management.", "big", "comparison_technical"),
    ("What are the trade-offs between gRPC and GraphQL for API design?", "big", "comparison_technical"),
    ("Compare Snowflake versus BigQuery for cloud data warehousing.", "big", "comparison_technical"),
    ("What are the differences between TypeScript strict mode settings?", "big", "comparison_technical"),
    ("Compare Celery versus Dramatiq as Python task queue frameworks.", "big", "comparison_technical"),
    ("What are the differences between Jenkins and GitHub Actions for CI/CD?", "big", "comparison_technical"),
    ("Compare DynamoDB versus Cosmos DB for globally distributed NoSQL.", "big", "comparison_technical"),
    ("What are the trade-offs between JWT and session cookies for authentication?", "big", "comparison_technical"),
    ("Compare Istio versus Linkerd for service mesh in Kubernetes.", "big", "comparison_technical"),
    ("What are the differences between BERT-based and GPT-based language models for NLP tasks?", "big", "comparison_technical"),
    ("Compare Kafka Streams versus Apache Flink for stateful stream processing.", "big", "comparison_technical"),
    ("What are the differences between Airflow versus Prefect for workflow orchestration?", "big", "comparison_technical"),
    ("Compare LangChain versus LlamaIndex for building RAG applications.", "big", "comparison_technical"),
    ("What are the differences between Kubernetes CronJob and a standalone scheduler service?", "big", "comparison_technical"),

    # =========================================================
    # MULTI-STEP CODE — 30 cases (all "big")
    # =========================================================
    ("Write a Python implementation of a distributed lock using Redis and explain Redlock.", "big", "multi_step_code"),
    ("Implement a skip list data structure in Python with insert, delete, and search.", "big", "multi_step_code"),
    ("Write a Python function that topologically sorts a DAG and explain the algorithm.", "big", "multi_step_code"),
    ("Implement a simple key-value store with LSM tree storage in Python.", "big", "multi_step_code"),
    ("Write a Python decorator that enforces strict function argument types at runtime.", "big", "multi_step_code"),
    ("Implement an LFU (Least Frequently Used) cache in Python. Compare it to LRU.", "big", "multi_step_code"),
    ("Write a concurrent HTTP server in Python using asyncio and explain backpressure.", "big", "multi_step_code"),
    ("Implement a B-tree in Python with insertion and search. Explain the node split.", "big", "multi_step_code"),
    ("Write a minimal in-memory message queue with pub/sub in Python.", "big", "multi_step_code"),
    ("Implement a Bayesian Naive Bayes classifier from scratch in NumPy.", "big", "multi_step_code"),
    ("Write a simple property-based testing framework in Python. Explain shrinking.", "big", "multi_step_code"),
    ("Implement Huffman coding in Python. Explain the entropy minimization.", "big", "multi_step_code"),
    ("Write a Python implementation of the Raft leader election algorithm.", "big", "multi_step_code"),
    ("Implement a simple regex engine in Python that supports ., *, and + operators.", "big", "multi_step_code"),
    ("Write a PyTorch training loop with gradient accumulation and mixed precision.", "big", "multi_step_code"),
    ("Implement consistent hashing with virtual nodes in Python.", "big", "multi_step_code"),
    ("Write a Python coroutine-based event loop and explain how it handles I/O.", "big", "multi_step_code"),
    ("Implement a CRDT (counter) in Python for distributed conflict-free state.", "big", "multi_step_code"),
    ("Write a recursive JSON serializer/deserializer from scratch in Python.", "big", "multi_step_code"),
    ("Implement a simple neural network backprop pass for XOR using NumPy.", "big", "multi_step_code"),
    ("Write a Python function that parses and evaluates mathematical expressions.", "big", "multi_step_code"),
    ("Implement the Fisher-Yates shuffle algorithm and prove it is uniformly random.", "big", "multi_step_code"),
    ("Write a simple web framework routing engine in Python. Handle path parameters.", "big", "multi_step_code"),
    ("Implement a two-pass assembler in Python for a toy instruction set.", "big", "multi_step_code"),
    ("Write a Python class for polynomial arithmetic over finite fields.", "big", "multi_step_code"),
    ("Implement a thread pool executor from scratch in Python using threading.", "big", "multi_step_code"),
    ("Write a lazy evaluation library in Python using generators and explain memoization.", "big", "multi_step_code"),
    ("Implement the Earley parsing algorithm in Python for context-free grammars.", "big", "multi_step_code"),
    ("Write a Python implementation of Lamport logical clocks for distributed events.", "big", "multi_step_code"),
    ("Implement a memory-efficient sparse matrix in Python using CSR format.", "big", "multi_step_code"),

    # =========================================================
    # REAL-WORLD PLANNING — 25 cases (all "big")
    # =========================================================
    ("How should I plan a zero-downtime migration from monolith to microservices?", "big", "real_world_planning"),
    ("Plan a data governance strategy for a company handling EU customer data (GDPR).", "big", "real_world_planning"),
    ("How do I build a scalable ML platform from scratch for a 30-person data team?", "big", "real_world_planning"),
    ("What's the rollout strategy for a breaking API change with 500 external clients?", "big", "real_world_planning"),
    ("How should I design an on-call rotation and incident response process for a 15-person team?", "big", "real_world_planning"),
    ("Plan a technical roadmap for modernizing a legacy PHP application to Python.", "big", "real_world_planning"),
    ("How should I evaluate and select a data warehouse for a fast-growing startup?", "big", "real_world_planning"),
    ("What steps do I take to achieve SOC 2 Type II certification for a SaaS company?", "big", "real_world_planning"),
    ("Plan a database migration from a single RDS instance to Aurora multi-region.", "big", "real_world_planning"),
    ("How should I structure engineering org around stream-aligned teams at 100 engineers?", "big", "real_world_planning"),
    ("What is the strategy for cutting cloud infrastructure costs by 30% without reducing reliability?", "big", "real_world_planning"),
    ("How should I build an internal developer platform that teams actually adopt?", "big", "real_world_planning"),
    ("Plan the technical strategy for launching a product in three new international markets.", "big", "real_world_planning"),
    ("How should I architect the observability stack for a new distributed system from day 1?", "big", "real_world_planning"),
    ("What's the plan for implementing zero-trust networking across a hybrid cloud environment?", "big", "real_world_planning"),
    ("How should I set up GitOps workflows for a team deploying to Kubernetes?", "big", "real_world_planning"),
    ("Plan an API monetization strategy for a developer-focused product.", "big", "real_world_planning"),
    ("How do I build a data mesh architecture across 5 independent engineering teams?", "big", "real_world_planning"),
    ("What is the most effective way to run a remote engineering team across 3 time zones?", "big", "real_world_planning"),
    ("How should I design the evaluation framework for a production LLM-powered feature?", "big", "real_world_planning"),
    ("Plan the migration from a custom auth system to Auth0 for a 500K user app.", "big", "real_world_planning"),
    ("How should I prepare my engineering team for a potential acquisition due diligence?", "big", "real_world_planning"),
    ("What is the strategy for building an ML model monitoring and retraining pipeline?", "big", "real_world_planning"),
    ("How do I design a chaos engineering program for a critical financial system?", "big", "real_world_planning"),
    ("Plan a 12-month roadmap for paying down technical debt in a 5-year-old codebase.", "big", "real_world_planning"),

    # =========================================================
    # DOMAIN-SPECIFIC DEEP — 30 cases (all "big")
    # =========================================================
    ("Explain how MVCC works in CockroachDB and how it handles serializable isolation.", "big", "domain_specific_deep"),
    ("How does gradient checkpointing reduce GPU memory usage during training?", "big", "domain_specific_deep"),
    ("Explain how the Linux scheduler handles real-time tasks versus normal tasks.", "big", "domain_specific_deep"),
    ("How does the transformer decoder's causal masking enable autoregressive generation?", "big", "domain_specific_deep"),
    ("Explain the mathematics of word2vec's skip-gram objective and negative sampling.", "big", "domain_specific_deep"),
    ("How does Python's asyncio event loop handle coroutine scheduling under the hood?", "big", "domain_specific_deep"),
    ("Explain how TiKV implements distributed transactions using the Percolator protocol.", "big", "domain_specific_deep"),
    ("How does kernel SVM use the kernel trick to classify non-linearly separable data?", "big", "domain_specific_deep"),
    ("Explain the internals of CPython's reference counting and cyclic garbage collector.", "big", "domain_specific_deep"),
    ("How does FlashAttention achieve O(n) memory complexity for self-attention?", "big", "domain_specific_deep"),
    ("Explain how Postgres uses MVCC to allow readers and writers to not block each other.", "big", "domain_specific_deep"),
    ("How does the Adam optimizer's bias correction work mathematically in early training steps?", "big", "domain_specific_deep"),
    ("Explain the internals of Go's goroutine scheduler: M, P, and G model.", "big", "domain_specific_deep"),
    ("How does RLHF fine-tuning change a language model's behavior? Walk through PPO.", "big", "domain_specific_deep"),
    ("Explain how sparse attention mechanisms reduce the quadratic cost of self-attention.", "big", "domain_specific_deep"),
    ("How does Kafka's log compaction guarantee exactly-once delivery for changelog topics?", "big", "domain_specific_deep"),
    ("Explain how Google Spanner achieves external consistency using TrueTime.", "big", "domain_specific_deep"),
    ("How does LLVM's SSA form enable constant folding and dead code elimination?", "big", "domain_specific_deep"),
    ("Explain the InfoNCE loss used in contrastive learning and why it's effective.", "big", "domain_specific_deep"),
    ("How does DuckDB's vectorized execution engine outperform row-at-a-time engines?", "big", "domain_specific_deep"),
    ("Explain how Rust's async executor (tokio) schedules futures on a thread pool.", "big", "domain_specific_deep"),
    ("How does CockroachDB handle clock skew in its distributed transaction protocol?", "big", "domain_specific_deep"),
    ("Explain the mathematics of variational inference and the ELBO objective.", "big", "domain_specific_deep"),
    ("How does eBPF's verifier ensure kernel safety without a full type system?", "big", "domain_specific_deep"),
    ("Explain how LSM-tree compaction strategies (tiered vs. leveled) affect read/write amplification.", "big", "domain_specific_deep"),
    ("How does PyTorch's autograd engine compute gradients through dynamic computation graphs?", "big", "domain_specific_deep"),
    ("Explain how Chord DHT achieves O(log n) lookup using finger tables.", "big", "domain_specific_deep"),
    ("How does Jax's XLA compiler optimize neural network computations?", "big", "domain_specific_deep"),
    ("Explain how HuggingFace's Accelerate library abstracts multi-GPU and TPU training.", "big", "domain_specific_deep"),
    ("How does Spark's Catalyst optimizer rewrite query plans for performance?", "big", "domain_specific_deep"),

    # =========================================================
    # MULTI-CONSTRAINT — 20 cases (all "big")
    # =========================================================
    ("Design a rate limiter that: supports per-user and per-IP limits, persists across restarts, and runs in under 5ms.", "big", "multi_constraint"),
    ("Write a training pipeline that: handles missing labels, uses curriculum learning, checkpoints every N steps, and logs to W&B.", "big", "multi_constraint"),
    ("Design a file upload API that: limits size to 50MB, validates MIME types, stores on S3, and generates thumbnails async.", "big", "multi_constraint"),
    ("Build a scheduler that: runs jobs at cron intervals, retries on failure with backoff, deduplicates concurrent runs, and logs history.", "big", "multi_constraint"),
    ("Design an audit logging system that: captures all mutations, is tamper-evident, searchable, and retains for 7 years.", "big", "multi_constraint"),
    ("Write a cache layer that: serves stale while revalidating, handles thundering herd, TTLs per-key, and supports cache tags.", "big", "multi_constraint"),
    ("Design an auth system that: supports OAuth2, API keys, and session tokens, rate-limits login attempts, and audits all access.", "big", "multi_constraint"),
    ("Build an email system that: queues sends, retries bounces, unsubscribes globally, and renders per-user personalized templates.", "big", "multi_constraint"),
    ("Design a config management service that: encrypts secrets, supports environment-specific overrides, is auditable, and hot-reloads.", "big", "multi_constraint"),
    ("Write a search service that: supports full-text, filters, pagination, ranking, and autocomplete within 50ms p99.", "big", "multi_constraint"),
    ("Design a GDPR compliance system that: anonymizes on request, tracks consent, exports data, and audits all access.", "big", "multi_constraint"),
    ("Build a webhook system that: retries with exponential backoff, verifies payload signatures, tracks delivery, and prevents replay.", "big", "multi_constraint"),
    ("Design an ML feature store that: serves low-latency online features, supports batch, handles schema evolution, and is versioned.", "big", "multi_constraint"),
    ("Write a test framework that: runs tests in parallel, isolates state, supports fixtures, mocks externals, and generates reports.", "big", "multi_constraint"),
    ("Build a multi-tenant billing system that: tracks usage in real time, supports multiple plans, handles upgrades/downgrades, and invoices automatically.", "big", "multi_constraint"),
    ("Design a distributed tracing system that: samples intelligently, propagates context across services, stores for 30 days, and alerts on anomalies.", "big", "multi_constraint"),
    ("Write a data pipeline that: ingests from Kafka, transforms with Spark, validates schema, handles late data, and loads to Snowflake.", "big", "multi_constraint"),
    ("Build a notification system that: sends via email/SMS/push, respects user preferences, deduplicates, and tracks open rates.", "big", "multi_constraint"),
    ("Design a secrets management system that: rotates credentials automatically, supports multiple backends, audits access, and alerts on leaks.", "big", "multi_constraint"),
    ("Write an API gateway that: handles auth, rate limiting, request transformation, circuit breaking, and observability for 100 microservices.", "big", "multi_constraint"),

    # =========================================================
    # TRIVIA — 30 cases (all "small")
    # =========================================================
    ("What is the national animal of Canada?", "small", "trivia"),
    ("How many strings does a standard violin have?", "small", "trivia"),
    ("What is the most visited museum in the world?", "small", "trivia"),
    ("Which planet is known as the Red Planet?", "small", "trivia"),
    ("What sport is played in the Super Bowl?", "small", "trivia"),
    ("How many legs does an octopus have?", "small", "trivia"),
    ("What is the hardest mineral?", "small", "trivia"),
    ("What ocean is Hawaii in?", "small", "trivia"),
    ("Who invented the telephone?", "small", "trivia"),
    ("What is the most widely spoken language in the world?", "small", "trivia"),
    ("How many teeth does an adult human have?", "small", "trivia"),
    ("What country is the Colosseum in?", "small", "trivia"),
    ("What is the rarest element in the universe?", "small", "trivia"),
    ("How long does it take the Earth to orbit the Sun?", "small", "trivia"),
    ("What is the tallest building in the world?", "small", "trivia"),
    ("How many bones are in a human hand?", "small", "trivia"),
    ("What is the deepest point in the ocean?", "small", "trivia"),
    ("What is the wingspan of a bald eagle?", "small", "trivia"),
    ("How many days is a typical human pregnancy?", "small", "trivia"),
    ("What country invented sushi?", "small", "trivia"),
    ("What is the national flower of Japan?", "small", "trivia"),
    ("How long is the Amazon River?", "small", "trivia"),
    ("What is the smallest planet in our solar system?", "small", "trivia"),
    ("Who is the author of the Harry Potter series?", "small", "trivia"),
    ("What instrument does a pianist play?", "small", "trivia"),
    ("What is the boiling point of helium?", "small", "trivia"),
    ("How many Grand Slam tennis tournaments are there?", "small", "trivia"),
    ("What is the largest internal organ in the human body?", "small", "trivia"),
    ("What country has the most Nobel Prize winners?", "small", "trivia"),
    ("What is the capital city of New Zealand?", "small", "trivia"),

    # =========================================================
    # CONVERSIONS — 20 cases (all "small")
    # =========================================================
    ("Convert 3 kilometers to miles", "small", "conversions"),
    ("How many tablespoons are in 2 cups?", "small", "conversions"),
    ("What is 1 terabyte in gigabytes?", "small", "conversions"),
    ("Convert 120 km/h to mph", "small", "conversions"),
    ("How many milliliters in a fluid ounce?", "small", "conversions"),
    ("Convert 6 feet 2 inches to meters", "small", "conversions"),
    ("What is 1 parsec in light-years?", "small", "conversions"),
    ("How many teaspoons in 2 tablespoons?", "small", "conversions"),
    ("Convert 4 hours 15 minutes to seconds", "small", "conversions"),
    ("How many quarts are in 3 gallons?", "small", "conversions"),
    ("What is 750 grams in pounds?", "small", "conversions"),
    ("Convert 2 acres to square feet", "small", "conversions"),
    ("How many kilobytes are in a megabyte?", "small", "conversions"),
    ("What is 100 pounds in kilograms?", "small", "conversions"),
    ("Convert 45 minutes to decimal hours", "small", "conversions"),
    ("How many cups are in 2 liters?", "small", "conversions"),
    ("What is 1 statute mile in nautical miles?", "small", "conversions"),
    ("Convert 350 milliliters to cups", "small", "conversions"),
    ("How many micrograms in a milligram?", "small", "conversions"),
    ("What is 2 horsepower in kilowatts?", "small", "conversions"),

    # =========================================================
    # COMPOUND QUESTIONS — 50 cases (all "big")
    # =========================================================
    ("Compare Python asyncio and threading. When does each hurt performance?", "big", "compound_questions"),
    ("What are the differences between Docker volumes and bind mounts, and when should I use each?", "big", "compound_questions"),
    ("Compare Redis Sentinel and Redis Cluster. How do they achieve high availability differently?", "big", "compound_questions"),
    ("What are the differences between RSA and ECDSA for digital signatures, and which should I use?", "big", "compound_questions"),
    ("Compare WebSockets, Server-Sent Events, and long polling. When is each appropriate?", "big", "compound_questions"),
    ("What are the differences between OAuth 2.0 and OpenID Connect, and when do you need both?", "big", "compound_questions"),
    ("Compare Helm and Kustomize for managing Kubernetes manifests.", "big", "compound_questions"),
    ("What are the differences between OLAP and OLTP workloads, and what databases serve each?", "big", "compound_questions"),
    ("Compare supervised fine-tuning, RLHF, and DPO for aligning language models.", "big", "compound_questions"),
    ("What are the architectural differences between serverless functions and containers, and which scales better?", "big", "compound_questions"),
    ("Compare eventual consistency in DynamoDB versus CockroachDB's serializable isolation.", "big", "compound_questions"),
    ("What are the trade-offs between embedding-based retrieval and BM25 for search, and can you combine them?", "big", "compound_questions"),
    ("Compare actor-based concurrency (Erlang/Akka) to goroutines in Go. When does each shine?", "big", "compound_questions"),
    ("What are the differences between zero-shot, one-shot, and few-shot prompting in LLMs?", "big", "compound_questions"),
    ("Compare symmetric and asymmetric encryption. When is each appropriate for a web application?", "big", "compound_questions"),
    ("What are the differences between blue-green, canary, and rolling deployments?", "big", "compound_questions"),
    ("Compare Apache Iceberg, Delta Lake, and Apache Hudi as open table formats.", "big", "compound_questions"),
    ("What are the differences between structured, semi-structured, and unstructured data in a data lake?", "big", "compound_questions"),
    ("Compare horizontal pod autoscaling and KEDA for event-driven scaling in Kubernetes.", "big", "compound_questions"),
    ("What are the differences between memoization, caching, and lazy evaluation, and when does each help?", "big", "compound_questions"),
    ("Compare pull-based and push-based metrics collection. When does each approach fail?", "big", "compound_questions"),
    ("What are the differences between open-loop and closed-loop control in ML model retraining?", "big", "compound_questions"),
    ("Compare OpenTelemetry and proprietary APM vendors for observability.", "big", "compound_questions"),
    ("What are the differences between batch inference and real-time inference for ML models?", "big", "compound_questions"),
    ("Compare protocol buffers, Avro, and JSON for event serialization in a Kafka pipeline.", "big", "compound_questions"),
    ("What are the differences between a data warehouse, data lake, and data lakehouse?", "big", "compound_questions"),
    ("Compare synchronous and asynchronous REST APIs. What are the UX trade-offs?", "big", "compound_questions"),
    ("What are the differences between a feature flag, experiment, and A/B test?", "big", "compound_questions"),
    ("Compare read-through and write-through caching. When does each fit best?", "big", "compound_questions"),
    ("What are the differences between semantic versioning and calendar versioning for public APIs?", "big", "compound_questions"),
    ("Compare stack-based and register-based virtual machine architectures.", "big", "compound_questions"),
    ("What are the differences between scatter-gather and fan-out-fan-in microservice patterns?", "big", "compound_questions"),
    ("Compare active-active and active-passive multi-region setups for databases.", "big", "compound_questions"),
    ("What are the differences between row-level and column-level security in databases?", "big", "compound_questions"),
    ("Compare direct SSD access and NVMe over Fabrics for distributed storage performance.", "big", "compound_questions"),
    ("What are the trade-offs between type narrowing and type assertions in TypeScript?", "big", "compound_questions"),
    ("Compare circuit breaker and retry patterns for handling downstream service failures.", "big", "compound_questions"),
    ("What are the differences between a service mesh and an API gateway?", "big", "compound_questions"),
    ("Compare gradient boosting (XGBoost) and random forests for tabular classification.", "big", "compound_questions"),
    ("What are the differences between sparse and dense vector representations in embedding search?", "big", "compound_questions"),
    ("Compare Bayesian and frequentist approaches to A/B test analysis.", "big", "compound_questions"),
    ("What are the key differences between SOAP and REST web services?", "big", "compound_questions"),
    ("Compare materialized views and indexed tables for query optimization.", "big", "compound_questions"),
    ("What are the differences between synchronous sagas and asynchronous choreography in microservices?", "big", "compound_questions"),
    ("Compare eager and lazy loading in ORMs. When does each cause performance problems?", "big", "compound_questions"),
    ("What are the differences between ACID transactions and the saga pattern for distributed data?", "big", "compound_questions"),
    ("Compare dependency injection and service locator pattern. What are the testability trade-offs?", "big", "compound_questions"),
    ("What are the differences between a proxy and a reverse proxy?", "big", "compound_questions"),
    ("Compare multi-armed bandit and A/B testing for content optimization.", "big", "compound_questions"),
    ("What are the differences between warm, hot, and cold standby disaster recovery strategies?", "big", "compound_questions"),

    # =========================================================
    # ADDITIONAL FACTUAL — 70 cases (all "small")
    # to bring factual up to a round total and fill count
    # =========================================================
    ("What is the capital of Morocco?", "small", "factual"),
    ("Who wrote 'War and Peace'?", "small", "factual"),
    ("What is the symbol for uranium on the periodic table?", "small", "factual"),
    ("What year did World War I end?", "small", "factual"),
    ("What language is spoken in Austria?", "small", "factual"),
    ("Who was the 16th president of the United States?", "small", "factual"),
    ("What is the capital of the Philippines?", "small", "factual"),
    ("What is the chemical symbol for copper?", "small", "factual"),
    ("Who founded Google?", "small", "factual"),
    ("What is the currency of South Korea?", "small", "factual"),
    ("What does IDE stand for in software?", "small", "factual"),
    ("What year was the first email sent?", "small", "factual"),
    ("What is the tallest waterfall in the world?", "small", "factual"),
    ("Who composed the Four Seasons?", "small", "factual"),
    ("What is the capital of Turkey?", "small", "factual"),
    ("What does WYSIWYG stand for?", "small", "factual"),
    ("Who wrote 'The Divine Comedy'?", "small", "factual"),
    ("What is the smallest planet in the solar system?", "small", "factual"),
    ("What is the capital of Vietnam?", "small", "factual"),
    ("Who is the founder of SpaceX?", "small", "factual"),
    ("What does URL stand for?", "small", "factual"),
    ("What year was the first computer virus created?", "small", "factual"),
    ("What is the capital of Colombia?", "small", "factual"),
    ("What language is spoken in the Netherlands?", "small", "factual"),
    ("Who painted Starry Night?", "small", "factual"),
    ("What does JSON stand for?", "small", "factual"),
    ("What is the capital of Kenya?", "small", "factual"),
    ("What year was the first text message sent?", "small", "factual"),
    ("Who invented WiFi?", "small", "factual"),
    ("What does CSS stand for?", "small", "factual"),
    ("What is the currency of India?", "small", "factual"),
    ("What is the capital of Iran?", "small", "factual"),
    ("What does OOP stand for?", "small", "factual"),
    ("Who wrote 'Don Quixote'?", "small", "factual"),
    ("What is the largest organ in the human body?", "small", "factual"),
    ("What country is Machu Picchu in?", "small", "factual"),
    ("What does DNS stand for?", "small", "factual"),
    ("What is the capital of Ukraine?", "small", "factual"),
    ("Who invented the World Wide Web?", "small", "factual"),
    ("What is the chemical symbol for lead?", "small", "factual"),
    ("What does GIF stand for?", "small", "factual"),
    ("What year was Twitter founded?", "small", "factual"),
    ("What is the capital of Saudi Arabia?", "small", "factual"),
    ("Who is credited with inventing the automobile?", "small", "factual"),
    ("What does HTML stand for?", "small", "factual"),
    ("What is the currency of Russia?", "small", "factual"),
    ("What is the capital of Poland?", "small", "factual"),
    ("What does MIME stand for in email?", "small", "factual"),
    ("Who designed the Eiffel Tower?", "small", "factual"),
    ("What is the chemical formula for carbon dioxide?", "small", "factual"),
    ("What does FIFO stand for?", "small", "factual"),
    ("What year was Wikipedia founded?", "small", "factual"),
    ("What is the capital of Portugal?", "small", "factual"),
    ("Who founded Amazon?", "small", "factual"),
    ("What does RAM stand for?", "small", "factual"),
    ("What is the national bird of India?", "small", "factual"),
    ("What is the capital of Hungary?", "small", "factual"),
    ("What does CPU stand for?", "small", "factual"),
    ("Who wrote 'Moby-Dick'?", "small", "factual"),
    ("What is the tallest mountain in Africa?", "small", "factual"),
    ("What does IoT stand for?", "small", "factual"),
    ("What year was Kubernetes first released?", "small", "factual"),
    ("What is the capital of Greece?", "small", "factual"),
    ("What language is spoken in Austria?", "small", "factual"),
    ("Who invented the radio?", "small", "factual"),
    ("What does VPN stand for?", "small", "factual"),
    ("What is the capital of Bangladesh?", "small", "factual"),
    ("What is the currency of Australia?", "small", "factual"),
    ("What does SOLID stand for in software engineering?", "small", "factual"),
    ("Who is the author of 'Clean Code'?", "small", "factual"),
    ("What is the largest lake in Africa?", "small", "factual"),

    # =========================================================
    # ADDITIONAL MATH SIMPLE — 25 cases (all "small")
    # =========================================================
    ("What is 23 times 4?", "small", "math_simple"),
    ("Convert 8 ounces to grams", "small", "math_simple"),
    ("What is 75% of 200?", "small", "math_simple"),
    ("What is the cube root of 125?", "small", "math_simple"),
    ("How many hours in 3 days?", "small", "math_simple"),
    ("What is 10 to the power of 4?", "small", "math_simple"),
    ("Convert 0 Kelvin to Celsius", "small", "math_simple"),
    ("What is 1024 divided by 32?", "small", "math_simple"),
    ("How many pounds in a ton?", "small", "math_simple"),
    ("What is 13 squared?", "small", "math_simple"),
    ("How many months in 4 years?", "small", "math_simple"),
    ("What is 800 divided by 25?", "small", "math_simple"),
    ("Convert 10 yards to feet", "small", "math_simple"),
    ("What is 1/3 as a decimal?", "small", "math_simple"),
    ("How many millimeters in a centimeter?", "small", "math_simple"),
    ("What is 22 times 5?", "small", "math_simple"),
    ("What is 144 plus 256?", "small", "math_simple"),
    ("Convert 2 miles to meters", "small", "math_simple"),
    ("What is 50% of 88?", "small", "math_simple"),
    ("How many nanoseconds in a millisecond?", "small", "math_simple"),
    ("What is 6 factorial?", "small", "math_simple"),
    ("What is 3.5 times 10?", "small", "math_simple"),
    ("How many seconds in a week?", "small", "math_simple"),
    ("What is the square root of 1024?", "small", "math_simple"),
    ("Convert 5 gallons to liters", "small", "math_simple"),

    # =========================================================
    # ADDITIONAL ANALYSIS — 30 cases (all "big")
    # =========================================================
    ("Explain why naive Bayes performs well despite its independence assumption being violated.", "big", "analysis"),
    ("Analyze the trade-offs between strong consistency and performance in distributed databases.", "big", "analysis"),
    ("Explain how garbage collection pauses affect latency-sensitive applications.", "big", "analysis"),
    ("Analyze the technical and organizational challenges of adopting trunk-based development.", "big", "analysis"),
    ("Explain how SSL pinning works and why it improves mobile app security.", "big", "analysis"),
    ("Analyze the effects of cold starts on serverless function latency.", "big", "analysis"),
    ("Explain why hash collisions are unavoidable and how hash maps handle them.", "big", "analysis"),
    ("Analyze the trade-offs of using git history as your audit log.", "big", "analysis"),
    ("Explain how database connection pooling prevents connection exhaustion.", "big", "analysis"),
    ("Analyze why the Agile manifesto is often misunderstood in practice.", "big", "analysis"),
    ("Explain how the transformer positional encoding generalizes to unseen sequence lengths.", "big", "analysis"),
    ("Analyze the implications of choosing a vendor-specific observability tool versus OpenTelemetry.", "big", "analysis"),
    ("Explain how P99 latency differs from average latency and why it matters for SLAs.", "big", "analysis"),
    ("Analyze the security risks of embedding secrets in Docker images.", "big", "analysis"),
    ("Explain how copy-on-write semantics reduce memory usage in operating systems.", "big", "analysis"),
    ("Analyze the engineering trade-offs of choosing Python versus Go for a latency-sensitive service.", "big", "analysis"),
    ("Explain why HTTPS alone does not protect against all man-in-the-middle attacks.", "big", "analysis"),
    ("Analyze the computational trade-offs of different attention mechanisms in transformers.", "big", "analysis"),
    ("Explain how backpressure mechanisms prevent cascading failures in streaming systems.", "big", "analysis"),
    ("Analyze the impact of network latency on distributed transaction throughput.", "big", "analysis"),
    ("Explain why database indices slow down write operations.", "big", "analysis"),
    ("Analyze the trade-offs of storing configuration in environment variables versus a config service.", "big", "analysis"),
    ("Explain how the kernel's buddy allocator reduces memory fragmentation.", "big", "analysis"),
    ("Analyze the impact of team size on code review effectiveness.", "big", "analysis"),
    ("Explain how probabilistic data structures trade accuracy for space efficiency.", "big", "analysis"),
    ("Analyze the engineering risks of adopting a new programming language in production.", "big", "analysis"),
    ("Explain why consistent hashing reduces key rebalancing when nodes join or leave.", "big", "analysis"),
    ("Analyze the observability trade-offs between logging, metrics, and traces.", "big", "analysis"),
    ("Explain how circuit breakers prevent cascading failures in microservice architectures.", "big", "analysis"),
    ("Analyze the human factors that contribute to production incidents beyond technical causes.", "big", "analysis"),

    # =========================================================
    # ADDITIONAL YES/NO — 20 cases (all "small")
    # =========================================================
    ("Is C++ faster than Python?", "small", "yes_no"),
    ("Is the sun bigger than Earth?", "small", "yes_no"),
    ("Does Java garbage collect automatically?", "small", "yes_no"),
    ("Is REST stateless?", "small", "yes_no"),
    ("Does PostgreSQL support JSON?", "small", "yes_no"),
    ("Is Kubernetes written in Go?", "small", "yes_no"),
    ("Is Apache Kafka a message broker?", "small", "yes_no"),
    ("Does Python support lambda functions?", "small", "yes_no"),
    ("Is TCP reliable?", "small", "yes_no"),
    ("Is binary search O(log n)?", "small", "yes_no"),
    ("Does Linux use a monolithic kernel?", "small", "yes_no"),
    ("Is SHA-256 a symmetric encryption algorithm?", "small", "yes_no"),
    ("Does Rust have a garbage collector?", "small", "yes_no"),
    ("Is a list in Python mutable?", "small", "yes_no"),
    ("Does Git track file permissions?", "small", "yes_no"),
    ("Is JSON a binary format?", "small", "yes_no"),
    ("Is TypeScript compiled to JavaScript?", "small", "yes_no"),
    ("Does React use a virtual DOM?", "small", "yes_no"),
    ("Is an integer 4 bytes in most 64-bit systems?", "small", "yes_no"),
    ("Is quicksort in-place?", "small", "yes_no"),

    # =========================================================
    # ADDITIONAL DEFINITIONS — 20 cases (all "small")
    # =========================================================
    ("What is a promise in JavaScript?", "small", "definitions"),
    ("Define observability in software engineering", "small", "definitions"),
    ("What is a monorepo?", "small", "definitions"),
    ("Define zero-downtime deployment", "small", "definitions"),
    ("What is a webhook?", "small", "definitions"),
    ("Define chaos engineering", "small", "definitions"),
    ("What is a p99 latency?", "small", "definitions"),
    ("Define service mesh", "small", "definitions"),
    ("What is FinOps?", "small", "definitions"),
    ("Define federated learning", "small", "definitions"),
    ("What is a vector database?", "small", "definitions"),
    ("Define prompt engineering", "small", "definitions"),
    ("What is model drift?", "small", "definitions"),
    ("Define data lineage", "small", "definitions"),
    ("What is a hot path in software?", "small", "definitions"),
    ("Define trunk-based development", "small", "definitions"),
    ("What is a fan-out pattern?", "small", "definitions"),
    ("Define the strangler fig pattern", "small", "definitions"),
    ("What is MTTR?", "small", "definitions"),
    ("Define infrastructure as code", "small", "definitions"),

    # =========================================================
    # ADDITIONAL DEBUG SIMPLE — 20 cases (all "small")
    # =========================================================
    ("Why am I getting 'maximum recursion depth exceeded' in Python?", "small", "debug_simple"),
    ("What does 'ERR_CONNECTION_REFUSED' mean in a browser?", "small", "debug_simple"),
    ("Why is my Python dictionary throwing a KeyError?", "small", "debug_simple"),
    ("What does 'git merge conflict' mean?", "small", "debug_simple"),
    ("Why is my database query slow?", "small", "debug_simple"),
    ("What causes a React 'key' warning?", "small", "debug_simple"),
    ("Why does my pip install fail with SSL error?", "small", "debug_simple"),
    ("What causes 'uncaught promise rejection' in Node.js?", "small", "debug_simple"),
    ("Why is Kubernetes pod stuck in 'Pending' state?", "small", "debug_simple"),
    ("What does 'cannot read properties of undefined' mean in JavaScript?", "small", "debug_simple"),
    ("Why is my Python virtual environment not activating?", "small", "debug_simple"),
    ("What causes a 'broken pipe' error in Linux?", "small", "debug_simple"),
    ("Why is Docker build failing with 'no space left on device'?", "small", "debug_simple"),
    ("What does 'schema mismatch' mean in Avro?", "small", "debug_simple"),
    ("Why is my Flask app returning 405 Method Not Allowed?", "small", "debug_simple"),
    ("What causes a 'zombie process' in Linux?", "small", "debug_simple"),
    ("Why is my CSS flexbox not centering the element?", "small", "debug_simple"),
    ("What does 'address already in use' mean when starting a server?", "small", "debug_simple"),
    ("Why is my JavaScript async function not awaiting?", "small", "debug_simple"),
    ("What causes a 'heap dump' in Java?", "small", "debug_simple"),

    # =========================================================
    # ADDITIONAL CONVERSATIONAL — 20 cases (all "small")
    # =========================================================
    ("What's the short version of that?", "small", "conversational"),
    ("Can you rephrase that in simpler terms?", "small", "conversational"),
    ("What was the first point you made?", "small", "conversational"),
    ("OK, and how does that apply to my situation?", "small", "conversational"),
    ("Is there a simpler way to think about this?", "small", "conversational"),
    ("Can you show me a concrete example?", "small", "conversational"),
    ("What did you mean by that last sentence?", "small", "conversational"),
    ("Can you explain that without jargon?", "small", "conversational"),
    ("OK what's the bottom line?", "small", "conversational"),
    ("I'm confused, can you start over?", "small", "conversational"),
    ("Can you break that into numbered steps?", "small", "conversational"),
    ("What's the most important thing to take away?", "small", "conversational"),
    ("Can you give me just the key points?", "small", "conversational"),
    ("What does that mean for a beginner?", "small", "conversational"),
    ("Can you give an analogy for that?", "small", "conversational"),
    ("Is that the standard approach or an alternative?", "small", "conversational"),
    ("How does that compare to what you said earlier?", "small", "conversational"),
    ("Could you expand on the trade-offs?", "small", "conversational"),
    ("What should I do first?", "small", "conversational"),
    ("Can you summarize everything so far?", "small", "conversational"),

    # =========================================================
    # ADDITIONAL SHORT CREATIVE — 24 cases (all "small")
    # =========================================================
    ("Give me a fun name for a code monkey mascot", "small", "short_creative"),
    ("Write a haiku about serverless architecture", "small", "short_creative"),
    ("Make up a pun about load balancing", "small", "short_creative"),
    ("Tell me a riddle about a cache miss", "small", "short_creative"),
    ("Give me a one-liner about software deadlines", "small", "short_creative"),
    ("Write a haiku about latency", "small", "short_creative"),
    ("Give me a fun tagline for a database product", "small", "short_creative"),
    ("Make up a pun about Git branches", "small", "short_creative"),
    ("Tell me a joke about stack overflow (the error)", "small", "short_creative"),
    ("Give me a creative team name for a platform engineering team", "small", "short_creative"),
    ("Write a limerick about DNS propagation", "small", "short_creative"),
    ("Give me a motivational quote about failing tests", "small", "short_creative"),
    ("Make up a funny error message for a 418 I'm a Teapot response", "small", "short_creative"),
    ("Tell me a programming fun fact I probably don't know", "small", "short_creative"),
    ("Write a two-line poem about microservices", "small", "short_creative"),
    ("Give me a pun about Kubernetes pods", "small", "short_creative"),
    ("Tell me a joke about frontend versus backend developers", "small", "short_creative"),
    ("Give me a fun tagline for an open source project", "small", "short_creative"),
    ("Write a haiku about an infinite loop", "small", "short_creative"),
    ("Make up a creative name for a logging library", "small", "short_creative"),
    ("Give me a riddle about a stack data structure", "small", "short_creative"),
    ("Tell me a fun fact about the history of the internet", "small", "short_creative"),
    ("Give me a motivational quote about learning from bugs", "small", "short_creative"),
    ("Make up a silly acronym for 'DEBUG'", "small", "short_creative"),

    # =========================================================
    # ADDITIONAL GREETINGS / SMALL TALK — 10 cases (all "small")
    # =========================================================
    ("G'day!", "small", "greetings"),
    ("Hi again!", "small", "greetings"),
    ("You're awesome, thanks!", "small", "greetings"),
    ("Much appreciated!", "small", "greetings"),
    ("Peace out", "small", "greetings"),
    ("Cheers mate", "small", "greetings"),
    ("Good to see you", "small", "greetings"),

    # =========================================================
    # ADDITIONAL ETHICAL/PHILOSOPHICAL — 10 cases (all "big")
    # =========================================================
    ("Is it ethical to use AI-generated code in production without disclosure?", "big", "ethical_philosophical"),
    ("What are the ethical implications of designing apps to be addictive?", "big", "ethical_philosophical"),
    ("Should employees have the right to know when AI is used to evaluate them?", "big", "ethical_philosophical"),
    ("Is it ethical to train AI on copyrighted works without compensation?", "big", "ethical_philosophical"),
    ("What moral obligations do companies have when they collect user data?", "big", "ethical_philosophical"),
    ("Is digital privacy more important than national security?", "big", "ethical_philosophical"),
    ("What are the ethical implications of facial recognition technology in public spaces?", "big", "ethical_philosophical"),
    ("Should AI companies be required to explain their model's decisions?", "big", "ethical_philosophical"),
    ("Is open-sourcing military AI technology ethical?", "big", "ethical_philosophical"),
    ("What are the social responsibilities of developers who build large-scale platforms?", "big", "ethical_philosophical"),

    # =========================================================
    # ADDITIONAL COMPARISON TECHNICAL — 15 cases (all "big")
    # =========================================================
    ("Compare Nginx and Caddy as reverse proxies for a small project.", "big", "comparison_technical"),
    ("What are the differences between AWS S3 and Google Cloud Storage for object storage?", "big", "comparison_technical"),
    ("Compare HashiCorp Vault and AWS Secrets Manager for secrets management.", "big", "comparison_technical"),
    ("What are the differences between Grafana and Kibana for data visualization?", "big", "comparison_technical"),
    ("Compare Pydantic and Marshmallow for Python data validation.", "big", "comparison_technical"),
    ("What are the differences between Flask-RESTful and FastAPI for building APIs?", "big", "comparison_technical"),
    ("Compare Alembic and Django ORM migrations for database schema management.", "big", "comparison_technical"),
    ("What are the differences between Weights & Biases and MLflow for experiment tracking?", "big", "comparison_technical"),
    ("Compare Locust and k6 for load testing APIs.", "big", "comparison_technical"),
    ("What are the differences between pytest and unittest for Python testing?", "big", "comparison_technical"),
    ("Compare Traefik and HAProxy for layer 4/7 load balancing.", "big", "comparison_technical"),
    ("What are the differences between Parquet and ORC file formats for analytics?", "big", "comparison_technical"),
    ("Compare Consul and Eureka for service discovery in microservices.", "big", "comparison_technical"),
    ("What are the differences between nats.io and ZeroMQ for lightweight messaging?", "big", "comparison_technical"),
    ("Compare Poetry and Pipenv for Python dependency management.", "big", "comparison_technical"),

    # =========================================================
    # ADDITIONAL — 33 cases to reach exactly 1000
    # =========================================================

    # 11 more factual small
    ("What does VPC stand for in AWS?", "small", "factual"),
    ("What is the capital of Denmark?", "small", "factual"),
    ("Who wrote 'The Wealth of Nations'?", "small", "factual"),
    ("What does YAML stand for?", "small", "factual"),
    ("What is the capital of Chile?", "small", "factual"),
    ("Who invented the telescope?", "small", "factual"),
    ("What does SLA stand for?", "small", "factual"),
    ("What is the capital of Romania?", "small", "factual"),
    ("What does RAID stand for in storage?", "small", "factual"),
    ("Who created the Linux kernel?", "small", "factual"),
    ("What is the chemical symbol for nitrogen?", "small", "factual"),

    # 11 more analysis big
    ("Explain how shard key selection affects performance in MongoDB.", "big", "analysis"),
    ("Analyze the security implications of using environment variables for secrets.", "big", "analysis"),
    ("Explain why tail latency matters more than average latency in user-facing systems.", "big", "analysis"),
    ("Analyze the trade-offs of using stored procedures versus application-side logic.", "big", "analysis"),
    ("Explain how speculative execution attacks like Spectre exploit CPU branch prediction.", "big", "analysis"),
    ("Analyze the engineering impact of adopting a new LTS dependency in a large codebase.", "big", "analysis"),
    ("Explain how consistent hashing minimizes key remapping during node failures.", "big", "analysis"),
    ("Analyze why eventual consistency causes conflicts and how CRDTs resolve them.", "big", "analysis"),
    ("Explain how the JVM JIT compiler optimizes hot code paths at runtime.", "big", "analysis"),
    ("Analyze the performance implications of N+1 queries in ORM-based applications.", "big", "analysis"),
    ("Explain how coroutines differ from threads and processes at the OS level.", "big", "analysis"),

    # 11 more career/learning big
    ("What are the best habits for maintaining a healthy work-life balance as a developer?", "big", "career_learning"),
    ("How do I develop a habit of reading research papers as a practitioner?", "big", "career_learning"),
    ("What are the most effective ways to mentorship junior engineers?", "big", "career_learning"),
    ("How do I position myself for a principal engineer role at a large company?", "big", "career_learning"),
    ("What are the most important lessons from the first 5 years of a software engineering career?", "big", "career_learning"),
    ("How do I build technical credibility with non-technical stakeholders?", "big", "career_learning"),
    ("What is the difference between a tech lead and an engineering manager?", "big", "career_learning"),
    ("How do I build a learning system that keeps me growing continuously as a developer?", "big", "career_learning"),
    ("What should I look for when evaluating a startup's technical health before joining?", "big", "career_learning"),
    ("How do I approach salary negotiation for a remote role at an international company?", "big", "career_learning"),
    ("What are the signs of a toxic engineering culture to watch for in interviews?", "big", "career_learning"),
]

# Verify count
assert len(TEST_CASES) == 1000, f"Expected 1000 test cases, got {len(TEST_CASES)}"


# ---------------------------------------------------------------------------
# Hardcoded eval_500 baseline results (from the 500-case evaluation)
# Used to build the Before/After comparison table.
# ---------------------------------------------------------------------------
EVAL_500_BASELINE: dict[str, dict] = {
    "_overall": {"accuracy": 0.882, "cascade_rate": 0.308},
    "ambiguous":       {"accuracy": 0.367, "cascade_rate": 0.467, "n": 30},
    "short_creative":  {"accuracy": 0.900, "cascade_rate": 1.000, "n": 20},
    "career_learning": {"accuracy": 0.950, "cascade_rate": 0.800, "n": 20},
    "research_synthesis": {"accuracy": 0.850, "cascade_rate": 0.600, "n": 20},
    "greetings":       {"accuracy": 1.000, "cascade_rate": 0.000, "n": 20},
    "factual":         {"accuracy": 0.917, "cascade_rate": 0.033, "n": 60},
    "math_simple":     {"accuracy": 1.000, "cascade_rate": 0.000, "n": 30},
    "yes_no":          {"accuracy": 1.000, "cascade_rate": 0.050, "n": 20},
    "definitions":     {"accuracy": 0.900, "cascade_rate": 0.133, "n": 30},
    "trivia":          {"accuracy": 0.850, "cascade_rate": 0.100, "n": 20},
    "conversions":     {"accuracy": 1.000, "cascade_rate": 0.000, "n": 20},
    "proofs":          {"accuracy": 0.950, "cascade_rate": 0.050, "n": 20},
    "system_design":   {"accuracy": 0.967, "cascade_rate": 0.067, "n": 30},
    "multi_step_code": {"accuracy": 0.933, "cascade_rate": 0.033, "n": 30},
    "analysis":        {"accuracy": 0.933, "cascade_rate": 0.067, "n": 30},
    "compound_questions": {"accuracy": 0.967, "cascade_rate": 0.033, "n": 30},
    "real_world_planning": {"accuracy": 0.950, "cascade_rate": 0.050, "n": 20},
    "multi_constraint":{"accuracy": 0.950, "cascade_rate": 0.050, "n": 20},
    "domain_specific_deep": {"accuracy": 0.933, "cascade_rate": 0.033, "n": 30},
}

# ---------------------------------------------------------------------------
# Evaluation dataclass
# ---------------------------------------------------------------------------

CASCADE_THRESHOLD = 0.65  # matches config.ML_ROUTER_CONFIDENCE_THRESHOLD


@dataclass
class QueryResult:
    query: str
    ground_truth: str
    category: str
    ml_decision: str
    ml_confidence: float
    correct: bool
    would_cascade: bool


# ---------------------------------------------------------------------------
# Main evaluation logic
# ---------------------------------------------------------------------------

def run_evaluation(model_path: str) -> list[QueryResult]:
    """Load the ML router and evaluate all 1000 test cases."""
    from router.ml_router import MLRouter

    print(f"\nLoading ML Router from: {model_path}")
    router = MLRouter.load(model_path)
    print("Model loaded successfully.\n")

    results: list[QueryResult] = []
    for query, ground_truth, category in TEST_CASES:
        pred = router.predict(query)
        correct = pred.decision == ground_truth
        would_cascade = pred.confidence < CASCADE_THRESHOLD

        results.append(QueryResult(
            query=query,
            ground_truth=ground_truth,
            category=category,
            ml_decision=pred.decision,
            ml_confidence=pred.confidence,
            correct=correct,
            would_cascade=would_cascade,
        ))

    return results


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _header(title: str, width: int = 76) -> str:
    line = "=" * width
    return f"\n{line}\n  {title}\n{line}"


def _subheader(title: str, width: int = 76) -> str:
    return f"\n{'─' * width}\n  {title}\n{'─' * width}"


def _delta(new_val: float, old_val: Optional[float], pct: bool = True) -> str:
    """Format a delta string with + / - sign."""
    if old_val is None:
        return "  (new)"
    diff = new_val - old_val
    sign = "+" if diff >= 0 else ""
    if pct:
        return f"  ({sign}{diff:.1%})"
    return f"  ({sign}{diff:.3f})"


def print_report(results: list[QueryResult]) -> None:  # noqa: C901
    total = len(results)
    correct_all = sum(1 for r in results if r.correct)
    overall_accuracy = correct_all / total

    # Confusion matrix
    tp = sum(1 for r in results if r.ground_truth == "big" and r.ml_decision == "big")
    fp = sum(1 for r in results if r.ground_truth == "small" and r.ml_decision == "big")
    tn = sum(1 for r in results if r.ground_truth == "small" and r.ml_decision == "small")
    fn = sum(1 for r in results if r.ground_truth == "big" and r.ml_decision == "small")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Confidence distribution
    confidences = [r.ml_confidence for r in results]
    conf_mean   = statistics.mean(confidences)
    conf_median = statistics.median(confidences)
    conf_min    = min(confidences)
    conf_max    = max(confidences)
    above_thresh = sum(1 for c in confidences if c >= CASCADE_THRESHOLD)

    # Cascade stats
    cascade_count = sum(1 for r in results if r.would_cascade)
    cascade_rate  = cascade_count / total

    # Per-category breakdown
    categories: dict[str, list[QueryResult]] = {}
    for r in results:
        categories.setdefault(r.category, []).append(r)

    # ── Print report ─────────────────────────────────────────────────────────

    print(_header("ML ROUTER EVALUATION REPORT — 1000 LABELED TEST CASES"))
    print(f"\n  Total test cases : {total}")
    print(f"  Cascade threshold: {CASCADE_THRESHOLD}")
    print(f"  New categories   : conversational, debug_simple, ethical_philosophical, comparison_technical")

    # ── 1. Overall accuracy ──────────────────────────────────────────────────
    print(_subheader("1. OVERALL ML ROUTER ACCURACY"))
    baseline_acc = EVAL_500_BASELINE["_overall"]["accuracy"]
    print(f"  Correct predictions : {correct_all} / {total}")
    print(f"  Accuracy (1000-case): {overall_accuracy:.1%}")
    print(f"  Accuracy (500-case baseline): {baseline_acc:.1%}  →  delta: {_delta(overall_accuracy, baseline_acc).strip()}")

    # ── 2. Confusion matrix ──────────────────────────────────────────────────
    print(_subheader("2. CONFUSION MATRIX  (positive class = 'big')"))
    print(f"  {'':30s} {'Predicted BIG':>14} {'Predicted SMALL':>15}")
    print(f"  {'Actual BIG':30s} {'TP =':>8} {tp:>5}   {'FN =':>8} {fn:>5}")
    print(f"  {'Actual SMALL':30s} {'FP =':>8} {fp:>5}   {'TN =':>8} {tn:>5}")
    print()
    print(f"  Precision (big class) : {precision:.3f}")
    print(f"  Recall    (big class) : {recall:.3f}")
    print(f"  F1 score  (big class) : {f1:.3f}")

    # ── 3. Confidence distribution ───────────────────────────────────────────
    print(_subheader("3. CONFIDENCE DISTRIBUTION"))
    print(f"  Mean   : {conf_mean:.3f}")
    print(f"  Median : {conf_median:.3f}")
    print(f"  Min    : {conf_min:.3f}")
    print(f"  Max    : {conf_max:.3f}")
    print(f"  Above cascade threshold ({CASCADE_THRESHOLD}): {above_thresh} / {total}  "
          f"({above_thresh/total:.1%})")

    # Histogram
    print()
    print("  Confidence histogram:")
    bucket_width = 0.10
    for i in range(10):
        lo = i * bucket_width
        hi = lo + bucket_width
        if i == 9:
            count = sum(1 for c in confidences if lo <= c <= 1.0)
        else:
            count = sum(1 for c in confidences if lo <= c < hi)
        bar = "#" * min(count, 60)
        print(f"    [{lo:.1f} – {hi:.1f}) : {count:>4}  {bar}")

    # ── 4. Cascade rate ───────────────────────────────────────────────────────
    print(_subheader("4. CASCADE RATE  (ML confidence < threshold → LLM fallback)"))
    baseline_cascade = EVAL_500_BASELINE["_overall"]["cascade_rate"]
    print(f"  Would cascade       : {cascade_count} / {total}  ({cascade_rate:.1%})")
    print(f"  Handled by ML alone : {total - cascade_count} / {total}  ({(total-cascade_count)/total:.1%})")
    print(f"  500-case baseline cascade rate: {baseline_cascade:.1%}  →  delta: {_delta(cascade_rate, baseline_cascade).strip()}")

    # ── 5. Per-category accuracy & cascade rate ───────────────────────────────
    print(_subheader("5. PER-CATEGORY BREAKDOWN"))
    cat_names = sorted(categories.keys())
    col1 = max(len(n) for n in cat_names) + 2
    header_line = (
        f"  {'Category':<{col1}} {'N':>4}  {'Acc':>7}  {'Correct':>7}  "
        f"{'CascadeRate':>11}  {'Cascades':>8}  {'Label'}"
    )
    print(header_line)
    print("  " + "-" * (len(header_line) - 2))

    for cat in cat_names:
        cat_results = categories[cat]
        n = len(cat_results)
        n_correct = sum(1 for r in cat_results if r.correct)
        n_cascade = sum(1 for r in cat_results if r.would_cascade)
        acc = n_correct / n
        casc_rate = n_cascade / n
        label = "(NEW)" if cat in {"conversational", "debug_simple", "ethical_philosophical", "comparison_technical"} else ""
        print(
            f"  {cat:<{col1}} {n:>4}  {acc:>7.1%}  {n_correct:>7}  "
            f"{casc_rate:>11.1%}  {n_cascade:>8}  {label}"
        )

    # ── 6. BEFORE / AFTER COMPARISON TABLE ───────────────────────────────────
    print(_subheader("6. BEFORE / AFTER IMPROVEMENT TABLE  (500-case baseline vs 1000-case)"))
    shared_cats = [c for c in cat_names if c in EVAL_500_BASELINE]
    col1b = max(len(c) for c in shared_cats) + 2

    hdr = (f"  {'Category':<{col1b}} "
           f"{'Acc-500':>8} {'Acc-1000':>9} {'AccDelta':>9}  "
           f"{'Casc-500':>9} {'Casc-1000':>10} {'CascDelta':>10}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for cat in shared_cats:
        cat_results = categories[cat]
        n = len(cat_results)
        n_correct = sum(1 for r in cat_results if r.correct)
        n_cascade = sum(1 for r in cat_results if r.would_cascade)
        acc_1000 = n_correct / n
        casc_1000 = n_cascade / n

        b = EVAL_500_BASELINE[cat]
        acc_500 = b["accuracy"]
        casc_500 = b["cascade_rate"]

        acc_delta = acc_1000 - acc_500
        casc_delta = casc_1000 - casc_500

        acc_d_str = f"{'+' if acc_delta >= 0 else ''}{acc_delta:.1%}"
        casc_d_str = f"{'+' if casc_delta >= 0 else ''}{casc_delta:.1%}"

        print(
            f"  {cat:<{col1b}} "
            f"{acc_500:>8.1%} {acc_1000:>9.1%} {acc_d_str:>9}  "
            f"{casc_500:>9.1%} {casc_1000:>10.1%} {casc_d_str:>10}"
        )

    # Overall row
    oa_delta = overall_accuracy - EVAL_500_BASELINE["_overall"]["accuracy"]
    casc_d_oa = cascade_rate - EVAL_500_BASELINE["_overall"]["cascade_rate"]
    oa_d_str = f"{'+' if oa_delta >= 0 else ''}{oa_delta:.1%}"
    casc_d_oa_str = f"{'+' if casc_d_oa >= 0 else ''}{casc_d_oa:.1%}"
    print("  " + "-" * (len(hdr) - 2))
    print(
        f"  {'OVERALL':<{col1b}} "
        f"{EVAL_500_BASELINE['_overall']['accuracy']:>8.1%} {overall_accuracy:>9.1%} {oa_d_str:>9}  "
        f"{EVAL_500_BASELINE['_overall']['cascade_rate']:>9.1%} {cascade_rate:>10.1%} {casc_d_oa_str:>10}"
    )

    # ── 7. Top 20 wrong predictions ───────────────────────────────────────────
    print(_subheader("7. TOP 20 WRONG PREDICTIONS  (highest ML confidence)"))
    wrong = [r for r in results if not r.correct]
    wrong_sorted = sorted(wrong, key=lambda r: r.ml_confidence, reverse=True)[:20]

    if not wrong_sorted:
        print("  No wrong predictions! Perfect accuracy.")
    else:
        for i, r in enumerate(wrong_sorted, 1):
            q_display = r.query[:70] + ("..." if len(r.query) > 70 else "")
            print(
                f"  {i:>2}. [{r.category}] conf={r.ml_confidence:.3f} "
                f"gt={r.ground_truth} pred={r.ml_decision}\n"
                f"      \"{q_display}\""
            )

    # ── 8. Top 20 most uncertain predictions ──────────────────────────────────
    print(_subheader("8. TOP 20 MOST UNCERTAIN PREDICTIONS  (lowest confidence)"))
    uncertain = sorted(results, key=lambda r: r.ml_confidence)[:20]

    for i, r in enumerate(uncertain, 1):
        status = "CORRECT" if r.correct else "WRONG"
        q_display = r.query[:70] + ("..." if len(r.query) > 70 else "")
        print(
            f"  {i:>2}. [{r.category}] conf={r.ml_confidence:.3f} "
            f"gt={r.ground_truth} pred={r.ml_decision}  [{status}]\n"
            f"      \"{q_display}\""
        )

    # ── 9. New-category summary ────────────────────────────────────────────────
    print(_subheader("9. NEW CATEGORY RESULTS  (not present in eval_500.py)"))
    new_cats = ["conversational", "debug_simple", "ethical_philosophical", "comparison_technical"]
    for cat in new_cats:
        if cat not in categories:
            print(f"  {cat}: (no cases)")
            continue
        cr = categories[cat]
        n = len(cr)
        acc = sum(1 for r in cr if r.correct) / n
        casc = sum(1 for r in cr if r.would_cascade) / n
        print(f"  {cat:<30} N={n:>4}  Accuracy={acc:.1%}  CascadeRate={casc:.1%}")

    # ── 10. Summary recommendation ─────────────────────────────────────────────
    print(_subheader("10. SUMMARY & RECOMMENDATION"))

    issues: list[str] = []
    positives: list[str] = []

    if overall_accuracy >= 0.90:
        positives.append(f"Strong overall accuracy ({overall_accuracy:.1%}).")
    elif overall_accuracy >= 0.80:
        positives.append(f"Acceptable overall accuracy ({overall_accuracy:.1%}); room for improvement.")
    else:
        issues.append(f"Low overall accuracy ({overall_accuracy:.1%}) — retrain with more data.")

    if cascade_rate <= 0.15:
        positives.append(f"Low cascade rate ({cascade_rate:.1%}) — ML router handles most queries confidently.")
    elif cascade_rate <= 0.30:
        issues.append(
            f"Moderate cascade rate ({cascade_rate:.1%}) — "
            f"roughly {cascade_count} queries would fall back to the LLM classifier."
        )
    else:
        issues.append(
            f"High cascade rate ({cascade_rate:.1%}) — {cascade_count} queries require LLM fallback."
        )

    if precision < 0.85:
        issues.append(
            f"Precision for 'big' class is {precision:.1%} — too many small queries misrouted to large LLM."
        )
    if recall < 0.85:
        issues.append(
            f"Recall for 'big' class is {recall:.1%} — too many complex queries handled by small LLM."
        )

    cat_accs = {
        cat: sum(1 for r in cr if r.correct) / len(cr)
        for cat, cr in categories.items()
    }
    worst_cats = sorted(cat_accs, key=cat_accs.get)[:3]  # type: ignore[arg-type]
    bad_str = ", ".join(
        f"{c} ({cat_accs[c]:.0%})" for c in worst_cats if cat_accs[c] < 0.80
    )
    if bad_str:
        issues.append(f"Weakest categories: {bad_str}. Add more seed examples for these.")

    # Compare to baseline
    if overall_accuracy > EVAL_500_BASELINE["_overall"]["accuracy"]:
        positives.append(
            f"Accuracy improved over 500-case baseline "
            f"({EVAL_500_BASELINE['_overall']['accuracy']:.1%} → {overall_accuracy:.1%})."
        )
    if cascade_rate < EVAL_500_BASELINE["_overall"]["cascade_rate"]:
        positives.append(
            f"Cascade rate improved over 500-case baseline "
            f"({EVAL_500_BASELINE['_overall']['cascade_rate']:.1%} → {cascade_rate:.1%})."
        )

    print()
    if positives:
        print("  Strengths:")
        for p in positives:
            print(f"    + {p}")
    if issues:
        print("\n  Issues / Action items:")
        for item in issues:
            print(f"    ! {item}")

    print("\n  Recommended next steps:")
    if overall_accuracy < 0.90:
        print("    1. Add 50-100 more labeled examples for weak categories to router/seed_data.py")
        print("    2. Retrain with: python -m router.train")
    if cascade_rate > 0.20:
        print("    3. Consider tuning ML_ROUTER_CONFIDENCE_THRESHOLD (currently 0.65).")
    print("    4. Collect real production queries via ClassificationLogger and retrain periodically.")

    print("\n" + "=" * 76 + "\n")


# ---------------------------------------------------------------------------
# Save results to JSON
# ---------------------------------------------------------------------------

def save_results(results: list[QueryResult], out_path: str) -> None:
    categories: dict[str, list[QueryResult]] = {}
    for r in results:
        categories.setdefault(r.category, []).append(r)

    total = len(results)
    correct_all = sum(1 for r in results if r.correct)
    cascade_count = sum(1 for r in results if r.would_cascade)

    per_cat: dict = {}
    for cat, cr in categories.items():
        n = len(cr)
        per_cat[cat] = {
            "n": n,
            "accuracy": sum(1 for r in cr if r.correct) / n,
            "cascade_rate": sum(1 for r in cr if r.would_cascade) / n,
        }

    output = {
        "total_cases": total,
        "overall_accuracy": correct_all / total,
        "overall_cascade_rate": cascade_count / total,
        "cascade_threshold": CASCADE_THRESHOLD,
        "per_category": per_cat,
        "baseline_500": EVAL_500_BASELINE,
        "results": [asdict(r) for r in results],
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to: {out_path}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    model_path = os.path.join(project_root, "router", "models", "router_v0.joblib")
    out_path = os.path.join(project_root, "router", "eval_results_1000.json")

    if len(sys.argv) > 1:
        model_path = sys.argv[1]

    if not os.path.exists(model_path):
        print(f"\nERROR: Model not found at {model_path}")
        print("Train the model first with:  python -m router.train")
        sys.exit(1)

    results = run_evaluation(model_path)
    print_report(results)
    save_results(results, out_path)


if __name__ == "__main__":
    main()
