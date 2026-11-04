
"""

Stage 1 — Establish your baseline
Use your existing SimpleContextBuilder unchanged.

For each evaluation query, save:
- The transcript.
- Retrieved document IDs and similarity scores.
- The full context passed to the generator.
- The generated answer.
- Context token count.
- Generation latency.
- Answer evaluation result.
This gives you something to compare your experiments against.


Stage 2 — Implement query-focused extraction
Instead of passing entire documents into the generator, split each retrieved document into sentences.

For example:

Question: What are the main causes of overfitting in machine learning?

Retrieved document:
- Sentence 1: Machine learning models learn patterns from training data.
- Sentence 2: Overfitting occurs when a model learns noise and details specific to its training data.
- Sentence 3: Training datasets are commonly divided into training and testing subsets.
- Sentence 4: A model with too many parameters relative to the available data may overfit.
Your context builder would identify sentences 2 and 4 as particularly relevant.

This is a useful first improvement because it reduces irrelevant context without asking another language model to summarise the evidence.

You could rank sentences using:
- Embedding similarity to the query.
- TF-IDF or BM25 relevance.
- A combination of lexical and semantic relevance.


Stage 3 — Reduce redundant evidence
Imagine three retrieved documents all explain the same definition of overfitting.

You don't necessarily need all three explanations in your prompt.

You could:
1. Compare sentences across retrieved documents.
2. Identify exact duplicates or highly similar sentences.
3. Keep the strongest or most informative version.
4. Preserve the source document ID for each retained sentence.
An interesting technique here is Maximal Marginal Relevance (MMR). It tries to select evidence that is both relevant to the query and different from evidence already selected.

Its basic formula is:
\[
MMR(d)=\lambda Sim(q,d)
-(1-\lambda)\max_{s\in S}Sim(d,s)
\]

Where:
- \(q\) is the query.
- \(d\) is a candidate sentence or passage.
- \(S\) is the evidence already selected.
- \(\lambda\) controls relevance versus diversity.
This is a particularly good candidate for your project because it is more sophisticated than simply removing duplicate text.


Stage 4 — Combine the strategies
Once the individual approaches work, combine them:
1. Retrieve the top-N documents.
2. Split documents into sentences.
3. Score sentence relevance.
4. Select relevant, diverse sentences using MMR.
5. Stop when you reach a defined context-token budget.
6. Add source labels and construct the final prompt.
Keep the original full-document builder as your control condition.


"""