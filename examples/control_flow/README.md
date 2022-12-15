# Control flow

When tracing some model, control flow is tricky as we can't trace it.

For condition statement, a workaround would be trace each branch and then merge them into a single graph. The solution need users to manually create `If` node.

        cond_input
            |
            If
        |--------|
        |        |
    subgraph1   subgraph2
        |        |
        |--------|
            |
            |
       Unified outputs


Hints:

- Two branches need to have the same outputs.
- Inputs of parent graph should be the union of inputs of subgraphs (which means during inference, you might need to create dummy inputs even if the branch you actually compute does't take some inputs).
- No need to give inputs when making subgraphs (just give all inputs to the parent graph).
- Subgraphs have access to initializers in the parent graph, so if your subgraphs have deplicated initializers. To reduce memory usage during inference, you can deduplicate them and put all unique initializers on the top level(watch out initializer name collision during deduplication).