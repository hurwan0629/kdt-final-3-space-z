
# Code Architecture Guidelines

## Goal

Write code that is easy to understand, maintain, and modify.

Optimize for reduced cognitive load, clear responsibilities,
and localized changes rather than fewer lines or files.

## 1. Deep Modules & Information Hiding

- Prefer simple interfaces that encapsulate meaningful complexity.
- Expose what a module does, not how it works internally.
- Hide implementation details that callers do not need.
- Avoid leaking internal state or implementation decisions.

## 2. High Cohesion & Low Coupling

- Group closely related responsibilities within a module.
- Minimize dependencies between unrelated modules.
- Communicate through explicit, stable interfaces.
- Do not split modules solely to reduce file or function length.
- Avoid modules that coordinate unrelated responsibilities.

## 3. Meaningful Abstraction

- Introduce abstractions only when they simplify understanding,
  isolate complexity, or support a concrete requirement.
- Avoid trivial wrappers, pass-through functions,
  and unnecessary delegation layers.
- Every abstraction should provide meaningful functionality.
- Prefer straightforward implementations over speculative
  extensibility or premature generalization.

## 4. Cognitive Load & Readability

- Minimize the concepts and context required to understand code.
- Keep related logic discoverable without excessive navigation.
- Use names that clearly express intent and responsibility.
- Make important behavior visible at the appropriate level.
- Avoid unnecessary indirection, hidden side effects,
  and scattered business logic.

## 5. Explicit Data & Control Flow

- Make execution entry points and major processing stages clear.
- Keep data creation, transformation, and ownership traceable.
- Prefer explicit dependencies over implicit shared state.
- Keep side effects identifiable and predictable.
- Avoid surprising control flow.

## 6. Change Locality

- Design modules so a single requirement change affects
  as few unrelated components as practical.
- Keep related business rules together.
- Avoid duplicating knowledge or business rules across modules.
- Maintain stable boundaries between independent concerns.

## 7. Design & Implementation Decisions

- Do not introduce major architectural changes without approval.
- Explain significant design trade-offs before implementation.
- Delegate routine implementation details without unnecessary approval.
- Prefer the smallest coherent change that satisfies the requirement.
- Preserve existing conventions unless there is a concrete
  reason to change them.

## 8. Code Review & Verification

Before completing a change, verify:

- Can the purpose be understood from the public interface?
- Does each module have a coherent responsibility?
- Is unnecessary complexity hidden from callers?
- Are abstraction layers meaningful?
- Can the execution and data flow be traced?
- Is the change localized without unnecessary dependencies?
- Were existing abstractions reused appropriately?
- Are relevant tests and existing behavior preserved?

Do not refactor unrelated code without a clear reason.

## Working Principle

A good codebase allows developers to understand and modify
the relevant part of the system without having to understand
the entire system.

Prefer clarity of intent and well-defined boundaries
over superficial simplicity.


## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

When the user types `/graphify`, use the installed graphify skill or instructions before doing anything else.

Rules:
- When graphify-out/graph.json exists, run `graphify query "<question>"` before your first grep, search or source-file read for a codebase question, and put this rule in every subagent prompt that explores code. Read raw files only for lines the graph or an earlier read already located. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- Dirty graphify-out/ files are expected after hooks or incremental updates; dirty graph files are not a reason to skip graphify. Only skip graphify if the task is about stale or incorrect graph output, or the user explicitly says not to use it.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
- A committed graphify-out/graph.json is rewritten whole on every build, so a large diff to it in a pull request is expected. Do not flag it as churn or suggest skipping the graph rebuild or its hooks.
