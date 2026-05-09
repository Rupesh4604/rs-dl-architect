# Contributing to rs-dl-architect

Thank you for your interest in contributing to rs-dl-architect! This skill serves the remote sensing and geospatial AI community, and contributions help make it more valuable for everyone.

## How to Contribute

### 1. Reporting Issues

Found a bug or have a suggestion? Open an issue:

- **Bug reports:** Include the query you used, what you expected, and what happened
- **Feature requests:** Describe the use case and why it would be valuable
- **Inaccuracies:** Point to the specific file and line, suggest a correction

### 2. Adding Content

The skill is organized into modular references. Contributions typically fall into:

#### Adding a New Reference

If you want to add coverage for a new subdomain (e.g., `super-resolution.md`, `3d-reconstruction.md`):

1. Create the reference file in `rs-dl-architect/references/`
2. Follow the existing pattern:
   - Start with task taxonomy (what variations exist)
   - Cover architecture families (when to use each)
   - Include design patterns specific to RS
   - Add common review findings
   - List relevant datasets
3. Add a routing entry in `SKILL.md` (the main file)
4. Test with 2-3 realistic queries

#### Extending an Existing Reference

- Add new architectures to the relevant section
- Update datasets with newer benchmarks
- Add code patterns that are repeatedly useful
- Flag new common failure modes

#### Code Templates

Located in `references/code-templates.md`:

- Must include shape annotations on every tensor operation
- Support both PyTorch and TensorFlow where feasible
- Should be copy-pasteable and runnable
- Follow the minimal-but-complete philosophy

### 3. Updating for New Models

Remote sensing foundation models evolve quickly. When a new model is released:

1. Add it to `references/foundation-models.md` with:
   - Pretraining dataset and modality
   - When to use it vs existing models
   - Adaptation pattern (if novel)
   - Link to paper and code
2. Update `SKILL.md` if it changes the default recommendation pattern

### 4. Improving Reviews

The review checklist (`references/review-checklist.md`) is never complete. Add items that catch real issues you've seen in papers or code.

## Style Guide

### Writing Style

- **Imperative, direct:** "Use X when Y" not "You could consider using X when Y"
- **Specific, not vague:** "Replace BatchNorm with GroupNorm (32 groups)" not "Consider different normalization"
- **Honest about tradeoffs:** State when something doesn't work, not just when it does
- **Assume technical audience:** MS/PhD level, no need to explain what a convolution is

### Structure

Each reference file should have:

```markdown
# Title

Brief intro (1-2 sentences on what this covers).

## Task taxonomy / Decision tree

What variations exist, when to pick each.

## Architecture families

Brief description of each family + when to use.

## Design patterns specific to RS

The things that are different from natural images.

## Common review findings

What goes wrong in practice.

## Datasets (optional)

Standard benchmarks.
```

### Code Style

- Shape annotations: `# (B, C, H, W)` on every intermediate tensor
- Minimal but complete: runnable, but stripped of non-essential details
- Comments explain **why**, not **what**

## Testing Contributions

Before submitting:

1. **Read the skill** — does your addition fit the structure?
2. **Test with realistic queries** — write 2-3 queries a user would actually ask, verify the skill routes correctly and produces useful output
3. **Check for duplicates** — is this already covered elsewhere?

You don't need to run formal evals (those are for major changes), but basic sanity testing helps.

## Pull Request Process

1. Fork the repository
2. Create a branch: `git checkout -b feature/add-super-resolution`
3. Make your changes
4. Update `README.md` if adding a new subdomain
5. Commit with clear messages: `Add super-resolution reference with RCAN, EDSR patterns`
6. Push and open a PR
7. Respond to review feedback

**PR Title Format:**

- `Add: [new content]`
- `Fix: [correction]`
- `Update: [modification]`

## Code of Conduct

- Be respectful and constructive
- Focus on making the skill more useful
- Assume good intent
- Cite sources when adding architectures or claims

## Questions?

Open an issue or reach out to the maintainer. We're happy to help!

---

**Thank you for contributing to the remote sensing ML community!**
