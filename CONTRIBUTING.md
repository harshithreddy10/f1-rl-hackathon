# Contributing to F1 Pit Stop RL

Thank you for your interest in contributing! 🏎️

---

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/f1-rl-hackathon.git
   ```
3. Create a new branch:
   ```bash
   git checkout -b feat/your-feature-name
   ```
4. Make your changes
5. Push and open a Pull Request

---

## Branch Naming Convention

| Type | Format | Example |
|------|--------|---------|
| Feature | `feat/description` | `feat/safety-car-events` |
| Bug fix | `fix/description` | `fix/tyre-cliff-calculation` |
| Docs | `docs/description` | `docs/update-readme` |
| Refactor | `refactor/description` | `refactor/env-rewards` |

---

## Commit Message Format

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add safety car simulation
fix: correct tyre cliff penalty calculation
docs: update installation guide
refactor: simplify reward function
test: add env step tests
```

---

## Code Style

- Use **Ruff** for linting and formatting
- Use **Mypy** for type checking
- All functions must have docstrings
- Max line length: 100 characters

```bash
# Run linter
ruff check .

# Run formatter
ruff format .

# Run type checker
mypy .
```

---

## Pull Request Guidelines

- Keep PRs focused — one feature/fix per PR
- Add tests for new features
- Update documentation if needed
- Ensure all checks pass before requesting review

---

## Reporting Issues

Use GitHub Issues with these labels:
- `bug` — Something isn't working
- `enhancement` — New feature request
- `documentation` — Docs improvement
- `question` — General questions

---

## Code of Conduct

Please read our [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.
