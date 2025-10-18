---
name: Bug Report
about: Create a report to help us improve
title: '[BUG] '
labels: bug
assignees: ''
---

## Describe the Bug
A clear and concise description of what the bug is.

## To Reproduce
Steps to reproduce the behavior:
1. Setup code '...'
2. Run command '....'
3. Observe error '....'

## Expected Behavior
A clear description of what you expected to happen.

## Actual Behavior
What actually happened.

## Code Example
```scala
// Minimal code example to reproduce the issue
val data = sc.parallelize(...)
val model = KMeans.train(...)
```

## Environment
- **Scala Version**: [e.g., 2.13.14]
- **Spark Version**: [e.g., 3.5.1]
- **Library Version**: [e.g., 0.6.0]
- **Java Version**: [e.g., 17]
- **OS**: [e.g., macOS 13, Ubuntu 22.04]

## Stack Trace
```
If applicable, paste the full stack trace here
```

## Additional Context
Add any other context about the problem here (logs, screenshots, etc.).

## Checklist
- [ ] I have searched existing issues to ensure this is not a duplicate
- [ ] I have provided a minimal code example to reproduce the issue
- [ ] I have included environment details
