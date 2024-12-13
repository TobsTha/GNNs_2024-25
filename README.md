# GNNs_2024-25
This repository is intended to efficiently combine our efforts to solve tasks of Ullrich Köthe's lecture "GNNs for the sciences".

## Workflow
`main` branch is the protected branch that should always be deployable (code runs without errors, current version we would hand in if asked to do so). New code is always implemented on another branch. 
Often these are called feature branches as they are created to implement a specific feature. In our case it might be enough to have the 3 additional branches `tobi`, `fabian` and `anton` that we use respectively for implementation. If new code runs without errors on the feature branch (and if it is sufficiently commented) the corresponding branch can be merged with the `main` branch. For that don't forget to pull the main branch into the implementation branch before merging for preventing merging conflicts.

### .gitignore file

## Code Style
### Enforced by [ruff](https://github.com/astral-sh/ruff)
A consistent and common code style is important to make the code more readable and to have meaningful diffs in merge requests where content is changed and not the
personal . With PEP-8 Python sets a basic style guide. To apply several parts of code style black, isort, flake8 are combined in a single tool called ruff.

## Good Practices
-  regularily pull the `main` branch into the branch you are currently working on to detect conflicts early on
-  satisfy ruff: an efficient tool for docstrings is the extension "autoDocstring - Python Docstring Generator" (from Nils Werner).
