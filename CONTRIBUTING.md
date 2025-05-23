# Contributing to Neural Noise Segmentation

Thank you for considering contributing to this project! Here are some guidelines to help you get started.

## Code of Conduct

By participating in this project, you are expected to uphold our Code of Conduct: be respectful, considerate, and collaborative.

## How Can I Contribute?

### Reporting Bugs

- Check if the bug has already been reported in the Issues section
- Use the bug report template to create a new issue
- Include as much relevant information as possible, including:
  - Steps to reproduce
  - Expected behavior
  - Actual behavior
  - Screenshots if applicable
  - Environment details (OS, Python version, etc.)

### Suggesting Enhancements

- Check if the enhancement has already been suggested
- Use the feature request template
- Describe the feature in detail and why it would be valuable

### Pull Requests

1. Fork the repository
2. Create a new branch: `git checkout -b feature/your-feature-name`
3. Make your changes
4. Run tests: `python -m pytest`
5. Commit your changes: `git commit -m 'Add some feature'`
6. Push to your branch: `git push origin feature/your-feature-name`
7. Submit a pull request

## Development Setup

```powershell
# Clone the repository
git clone https://github.com/yourusername/neural-noise-segmentation.git
cd neural-noise-segmentation

# Create a virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

## Coding Guidelines

- Follow PEP 8 style guide
- Write docstrings for all functions and classes
- Include type hints where possible
- Write unit tests for new features
- Keep functions small and focused on a single task

## Git Workflow

- Keep commits small and focused
- Write clear commit messages in the imperative tense
- Rebase your branch before submitting a pull request

Thank you for contributing!