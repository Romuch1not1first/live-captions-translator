# Contributing to Live Captions Interactive Translator

Thank you for your interest in contributing to this project! This document provides guidelines and information for contributors.

## How to Contribute

### Reporting Issues

1. **Bug Reports**: If you find a bug, please create an issue with:
   - Clear description of the problem
   - Steps to reproduce
   - Expected vs actual behavior
   - System information (Windows version, Python version)
   - Screenshots if applicable

2. **Feature Requests**: For new features, please include:
   - Clear description of the proposed feature
   - Use case and motivation
   - Any implementation ideas (optional)

### Code Contributions

1. **Fork the Repository**: Create your own fork of the project
2. **Create a Branch**: Use descriptive branch names like `feature/new-translation-service` or `fix/scroll-issue`
3. **Make Changes**: Follow the coding standards below
4. **Test Your Changes**: Ensure the application works correctly
5. **Submit a Pull Request**: Provide a clear description of your changes

## Coding Standards

### Python Style
- Follow PEP 8 guidelines
- Use type hints where appropriate
- Write clear, descriptive variable and function names
- Add docstrings for functions and classes

### Code Structure
- Keep functions focused and single-purpose
- Use meaningful comments for complex logic
- Maintain the existing architecture patterns
- Follow the existing error handling approach

### Testing
- Test your changes thoroughly
- Verify the application works with different languages
- Test edge cases and error conditions
- Ensure no regressions in existing functionality

## Development Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/live-captions-translator.git
   cd live-captions-translator
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Tesseract OCR**:
   - Download from: https://github.com/UB-Mannheim/tesseract/wiki
   - Install to default location

4. **Run the Application**:
   ```bash
   python Caption_live_translater.py
   ```

## Areas for Contribution

### High Priority
- **Translation Services**: Add support for more translation APIs
- **Language Support**: Improve fallback translations for more languages
- **Performance**: Optimize OCR and text processing
- **Accessibility**: Improve accessibility features

### Medium Priority
- **UI Improvements**: Better visual design and user experience
- **Configuration**: More customization options
- **Logging**: Enhanced logging and debugging features
- **Documentation**: Improve code documentation

### Low Priority
- **Platform Support**: Support for other operating systems
- **Advanced Features**: Voice synthesis, custom dictionaries
- **Integration**: Integration with other applications

## Pull Request Process

1. **Update Documentation**: Update README.md if needed
2. **Add Tests**: Include tests for new functionality
3. **Update Version**: Increment version numbers if appropriate
4. **Check Compatibility**: Ensure changes work with existing features
5. **Review**: Self-review your code before submitting

### Pull Request Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tested on Windows 10/11
- [ ] Tested with different languages
- [ ] No regressions in existing functionality

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings or errors
```

## Code Review Process

1. **Automated Checks**: All PRs must pass automated checks
2. **Manual Review**: At least one maintainer will review
3. **Testing**: Changes will be tested before merging
4. **Feedback**: Constructive feedback will be provided

## Community Guidelines

### Be Respectful
- Use welcoming and inclusive language
- Be respectful of differing viewpoints
- Accept constructive criticism gracefully

### Be Collaborative
- Focus on what's best for the community
- Show empathy towards other community members
- Help others learn and grow

### Be Professional
- Keep discussions focused on the project
- Avoid off-topic conversations
- Follow the project's code of conduct

## Getting Help

- **Documentation**: Check README.md and code comments
- **Issues**: Search existing issues before creating new ones
- **Discussions**: Use GitHub Discussions for questions
- **Contact**: Reach out to maintainers for guidance

## Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation

Thank you for contributing to make this project better!
