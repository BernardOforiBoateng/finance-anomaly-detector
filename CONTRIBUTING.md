# Contributing to Personal Finance Anomaly Detector

Thank you for your interest in contributing to this fraud detection project! This document provides guidelines for contributing to the codebase.

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Node.js 14+
- Git

### Development Setup
1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/yourusername/finance-anomaly-detector.git
   cd finance-anomaly-detector
   ```
3. Set up the backend:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```
4. Set up the frontend:
   ```bash
   cd frontend
   npm install
   ```

## 🔄 Development Workflow

### 1. Create a Branch
```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes
- Follow the coding standards below
- Add tests for new functionality
- Update documentation as needed

### 3. Test Your Changes
```bash
# Backend tests
cd backend && python -m pytest

# Frontend tests  
cd frontend && npm test

# Integration tests
python test_integration.py
```

### 4. Commit Your Changes
```bash
git add .
git commit -m "feat: add new fraud detection feature"
```

Use conventional commit messages:
- `feat:` new feature
- `fix:` bug fix
- `docs:` documentation changes
- `test:` test additions/changes
- `refactor:` code refactoring

### 5. Submit a Pull Request
- Push to your fork
- Create a pull request from your branch to `main`
- Provide a clear description of your changes

## 📝 Coding Standards

### Python (Backend)
- Follow PEP 8 style guide
- Use type hints where possible
- Maximum line length: 88 characters
- Use meaningful variable names
- Add docstrings for functions and classes

### TypeScript (Frontend)
- Use TypeScript strict mode
- Follow React best practices
- Use functional components with hooks
- Add proper type definitions

### Machine Learning
- Document model changes thoroughly
- Include performance metrics for new models
- Validate models on test data
- Follow MLOps best practices

## 🧪 Testing Guidelines

### Backend Tests
- Unit tests for individual functions
- Integration tests for API endpoints
- Model performance tests
- Coverage target: >80%

### Frontend Tests
- Component unit tests
- Integration tests for user flows
- API client tests
- Accessibility tests

### ML Tests
- Model accuracy validation
- Feature engineering tests
- Data pipeline tests
- Model serving tests

## 📊 Types of Contributions

### 🐛 Bug Reports
Include:
- Description of the issue
- Steps to reproduce
- Expected vs actual behavior
- Environment details
- Screenshots if relevant

### ✨ Feature Requests
Include:
- Use case description
- Proposed solution
- Alternative solutions considered
- Impact assessment

### 📖 Documentation
- API documentation improvements
- Tutorial enhancements
- Code comment improvements
- Architecture documentation

### 🤖 Model Improvements
- New algorithms or techniques
- Feature engineering enhancements
- Performance optimizations
- Bias detection and mitigation

## 🔍 Code Review Process

### For Contributors
- Ensure all tests pass
- Update documentation
- Follow coding standards
- Keep changes focused and atomic

### For Reviewers
- Check code quality and standards
- Verify test coverage
- Validate performance impact
- Ensure documentation is updated

## 📚 Resources

### Documentation
- [API Documentation](docs/API.md)
- [Model Documentation](docs/MODEL.md)
- [Architecture Overview](docs/ARCHITECTURE.md)

### Tools
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://reactjs.org/docs/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

## 🤝 Community Guidelines

### Be Respectful
- Use inclusive language
- Respect different viewpoints
- Provide constructive feedback
- Help newcomers learn

### Be Collaborative
- Share knowledge and expertise
- Ask questions when unclear
- Offer help to others
- Celebrate contributions

## 📞 Getting Help

- Open an issue for bugs or feature requests
- Use discussions for questions
- Check existing issues before creating new ones
- Provide clear and detailed information

## 🏆 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes for significant contributions
- Project documentation

Thank you for contributing to the Personal Finance Anomaly Detector! 🛡️