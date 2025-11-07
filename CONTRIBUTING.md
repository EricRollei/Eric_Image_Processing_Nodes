# Contributing to Eric's Image Processing Nodes

Thank you for your interest in contributing to this project! This document provides guidelines for contributing code, documentation, and other improvements.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Code Style Guidelines](#code-style-guidelines)
- [License Agreement](#license-agreement)
- [Pull Request Process](#pull-request-process)
- [Attribution Requirements](#attribution-requirements)

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors, regardless of experience level, gender identity, sexual orientation, disability, personal appearance, race, ethnicity, age, religion, or nationality.

### Expected Behavior

- Be respectful and constructive in discussions
- Provide helpful feedback
- Focus on the code and ideas, not the person
- Be patient with newcomers
- Give credit where it's due

### Unacceptable Behavior

- Harassment, discrimination, or offensive comments
- Trolling or deliberately inflammatory behavior
- Publishing others' private information without permission
- Any conduct that would be considered inappropriate in a professional setting

## How Can I Contribute?

### Reporting Bugs

**Before submitting a bug report:**

1. Check the existing [GitHub Issues](https://github.com/EricRollei/Eric_Image_Processing_Nodes/issues)
2. Test with the latest version
3. Ensure it's not a ComfyUI issue (try with vanilla ComfyUI)

**When submitting a bug report, include:**

- ComfyUI version
- Python version
- Operating system
- GPU/CUDA version (if using GPU features)
- Complete error message and traceback
- Minimal reproducible example
- Screenshots or workflow JSON if applicable

### Suggesting Enhancements

We welcome suggestions for new features! Please include:

- Clear description of the feature
- Use case and motivation
- Example of how it would work
- Any relevant research papers or algorithms
- Whether you're willing to implement it yourself

### Contributing Code

We welcome contributions of:

- **New Nodes**: Additional image processing algorithms
- **Bug Fixes**: Corrections to existing code
- **Performance Improvements**: Optimizations
- **Documentation**: Improvements to guides and examples
- **Tests**: Unit tests and integration tests
- **Examples**: Sample workflows and use cases

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork:
git clone https://github.com/YOUR_USERNAME/Eric_Image_Processing_Nodes.git
cd Eric_Image_Processing_Nodes
```

### 2. Set Up Development Environment

```bash
# Create a virtual environment
python -m venv .venv

# Activate it
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8 mypy
```

### 3. Install Pre-commit Hooks (Optional but Recommended)

```bash
pip install pre-commit
pre-commit install
```

### 4. Link to ComfyUI

```bash
# Create a symlink in ComfyUI's custom_nodes directory
# Windows (run as administrator):
mklink /D "C:\path\to\ComfyUI\custom_nodes\Eric_Image_Processing_Nodes" "C:\path\to\your\clone"

# Linux/Mac:
ln -s /path/to/your/clone /path/to/ComfyUI/custom_nodes/Eric_Image_Processing_Nodes
```

## Code Style Guidelines

### Python Style

We follow **PEP 8** with some modifications:

```python
# Line length: 100 characters (not 79)
# Use 4 spaces for indentation (no tabs)
# Use double quotes for strings (unless single quotes avoid escaping)
```

### File Header Template

All new node files must include a proper header:

```python
"""
[Node Name] for ComfyUI
[Brief description]

Author: Eric Hiss (GitHub: EricRollei)
License: See LICENSE file in repository root

[If using external algorithm/model:]
Original Algorithm:
    Paper: [Title]
    Authors: [Authors]
    Source: [URL]
    License: [License]
    
    Citation:
    @[type]{[key],
      title={[Title]},
      author={[Authors]},
      year={[Year]}
    }

Dependencies:
    - Library (License)
    - Library (License)
"""
```

### Node Structure

Follow the established pattern:

```python
class YourNode(BaseImageProcessingNode):
    """
    Detailed docstring explaining:
    - What the node does
    - When to use it
    - Key parameters
    - Typical use cases
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                # ... other inputs
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("processed_image",)
    FUNCTION = "process"
    CATEGORY = "Eric's Image Processing/Your Category"
    
    def process(self, image, **kwargs):
        # Process logic here
        pass
```

### Processing Script Structure

Keep processing logic separate from node code:

```python
# scripts/your_algorithm.py
"""
Pure processing functions (no ComfyUI dependencies)
Can be tested standalone
"""

def your_algorithm(image_np: np.ndarray, **params) -> np.ndarray:
    """
    Process image with your algorithm.
    
    Args:
        image_np: Input image as numpy array [H, W, C] with values 0-255
        **params: Algorithm parameters
        
    Returns:
        Processed image as numpy array [H, W, C] with values 0-255
    """
    # Ensure contiguous array for OpenCV compatibility
    image_np = np.ascontiguousarray(image_np)
    
    # Your processing logic
    
    # Ensure output is contiguous
    result = np.ascontiguousarray(result)
    return result
```

### Testing

Add tests for new functionality:

```python
# tests/test_your_node.py
import pytest
import numpy as np
from scripts.your_algorithm import your_algorithm

def test_your_algorithm_basic():
    """Test basic functionality"""
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    result = your_algorithm(image)
    assert result.shape == image.shape
    assert result.dtype == np.uint8

def test_your_algorithm_edge_cases():
    """Test edge cases"""
    # Test with all black
    black = np.zeros((100, 100, 3), dtype=np.uint8)
    result = your_algorithm(black)
    assert result is not None
```

## License Agreement

### By Contributing, You Agree That:

1. **Your contributions will be licensed** under the same dual license as the project:
   - Non-commercial: CC BY-NC 4.0
   - Commercial: Separate license required (contact eric@historic.camera)

2. **You have the right to contribute** the code (it's your original work or you have permission)

3. **You will provide proper attribution** for any:
   - External algorithms or models used
   - Research papers implemented
   - Third-party code adapted
   - Dependencies added

4. **You grant the project maintainer** (Eric Hiss) the right to:
   - Include your contribution in the project
   - Modify your contribution as needed
   - Relicense your contribution if the project license changes

### For Significant Contributions

For major contributions (new models, significant features), please:

1. Add your name to the CONTRIBUTORS list
2. Include your email for attribution (optional)
3. Document your contribution in the commit message

## Pull Request Process

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

### 2. Make Your Changes

- Follow the code style guidelines
- Add proper headers to new files
- Update documentation as needed
- Add tests for new functionality
- Ensure all tests pass

### 3. Commit Your Changes

Use clear, descriptive commit messages:

```bash
git commit -m "Add [Feature]: Brief description

- Detailed point 1
- Detailed point 2
- References #issue-number if applicable"
```

### 4. Push to Your Fork

```bash
git push origin feature/your-feature-name
```

### 5. Submit Pull Request

On GitHub:

1. Go to your fork
2. Click "New Pull Request"
3. Select your branch
4. Fill out the PR template:
   - Description of changes
   - Motivation and context
   - Testing performed
   - Screenshots if UI changes
   - Checklist items completed

### 6. Code Review

- Respond to review comments
- Make requested changes
- Update the PR
- Be patient and respectful

### 7. Merge

Once approved, the maintainer will merge your PR. Thank you! 🎉

## Attribution Requirements

### When Adding External Algorithms

You **must** include:

1. **Original paper citation** in docstring
2. **Source repository link** if applicable
3. **License information** of the original work
4. **Author credits**
5. **Link to pretrained weights** if used

### When Adding Dependencies

Update these files:

1. `requirements.txt` - Add the dependency
2. `LICENSE` - Add dependency license info
3. `Docs/MODEL_WEIGHTS.md` - If models are involved
4. File header - List new dependencies

### Example Attribution

```python
"""
Original Algorithm:
    Paper: "Algorithm Name"
    Authors: Smith et al., Conference 2023
    Source: https://github.com/author/repo
    License: MIT License
    Pretrained weights: https://github.com/author/repo/releases
    
    Citation:
    @inproceedings{smith2023algorithm,
      title={Algorithm Name},
      author={Smith, John and Doe, Jane},
      booktitle={Conference},
      year={2023}
    }
    
Implementation Notes:
    [Your modifications or adaptations]
    
Contributed by: [Your Name] (GitHub: @yourusername)
Date: [Month Year]
"""
```

## Questions?

- **Issues**: Use [GitHub Issues](https://github.com/EricRollei/Eric_Image_Processing_Nodes/issues) for questions
- **Discussions**: Use [GitHub Discussions](https://github.com/EricRollei/Eric_Image_Processing_Nodes/discussions) for general discussions
- **Email**: For private matters, contact eric@historic.camera

## Recognition

Contributors will be acknowledged in:

- The CONTRIBUTORS file
- Release notes
- Documentation credits section

Thank you for contributing! Your work helps the ComfyUI community! 🚀
