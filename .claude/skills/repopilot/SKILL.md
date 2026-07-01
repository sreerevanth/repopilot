```markdown
# repopilot Development Patterns

> Auto-generated skill from repository analysis

## Overview
This skill teaches you the core development patterns and conventions used in the `repopilot` Python codebase. You'll learn about file organization, code style, commit patterns, and how to write and run tests. The repository uses Python with no detected framework, and follows clear naming and import/export conventions to ensure maintainability and readability.

## Coding Conventions

### File Naming
- Use **snake_case** for all file names.
  - Example: `data_processor.py`, `user_utils.py`

### Import Style
- Use **relative imports** within the package.
  - Example:
    ```python
    from .utils import helper_function
    ```

### Export Style
- Use **named exports** (explicitly define what is exported).
  - Example:
    ```python
    __all__ = ['main_function', 'HelperClass']
    ```

### Commit Patterns
- **Type:** Mixed (features, fixes, etc.)
- **Prefix:** Use `feat` for new features.
  - Example commit message: `feat: add support for batch processing`
- **Length:** Keep commit messages concise (average ~53 characters).

## Workflows

### Add a New Feature
**Trigger:** When you need to implement a new feature.
**Command:** `/add-feature`

1. Create a new Python file using snake_case if needed.
2. Implement the feature using relative imports for internal modules.
3. Add named exports to the file.
4. Write or update tests in a corresponding `*.test.*` file.
5. Commit your changes with a message starting with `feat:`.
   - Example: `feat: implement data normalization utility`

### Run Tests
**Trigger:** When you want to verify your code changes.
**Command:** `/run-tests`

1. Identify all test files matching the pattern `*.test.*`.
2. Run the tests using your preferred Python test runner (e.g., `pytest`, `unittest`).
3. Review the results and fix any failing tests.

### Refactor Code
**Trigger:** When improving code structure or readability.
**Command:** `/refactor`

1. Rename files or functions to follow snake_case if necessary.
2. Update imports to use relative paths.
3. Ensure all exports are named.
4. Update or add tests as needed.
5. Commit changes with a clear message (e.g., `refactor: improve import structure`).

## Testing Patterns

- **Framework:** Not explicitly specified; use standard Python testing tools.
- **File Pattern:** Test files are named using `*.test.*` (e.g., `module.test.py`).
- **Example Test File:**
  ```python
  # my_module.test.py
  import unittest
  from .my_module import my_function

  class TestMyFunction(unittest.TestCase):
      def test_basic(self):
          self.assertEqual(my_function(2), 4)
  ```

## Commands
| Command        | Purpose                                   |
|----------------|-------------------------------------------|
| /add-feature   | Start workflow to add a new feature       |
| /run-tests     | Run all tests in the repository           |
| /refactor      | Start workflow to refactor code           |
```
