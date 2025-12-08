Here's the equivalent test case using pytest style:

```python
import pytest

def add(a, b):
    return a + b

def test_add_simple():
    assert add(2, 3) == 5

def test_add_large_numbers():
    assert add(1000, 2000) == 3000

def test_add_zero_positive():
    assert add(0, 5) == 5

def test_add_negative_positive():
    assert add(-2, 3) == 1

def test_add_invalid_type_string():
    with pytest.raises(TypeError):
        add('hello', 3)

def test_add_invalid_type_list():
    with pytest.raises(TypeError):
        add([1, 2], 3)

def test_add_non_numeric():
    with pytest.raises(TypeError):
        add('a', 3)

def test_add_infinity():
    with pytest.raises(TypeError):
        add(float('inf'), 3)

def test_add_invalid_type_none():
    with pytest.raises(TypeError):
        add(None, 3)

def test_add_invalid_type_dict():
    with pytest.raises(TypeError):
        add({'a': 1}, 3)

def test_add_invalid_type_set():
    with pytest.raises(TypeError):
        add({1, 2}, 3)

def test_add_invalid_type_tuple():
    with pytest.raises(TypeError):
        add((1, 2), 3)

def test_add_invalid_type_complex():
    with pytest.raises(TypeError):
        add(1 + 2j, 3)
```

You can run the tests using the following command:

```bash
pytest test_add.py
```

Note that pytest provides more features and flexibility than the built-in unittest module, such as better error messages and support for fixtures. However, the test code itself is similar to the original unittest code.