from typing import Generic, TypeVar

T = TypeVar('T')

class A(Generic[T]):
    pass

class B(A[int]):
    pass

def foo(x: A) -> None:
    reveal_type(x)
    if isinstance(x, B):
        reveal_type(x)
