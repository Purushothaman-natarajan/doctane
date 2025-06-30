
from typing import List, Any

__all__ = ["NestedObject"]


def _addindent(s_: str, num_spaces: int) -> str:
    """Indent all lines of a multiline string except the first."""
    lines = s_.split("\n")
    if len(lines) == 1:
        return s_
    first = lines.pop(0)
    indented = [(num_spaces * " ") + line for line in lines]
    return first + "\n" + "\n".join(indented)


class NestedObject:
    """Base class for all nested objects in Doctane with clean hierarchical string representation."""

    _children_names: List[str]

    def extra_repr(self) -> str:
        """Extra information to be printed in the representation, meant to be overridden."""
        return ""

    def __repr__(self) -> str:
        """Custom string representation showing child modules and extra information."""
        extra_lines: List[str] = []
        extra_repr = self.extra_repr()

        if extra_repr:
            extra_lines = extra_repr.split("\n")

        child_lines: List[str] = []
        if hasattr(self, "_children_names"):
            for key in self._children_names:
                child = getattr(self, key)

                if isinstance(child, list) and child:
                    # Handle list of children
                    child_str = ",\n".join([repr(subchild) for subchild in child])
                    if len(child) > 1:
                        child_str = _addindent(f"\n{child_str},", 2) + "\n"
                    child_str = f"[{child_str}]"
                else:
                    child_str = repr(child)
                
                child_str = _addindent(child_str, 2)
                child_lines.append(f"({key}): {child_str}")

        # Combine all lines
        lines = extra_lines + child_lines
        main_str = self.__class__.__name__ + "("

        if lines:
            # Simple one-liner
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        return main_str
