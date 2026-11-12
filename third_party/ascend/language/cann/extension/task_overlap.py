# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

__all__ = ["task_overlap"]

# Read by the Ascend patch on `ASTSource.make_ir`; kept as a plain attribute so
# the compiler core does not have to import the Ascend language module.
TASK_OVERLAP_ATTR = "_ascend_task_overlap"


class TaskOverlapConfig:
    """Per-kernel task-overlap setting extracted from the decorator."""

    __slots__ = ("depends_on_prev", )

    def __init__(self, depends_on_prev: bool) -> None:
        self.depends_on_prev = depends_on_prev


def task_overlap(depends_on_prev: bool = True):
    """Record where this kernel may wait for its predecessor.

    The decorator only emits the per-kernel hint ``hivm.task_overlap``; it does
    not switch on the early-start pipeline. Compile the kernel with
    ``enable_auto_early_start=True`` and ``enable_early_start=True`` to let
    the BMU planner return the Main allocation while the Tail output is still
    being written back and dispatch the successor task before this one drains.
    Without that pipeline the hint is inert.

    ``depends_on_prev`` states whether this kernel reads the output of the task
    that ran before it:

    - ``True`` (default): disable automatic task-control insertion. Use a
      hand-written ``wait_prev_task`` or let BiSheng insert the wait.
    - ``False``: allow automatic insertion after the first independent input
      load. A hand-written ``wait_prev_task`` always takes precedence and
      disables automatic insertion for this kernel.

    ``enable_early_start`` separately controls BiSheng's low-level early-start
    support. Automatic insertion also removes the A6 kernel-end barrier.

    The decorator may be applied above or below ``@triton.jit``, for example::

        @triton.jit
        @al.task_overlap(depends_on_prev=False)
        def kernel(...):
            ...

    :param depends_on_prev: Whether the kernel depends on the predecessor's
        output.
    :type depends_on_prev: bool
    """
    if not isinstance(depends_on_prev, bool):
        raise TypeError(f"depends_on_prev must be a bool, got {type(depends_on_prev).__name__}")

    config = TaskOverlapConfig(depends_on_prev)

    def decorator(fn):
        setattr(fn, TASK_OVERLAP_ATTR, config)
        return fn

    return decorator
