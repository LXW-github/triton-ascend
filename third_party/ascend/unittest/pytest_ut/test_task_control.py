"""Exercise source-loaded task policy without an installed Ascend toolchain."""
import ast
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.backend("none")


@pytest.fixture(scope="module")
def policy():
    source = Path(__file__).resolve().parents[2] / "backend" / "compiler.py"
    tree = ast.parse(source.read_text())
    names = {"_is_a6_target_arch", "_parse_task_control_metadata",
             "_set_bisheng_mllvm_option", "_task_control_bisheng_options",
             "_parse_linalg_metadata", "_parse_ttir_metadata"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {"re": re, "_get_auto_blockify_blacklist_reasons": lambda _ir: []}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in names})


def metadata(**kwargs):
    return dict(target=SimpleNamespace(arch="Ascend960"), **kwargs)


@pytest.mark.parametrize("enabled", [False, True])
def test_backend_switch_is_independent(policy, enabled):
    data = metadata(enable_early_start=enabled)
    policy._parse_task_control_metadata("module {}", data)
    options = policy._task_control_bisheng_options(data)
    assert f"--early-start-enable={str(enabled).lower()}" in options.split()
    assert not data.get("enable_auto_early_start")
    assert not data.get("enable_bmu_plan")
    assert "cce-insert-wait-prev-task-enable" not in options


def test_auto_mode_removes_barrier_and_enables_ring(policy):
    data = metadata(enable_auto_early_start=True, disable_hivm_kernel_end_barrier=False)
    policy._parse_task_control_metadata("module {}", data)
    assert data["enable_bmu_plan"] is True
    assert data["disable_hivm_kernel_end_barrier"] is True
    assert data["disable_auto_blockify_loop"] is True
    assert data["set_workspace_multibuffer"] == 0
    assert policy._task_control_bisheng_options(data) is None


@pytest.mark.parametrize("text", [
    "hivm.hir.wait_prev_task[#hivm.pipe<PIPE_S>]",
    '\"hivm.hir.wait_prev_task\"() {pipe = #hivm.pipe<PIPE_S>} : () -> ()',
    'module attributes {hivm.task_overlap = "depends"} {}',
])
def test_manual_wait_and_dependency_override_auto(policy, text):
    data = metadata(enable_early_start=True, enable_auto_early_start=True)
    policy._parse_task_control_metadata(text, data)
    assert data["enable_auto_early_start"] is False
    assert not data.get("disable_hivm_kernel_end_barrier")
    options = policy._task_control_bisheng_options(data)
    assert "--early-start-enable=true" in options.split()
    assert ("--cce-insert-wait-prev-task-enable=false" in options.split()) == data["has_wait_prev_task"]


def test_comments_do_not_disable_auto(policy):
    data = metadata(enable_auto_early_start=True)
    policy._parse_task_control_metadata('// hivm.hir.wait_prev_task\n// hivm.task_overlap = "depends"', data)
    assert data["enable_auto_early_start"] is True


@pytest.mark.parametrize("text", [
    "-mllvm -cce-insert-wait-prev-task-enable=true -mllvm -other=1",
    "-mllvm --cce-insert-wait-prev-task-enable true -mllvm -other=1",
    "--cce-insert-wait-prev-task-enable=true -mllvm -other=1",
])
def test_manual_wait_overrides_backend_wait_option(policy, text):
    data = metadata(bisheng_options=text, enable_early_start=True)
    policy._parse_task_control_metadata("hivm.hir.wait_prev_task [<PIPE_S>]", data)
    tokens = policy._task_control_bisheng_options(data).split()
    assert tokens.count("--cce-insert-wait-prev-task-enable=false") == 1
    assert not any("wait-prev-task-enable=true" in token for token in tokens)
    assert "-other=1" in tokens
    assert tokens.count("-mllvm") == 3


def test_absent_wait_preserves_backend_default(policy):
    data = metadata()
    policy._parse_task_control_metadata("module {}", data)
    assert policy._task_control_bisheng_options(data) is None


@pytest.mark.parametrize("arch", ["Ascend950", "Ascend910B1"])
def test_non_a6_does_not_enable_task_control(policy, arch):
    data = metadata(enable_early_start=True, enable_auto_early_start=True)
    data["target"].arch = arch
    policy._parse_task_control_metadata("module {}", data)
    assert data["enable_early_start"] is None
    assert data["enable_auto_early_start"] is None
    assert policy._task_control_bisheng_options(data) is None


@pytest.mark.parametrize("kind", ["linalg", "ttir"])
def test_both_compilation_routes_apply_priority(policy, kind):
    data = metadata(enable_early_start=True, enable_auto_early_start=True)
    if kind == "linalg":
        text = 'module attributes {mix_mode = "aiv", parallel_mode = "simd", hivm.task_overlap = "depends"} {\n func.func @kernel() { return }\n}'
        _, result = policy._parse_linalg_metadata(text, data)
    else:
        text = 'module attributes {hivm.task_overlap = "depends"} {\n tt.func public @kernel() { tt.return }\n}'
        result = policy._parse_ttir_metadata(text, data)
    assert result["enable_auto_early_start"] is False
    assert result["enable_early_start"] is True
