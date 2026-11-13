"""Exercise source-loaded task policy without an installed Ascend toolchain."""
import ast
import ctypes
import os
import re
import shlex
import subprocess
import tempfile
import warnings
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
             "_parse_linalg_metadata", "_parse_ttir_metadata", "_drop_get_next_task",
             "_BmuCompileFallback", "_check_bmu_compile_fallback",
             "_bmu_compile_fallback_reason", "_compile_with_bmu_fallback",
             "_finalize_program_launch_policy", "get_auto_bind_sub_block_option",
             "linalg_to_bin_enable_npu_compile_910_95", "_linalg_to_bin_enable_npu_compile_910_95_once"}
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    namespace = {"re": re, "subprocess": subprocess, "warnings": warnings,
                 "Path": Path, "os": os, "tempfile": tempfile, "ctypes": ctypes, "shlex": shlex,
                 "_get_auto_blockify_blacklist_reasons": lambda _ir: []}
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


@pytest.mark.parametrize("task_request", [
    "hivm.hir.get_next_task",
    "hivm.hir.get_next_task[#hivm.tcore_type<aiv>]",
    '\"hivm.hir.get_next_task\"() {tcore_type = #hivm.tcore_type<aiv>} : () -> ()',
])
def test_temporary_request_filter_keeps_wait(policy, task_request):
    source = 'module {\n  func.func @kernel() {\n    ' + task_request + '\n    hivm.hir.wait_prev_task[#hivm.pipe<PIPE_S>]\n    return\n  }\n}\n'
    result = policy._drop_get_next_task(source)
    assert "get_next_task" not in result
    assert "hivm.hir.wait_prev_task[#hivm.pipe<PIPE_S>]" in result
    assert result.endswith("  }\n}\n")
    assert policy._drop_get_next_task(result) == result


def test_temporary_filter_preserves_other_ir(policy):
    source = '// hivm.hir.get_next_task\nfunc.func @kernel() {\n  hivm.hir.get_next_task_other\n  return\n}\n'
    assert policy._drop_get_next_task(source) == source


@pytest.mark.parametrize("diagnostic", [
    "error: mark-tail-buffers: cannot trace an output",
    "error: bmu plan: grouped allocations exceed the ring capacity",
    "error: 'bmu.region_free' op access is not ordered before release",
    "error: bmu task wait prefix: synchronization may precede the wait",
    "[BMU_PLAN_FALLBACK] UB allocation plan was rejected",
])
def test_compile_fallback_rebuilds_original_options(policy, diagnostic):
    data = metadata(enable_auto_early_start=True, enable_bmu_plan=None,
                    enable_early_start=True, set_workspace_multibuffer=3,
                    disable_auto_blockify_loop=False,
                    disable_hivm_kernel_end_barrier=False)
    source = "original module"
    calls = []

    def compile_once(ir, current, _opt):
        assert ir is source
        policy._parse_task_control_metadata(ir, current)
        calls.append(dict(current))
        if len(calls) == 1:
            current["workspace_size"] = 4096
            if "[BMU_PLAN_FALLBACK]" in diagnostic:
                policy._check_bmu_compile_fallback(SimpleNamespace(stderr=diagnostic.encode()), current)
            raise subprocess.CalledProcessError(1, ["bishengir-compile"], stderr=diagnostic.encode())
        assert "workspace_size" not in current
        current["workspace_size"] = 128
        return b"baseline"

    with pytest.warns(RuntimeWarning, match="fell back to the static plan"):
        binary = policy._compile_with_bmu_fallback(compile_once, source, data, None)
    assert binary == b"baseline"
    assert len(calls) == 2
    assert calls[0]["set_workspace_multibuffer"] == 0
    assert calls[0]["disable_auto_blockify_loop"] is True
    assert data["set_workspace_multibuffer"] == 3
    assert data["disable_auto_blockify_loop"] is False
    assert data["enable_auto_early_start"] is False
    assert data["enable_bmu_plan"] is False
    assert data["enable_early_start"] is False
    assert data["disable_hivm_kernel_end_barrier"] is False
    assert data["conservative_task_wait"] is True
    assert data["workspace_size"] == 128


@pytest.mark.parametrize("options,diagnostic", [
    ({"enable_bmu_plan": False}, "error: bmu plan: unsupported"),
    ({"enable_bmu_plan": True, "has_wait_prev_task": True}, "error: bmu plan: unsupported"),
    ({"enable_bmu_plan": True}, "error: unrelated failure\nIR dump: bmu.region_alloc"),
])
def test_compile_fallback_does_not_mask_other_failures(policy, options, diagnostic):
    data = metadata(**options)
    before = dict(data)
    error = subprocess.CalledProcessError(1, ["compiler"], stderr=diagnostic.encode())
    calls = []

    def compile_once(_source, current, _opt):
        calls.append(current)
        raise error

    with pytest.raises(subprocess.CalledProcessError) as caught:
        policy._compile_with_bmu_fallback(compile_once, "IR", data, None)
    assert caught.value is error
    assert len(calls) == 1
    assert data == before


def test_compile_fallback_retries_only_once(policy):
    data = metadata(enable_bmu_plan=True, set_workspace_multibuffer=2)
    before = dict(data)
    calls = []

    def compile_once(_source, current, _opt):
        calls.append(current)
        raise subprocess.CalledProcessError(1, ["compiler"], stderr=b"error: bmu plan: unsupported")

    with pytest.warns(RuntimeWarning), pytest.raises(subprocess.CalledProcessError):
        policy._compile_with_bmu_fallback(compile_once, "IR", data, None)
    assert len(calls) == 2
    assert data == before


def test_successful_bmu_compilation_keeps_effective_metadata(policy):
    data = metadata(enable_auto_early_start=True, set_workspace_multibuffer=2)

    def compile_once(source, current, _opt):
        policy._parse_task_control_metadata(source, current)
        current["workspace_size"] = 64
        return b"optimized"

    assert policy._compile_with_bmu_fallback(compile_once, "IR", data, None) == b"optimized"
    assert data["enable_bmu_plan"] is True
    assert data["set_workspace_multibuffer"] == 0
    assert data["workspace_size"] == 64
    assert "bmu_plan_fallback_reason" not in data


@pytest.mark.parametrize("rejection", ["remark", "error"])
@pytest.mark.parametrize("auto_mode", [False, True])
def test_public_compiler_retries_command_and_binary(policy, monkeypatch, rejection, auto_mode):
    namespace = policy.linalg_to_bin_enable_npu_compile_910_95.__globals__
    raw = ast.parse((Path(__file__).resolve().parents[2] / "backend/compiler.py").read_text())
    function = next(n for n in raw.body if isinstance(n, ast.FunctionDef) and
                    n.name == "_linalg_to_bin_enable_npu_compile_910_95_once")
    keys = {n.slice.value for n in ast.walk(function) if isinstance(n, ast.Subscript) and
            isinstance(n.value, ast.Name) and n.value.id == "metadata" and
            isinstance(n.slice, ast.Constant) and isinstance(n.slice.value, str)}
    data = {key: None for key in keys}
    data.update(metadata(enable_auto_early_start=auto_mode, enable_bmu_plan=True, enable_early_start=True,
                         set_workspace_multibuffer=3, disable_auto_blockify_loop=False,
                         disable_hivm_kernel_end_barrier=False, program_grid_transforms=None,
                         program_grid_mapping_applied=False, row_coalescing_applied=False,
                         has_auto_blockify_blacklist_op=False, enable_auto_bind_sub_block=None))
    source = 'module attributes {mix_mode = "aiv", parallel_mode = "simd"} {\n func.func @kernel() { return }\n}'
    options = SimpleNamespace(debug=False, target_arch="Ascend960", is_pure_simt=False)
    stubs = {
        "get_common_bishengir_compile_options": lambda _metadata: [],
        "_check_bishengir_api_change": lambda: True,
        "_is_auto_map_parallel_blocks_enabled": lambda: True,
        "NPUUtils": lambda: SimpleNamespace(has_device_limit=lambda: False),
        "is_cann_version_at_least": lambda *_args: True,
        "force_disable_ffts": lambda _arch: False,
        "_is_ascend_sanitizer_enabled": lambda: False,
        "_is_debug_line_info_disabled": lambda: True,
        "_enable_msdebug": lambda: False,
        "_enable_print_ub_bits": lambda: False,
        "_get_npucompiler_path": lambda: ("bishengir-compile", {}),
        "_full_row_copy_compile_options": lambda *_args: [],
        "knobs": SimpleNamespace(autotuning=SimpleNamespace(print=False)),
    }
    for key, value in stubs.items():
        monkeypatch.setitem(namespace, key, value)
    commands = []

    def run(command, **_kwargs):
        commands.append(command)
        assert Path(command[1]).read_text() == source
        if len(commands) == 1 and rejection == "error":
            raise subprocess.CalledProcessError(1, command, stderr=b"error: bmu plan: capacity")
        Path(command[-1] + ".o").write_bytes(b"rejected" if len(commands) == 1 else b"baseline")
        stderr = b"[BMU_PLAN_FALLBACK] rejected" if len(commands) == 1 else b""
        return subprocess.CompletedProcess(command, 0, stdout=b"UB size = 512 bits", stderr=stderr)

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.warns(RuntimeWarning):
        result = policy.linalg_to_bin_enable_npu_compile_910_95(source, data, options)
    assert result == b"baseline"
    assert len(commands) == 2
    first, second = commands
    assert "--enable-bmu-plan=True" in first
    assert "--set-workspace-multibuffer=0" in first
    assert ("--enable-auto-blockify-loop" not in first) == auto_mode
    assert "--enable-bmu-plan=False" in second
    assert "--enable-auto-early-start=False" in second
    assert "--enable-conservative-task-wait=true" in second
    assert "--set-workspace-multibuffer=3" in second
    assert "--enable-auto-blockify-loop" in second
    assert "--disable-hivm-kernel-end-barrier" not in second
    assert any("--early-start-enable=false" in arg for arg in second)
    assert data["required_ub_bits"] == 512
    assert data["enable_bmu_plan"] is False
