import traceback
import warnings
from datetime import datetime
from pathlib import Path

import torch

from tests.old.test_mock_agents.mock_agent_original import MockAgentOriginal


class InOutChecker:
    """
    is added to agent new, gets agent original and confirms that all input and outputs match.
    """

    def __init__(self, agent_original: MockAgentOriginal, raise_at_warnings: bool):
        self.agent_original = agent_original
        self.call_id = 0
        # whethere to create a traceback even when the missmatch is only worthy a warning 
        # (e.g. a masked value missmatch)
        self.raise_at_warnings = raise_at_warnings

        # Create timestamped output folder
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.output_dir = Path(__file__).parent / "test_outputs" / ts
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._verbose = open(self.output_dir / "verbose.log", "w")
        self._standard = open(self.output_dir / "standard.log", "w")

    def close(self):
        self._verbose.close()
        self._standard.close()

    def _write(self, msg: str, *, verbose_only: bool = False):
        """Write to verbose.log always; write to standard.log and stdout unless verbose_only."""
        self._verbose.write(msg + "\n")
        if not verbose_only:
            self._standard.write(msg + "\n")
            print(msg)

    def check_are_different_OLD(self, args_new: dict, res_new: dict):

        # different function names/signatures -> ValueError
        # otherwise different args -> Warning
        # different returns -> ValueError

        # True: raise an expetion if arguments differ, False: only warn
        raise_args_mismatch = True
        raise_returns_mismatch = True

        args_original = self.agent_original.args_sequence[self.call_id]
        res_original = self.agent_original.returns_sequence[self.call_id]

        f_name = args_original["function_name"]

        def args_mismatch_reaction(key: str, value_new, value_original):
            msg = (
                f"{f_name} call {self.call_id} arg '{key}' mismatch: original={value_original.shape} "
                f"new={value_new.shape}"
            )

            warnings.warn(msg)

        def result_mismatch_reaction(key: str, value_new, value_original):
            msg = (
                f"{f_name} call {self.call_id} return '{key}' mismatch: original={value_original.shape} "
                f"new={value_new.shape}"
            )
            warnings.warn(msg)

        ########
        # Check same function
        ########

        if args_new["function_name"] != args_original["function_name"]:
            raise ValueError(
                f"{f_name} call {self.call_id} arg 'function_name' "
                f"mismatch: original='{args_original['function_name']}' new='{args_new['function_name']}'"
            )

        #########
        # Compare Signatures
        #########

        if args_new.keys() != args_original.keys():
            raise ValueError(
                f"{f_name} call {self.call_id} arg keys mismatch: "
                f"original={args_original.keys()} new={args_new.keys()}"
            )
        if res_new.keys() != res_original.keys():
            raise ValueError(
                f"{f_name} call {self.call_id} return keys mismatch: "
                f"original={res_original.keys()} new={res_new.keys()}"
            )

        ##########
        # Compare RNG states
        ##########
        # states in
        if (args_new["in_rng_state"] != args_original["in_rng_state"]).any():
            raise ValueError(
                f"{f_name} call {self.call_id} arg 'in_rng_state' mismatch: "
                f"original={args_original['in_rng_state']} new={args_new['in_rng_state']}"
            )
        # states out
        elif (res_new["out_rng_state"] != res_original["out_rng_state"]).any():
            raise ValueError(
                f"{f_name} call {self.call_id} return 'out_rng_state' mismatch: "
                f"original={res_original['out_rng_state']} new={res_new['out_rng_state']}"
            )
        else:
            print(f"RNG states match: {f_name} call {self.call_id}")
        ##########
        # Compare Args
        ##########
        args_matches = 0
        args_miss = 0
        for key, value in args_new.items():
            value_original = args_original[key]
            if value is None and value_original is None:
                args_matches += 1
            elif isinstance(value, str):
                if value != value_original:
                    raise ValueError(
                        f"{f_name} call {self.call_id} arg '{key}' mismatch: "
                        f"original='{value_original}' new='{value}'"
                    )
                    args_miss += 1
                else:
                    args_matches += 1
            elif isinstance(value, list):
                if not all(torch.equal(v, vo) for v, vo in zip(value, value_original)):
                    self.show_tensor_diff(
                        key, f_name, orig=torch.stack(value_original), new=torch.stack(value)
                    )
                    args_mismatch_reaction(key, value, value_original)
                    args_miss += 1
                else:
                    args_matches += 1
            # for memory frames we currently only compare the masked frames
            elif key == "memory_frames":
                mask_new = args_new["memory_mask"]
                mask_original = args_original["memory_mask"]
                # testing if the masks are equal first, to avoid thinking the frames are wrong when the masks
                # actuall differ.
                # now they should be printed anyway,
                # if not torch.equal(mask_new, mask_original):
                #   self.show_tensor_diff("memory_mask", f_name, orig = mask_original, new = mask_new)
                #   args_mismatch_reaction("memory_mask", mask_new, mask_original)

                masked_frames_new = value * mask_new.unsqueeze(-1).unsqueeze(-1)
                masked_frames_original = value_original * mask_original.unsqueeze(-1).unsqueeze(-1)
                if not torch.equal(masked_frames_new, masked_frames_original):
                    self.show_tensor_diff(
                        "(masked_)memory_frames",
                        f_name,
                        orig=masked_frames_original,
                        new=masked_frames_new,
                    )
                    args_mismatch_reaction(
                        "(masked_)memory_frames", masked_frames_new, masked_frames_original
                    )
                    args_miss += 1
                elif not torch.equal(value, value_original):
                    warnings.warn("Masked frames are equal, but unmasked frames still differ.")
                    args_matches += 1
                else:
                    args_matches += 1
            elif not torch.equal(value, value_original):
                self.show_tensor_diff(key, f_name, orig=value_original, new=value)
                args_mismatch_reaction(key, value, value_original)
                args_miss += 1
            else:
                args_matches += 1

        assert args_matches + args_miss == len(args_new)

        ##########
        # Compare Returns
        ##########

        returns_matches = 0
        returns_miss = 0
        for key, value in res_new.items():
            value_original = res_original[key]
            if value is None and value_original is None:
                returns_matches += 1
            elif isinstance(value, list):
                if not all(torch.equal(v, vo) for v, vo in zip(value, value_original)):
                    self.show_tensor_diff(
                        key, f_name, orig=torch.stack(value_original), new=torch.stack(value)
                    )
                    result_mismatch_reaction(key, value, value_original)
                    returns_miss += 1
                else:
                    returns_matches += 1
            elif not torch.equal(value, value_original):
                self.show_tensor_diff(key, f_name, orig=value_original, new=value)
                result_mismatch_reaction(key, value, value_original)
                returns_miss += 1
            else:
                returns_matches += 1

        assert returns_matches + returns_miss == len(res_new), (
            f"{returns_matches=}, {returns_miss=}, {len(res_new)=}"
        )

        ###########
        # finish of
        ###########

        assert args_matches + args_miss + returns_matches + returns_miss == len(args_new) + len(
            res_new
        )

        if (
            raise_args_mismatch
            and args_matches != len(args_new)
            or raise_returns_mismatch
            and returns_matches != len(res_new)
        ):
            print("|" * 100)
            raise ValueError(
                f"{f_name} call:{self.call_id}, matches: {args_matches}/{len(args_new)} args, "
                f"{returns_matches}/{len(res_new)} returns"
            )

        self.call_id += 1

        print(f"get_action_and_value call: {self.call_id} inputs and outputs match.")
        print("-" * 100)

    def check_are_different_CLEAN(self, args_new: dict, res_new: dict):
        args_original = self.agent_original.args_sequence[self.call_id]
        res_original = self.agent_original.returns_sequence[self.call_id]
        f_name = args_original["function_name"]

        # ── Structural checks (fail fast) ──
        if args_new["function_name"] != args_original["function_name"]:
            raise ValueError(
                f"Call {self.call_id}: function name mismatch: "
                f"'{args_original['function_name']}' vs '{args_new['function_name']}'"
            )
        if args_new.keys() != args_original.keys():
            raise ValueError(
                f"Call {self.call_id} ({f_name}): arg keys mismatch: "
                f"{set(args_original.keys())} vs {set(args_new.keys())}"
            )
        if res_new.keys() != res_original.keys():
            raise ValueError(
                f"Call {self.call_id} ({f_name}): return keys mismatch: "
                f"{set(res_original.keys())} vs {set(res_new.keys())}"
            )

        # ── Compare all fields, collecting results ──
        matched: list[str] = []
        mismatched: list[
            tuple[str, str, torch.Tensor, torch.Tensor]
        ] = []  # (section, key, orig, new)
        warned: list[str] = []  # fields where masked-out values differ

        def _compare_tensors(section: str, key: str, orig: torch.Tensor, new: torch.Tensor):
            if torch.equal(orig, new):
                matched.append(f"{section}/{key}")
            else:
                mismatched.append((section, key, orig, new))

        def _compare_memory_field(section: str, key: str, val_new, val_orig):
            """Compare memory_frames or memory_indices using only masked (valid) positions."""
            mask_new = (args_new if section == "arg" else res_new)["memory_mask"]
            mask_orig = (args_original if section == "arg" else res_original)["memory_mask"]
            mask_new_f = mask_new.float()
            mask_orig_f = mask_orig.float()
            if key == "memory_frames":
                masked_new = val_new * mask_new_f.unsqueeze(-1).unsqueeze(-1)
                masked_orig = val_orig * mask_orig_f.unsqueeze(-1).unsqueeze(-1)
            else:  # memory_indices — 2D, no extra unsqueeze needed
                masked_new = val_new.float() * mask_new_f
                masked_orig = val_orig.float() * mask_orig_f
            if torch.equal(masked_new, masked_orig):
                matched.append(f"{section}/{key} (masked)")
                if not torch.equal(val_new, val_orig):
                    warned.append(f"{section}/{key} (masked-out values differ)")
            else:
                mismatched.append((section, f"{key} (masked)", masked_orig, masked_new))

        def _compare_entry(section: str, key: str, val_new, val_orig):
            if val_new is None and val_orig is None:
                matched.append(f"{section}/{key}")
            elif isinstance(val_new, str):
                if val_new == val_orig:
                    matched.append(f"{section}/{key}")
                else:
                    raise ValueError(
                        f"Call {self.call_id} ({f_name}): {section} '{key}' "
                        f"string mismatch: '{val_orig}' vs '{val_new}'"
                    )
            elif isinstance(val_new, list):
                stacked_new = torch.stack(val_new)
                stacked_orig = torch.stack(val_orig)
                _compare_tensors(section, key, stacked_orig, stacked_new)
            elif key in ("memory_frames", "memory_indices"):
                _compare_memory_field(section, key, val_new, val_orig)
            elif isinstance(val_new, torch.Tensor):
                _compare_tensors(section, key, val_orig, val_new)
            else:
                matched.append(f"{section}/{key}")

        skip_keys = {"function_name", "in_rng_state"}
        skip_res_keys = {"out_rng_state"}

        # RNG states
        if (args_new["in_rng_state"] != args_original["in_rng_state"]).any():
            mismatched.append(
                ("rng", "in_rng_state", args_original["in_rng_state"], args_new["in_rng_state"])
            )
        else:
            matched.append("rng/in_rng_state")
        if (res_new["out_rng_state"] != res_original["out_rng_state"]).any():
            mismatched.append(
                ("rng", "out_rng_state", res_original["out_rng_state"], res_new["out_rng_state"])
            )
        else:
            matched.append("rng/out_rng_state")

        # Args
        for key in args_new:
            if key in skip_keys:
                continue
            _compare_entry("arg", key, args_new[key], args_original[key])

        # Returns
        for key in res_new:
            if key in skip_res_keys:
                continue
            _compare_entry("ret", key, res_new[key], res_original[key])

        # ── Print overview ──
        total = len(matched) + len(mismatched)

        if not mismatched and not warned:
            line = f"Call {self.call_id} ({f_name}) — all {total} fields match"
            self._write(line)
            self.call_id += 1
            if self.call_id % 50 == 0:
                self._flush()
            return

        header = f"Call {self.call_id} ({f_name})  —  {len(matched)}/{total} fields match, {len(warned)} warnings"
        w = max(len(header) + 4, 80)
        self._write("=" * w)
        self._write(f"  {header}")
        self._write("=" * w)

        col_w = 40
        self._write(f"  {'FIELD':<{col_w}} {'STATUS'}")
        self._write(f"  {'-' * col_w} {'-' * 30}")
        all_keys = (
            [(m, "OK") for m in matched]
            + [(f"{s}/{k}", "MISMATCH") for s, k, _, _ in mismatched]
            + [(w_name, "WARN") for w_name in warned]
        )
        all_keys.sort(
            key=lambda x: (
                0 if x[0].startswith("rng") else 1 if x[0].startswith("arg") else 2,
                x[0],
            )
        )
        for name, status in all_keys:
            self._write(f"  {name:<{col_w}} {status}")
        self._write("=" * w)

        # ── Print tensor diffs for mismatches ──
        if mismatched or (warned and self.raise_at_warnings):
            if not mismatched and warned:
                self._write(
                    f"\n {'#' * 80} \nRAISING EXCEPTION BECAUSE RAISE AT WARNINGS WAS SPECIFIED: \n {'#' * 80}"
                )

            self._write(f"\n  TENSOR DIFFS ({len(mismatched)} mismatches):")
            self._write("  " + "-" * (w - 2))
            for section, key, orig, new in mismatched:
                self._print_tensor_diff(section, key, orig, new, indent=4)
            self._write("=" * w)

            # Also dump unmasked memory_indices and memory_frames for context
            self._write("\n  UNMASKED MEMORY DIFFS (for reference):")
            self._write("  " + "-" * (w - 2))
            for section_name, data_new, data_orig in [
                ("arg", args_new, args_original),
                ("ret", res_new, res_original),
            ]:
                for mem_key in ("memory_indices", "memory_frames"):
                    if mem_key in data_new and isinstance(data_new[mem_key], torch.Tensor):
                        orig_t, new_t = data_orig[mem_key], data_new[mem_key]
                        if not torch.equal(orig_t, new_t):
                            self._print_tensor_diff(
                                section_name, f"{mem_key} (unmasked)", orig_t, new_t, indent=4
                            )
                        else:
                            self._write(f"    [{section_name}/{mem_key} (unmasked)] — identical")
            self._write("=" * w)

            # Print traceback up to trainer
            self._write("\n  TRACEBACK (up to trainer):")
            self._write("  " + "-" * (w - 2))
            for frame in traceback.extract_stack():
                self._write(f"    {frame.filename}:{frame.lineno} in {frame.name}")
                self._write(f"      {frame.line}")
                if "trainer" in frame.filename.lower():
                    break
            self._write("=" * w)

            self._flush()
            raise ValueError(
                f"Call {self.call_id} ({f_name}): {len(mismatched)} field(s) differ — "
                f"see logs in {self.output_dir}"
            )

        self._flush()
        self.call_id += 1

    def _flush(self):
        self._verbose.flush()
        self._standard.flush()

    def _print_tensor_diff(
        self, section: str, key: str, orig: torch.Tensor, new: torch.Tensor, indent: int = 4
    ):
        pad = " " * indent
        self._write(f"\n{pad}[{section}/{key}]")
        if orig.shape != new.shape:
            self._write(f"{pad}  shape: orig={orig.shape}  new={new.shape}")
            return
        if orig.dtype != new.dtype:
            self._write(f"{pad}  dtype: orig={orig.dtype}  new={new.dtype}")

        if orig.dtype == torch.bool:
            diff = orig != new
            n_diff = diff.sum().item()
            max_diff = n_diff  # for bools, "max diff" is just count
            self._write(f"{pad}  max absolute difference: {max_diff}")
            self._write(f"{pad}  differing elements: {n_diff} / {orig.numel()}")
        else:
            diff = (orig - new).abs()
            n_diff = (diff > 1e-6).sum().item()
            self._write(f"{pad}  max |diff|: {diff.max().item():.6e}")
            self._write(f"{pad}  differing elements (>1e-6): {n_diff} / {orig.numel()}")

        # First 5 elements
        self._write(f"{pad}  orig (first 5): {orig.flatten()[:5]}")
        self._write(f"{pad}  new  (first 5): {new.flatten()[:5]}")

        # Per-dimension mismatch indices
        if orig.dim() > 1:
            dim0_mask = (diff > 1e-6 if diff.dtype != torch.bool else diff).any(
                dim=tuple(range(1, diff.dim()))
            )
            self._write(
                f"{pad}  dim-0 mismatch indices: {dim0_mask.nonzero(as_tuple=True)[0].tolist()}"
            )
            if diff.dim() > 2:
                dim1_mask = (diff > 1e-6 if diff.dtype != torch.bool else diff).any(
                    dim=tuple(i for i in range(diff.dim()) if i != 1)
                )
                self._write(
                    f"{pad}  dim-1 mismatch indices: {dim1_mask.nonzero(as_tuple=True)[0].tolist()}"
                )

        # Vector with largest total diff
        if orig.dim() > 1:
            per_row_sum = (
                diff.view(orig.shape[0], -1).sum(dim=1)
                if diff.dtype != torch.bool
                else diff.view(orig.shape[0], -1).float().sum(dim=1)
            )
            worst_idx = per_row_sum.argmax().item()
            self._write(f"{pad}  worst row (idx {worst_idx}):")
            self._write(f"{pad}    orig: {orig[worst_idx]}")  # type: ignore
            self._write(f"{pad}    new:  {new[worst_idx]}")  # type: ignore

        # First mismatching vector
        per_row = diff.view(orig.shape[0], -1).any(dim=1) if orig.dim() > 1 else diff
        bad_idx = per_row.nonzero(as_tuple=False)
        if bad_idx.numel() > 0:
            i = bad_idx[0].item()
            self._write(f"{pad}  first mismatch (idx {i}):")
            self._write(f"{pad}    orig: {orig[i]}")  # type: ignore
            self._write(f"{pad}    new:  {new[i]}")  # type: ignore

        # First vector of each tensor
        first_orig = orig[(0,) * (orig.dim() - 1)] if orig.dim() > 0 else orig
        first_new = new[(0,) * (new.dim() - 1)] if new.dim() > 0 else new
        self._write(f"{pad}  first vector orig: {first_orig}")
        self._write(f"{pad}  first vector new:  {first_new}")

        # Full tensors (verbose.log only)
        self._write(f"{pad}  --- full orig tensor ---", verbose_only=True)
        torch.set_printoptions(profile="full")
        self._write(f"{pad}  {orig}", verbose_only=True)
        self._write(f"{pad}  --- full new tensor ---", verbose_only=True)
        self._write(f"{pad}  {new}", verbose_only=True)
        torch.set_printoptions(profile="default")

    def show_tensor_diff(self, key: str, f_name: str, orig: torch.Tensor, new: torch.Tensor):
        print((("#" * 200) + "\n") + "# DIFERENCE IN " + key + f" at {f_name}\n:" + ("#" * 200))
        if orig.shape != new.shape:
            print(f"Shape mismatch: original={orig.shape} new={new.shape}")

        if orig.dtype != new.dtype:
            print(f"Dtype mismatch: original={orig.dtype} new={new.dtype}")

        if orig.dtype == torch.bool:
            diff = orig != new
        else:
            diff = (orig - new).abs()
        max_diff = diff.max().item()
        print(f"Max absolute difference: {max_diff:.6f}")
        print(f"Number of differing elements: {(diff > 1e-6).sum().item()} out of {orig.numel()}")
        print(f"Original tensor (first 5 elements): {orig.flatten()[:5]}")
        print(f"New tensor (first 5 elements): {new.flatten()[:5]}")
        print(
            f"First dimension mismatches at indices: {(diff > 1e-6).any(dim=tuple(range(1, diff.dim()))).nonzero(as_tuple=True)[0]}"
        )
        print(
            f"Second dimension mismatches at indices: {(diff > 1e-6).any(dim=tuple(range(0, diff.dim(), 2))).nonzero(as_tuple=True)[0]}"
        )
        print("-" * 50)
        print(
            f"First vector with max diff orig: {orig[diff.sum(dim=tuple(range(1, diff.dim()))).argmax()]} "
            f"vs new :{new[diff.sum(dim=tuple(range(1, diff.dim()))).argmax()]}"
        )
        print(f"First vector of orig tensor: {orig[(0,) * (orig.dim() - 1)]}")
        print(f"First vector of new tensor: {new[(0,) * (new.dim() - 1)]}")
        print(("-" * 50 + "\n") * 3)
        torch.set_printoptions(profile="full")  # disables truncation
        print(f"orig tensor {orig}")
        print("-" * 50)
        print(f"new tensor {new}")
        torch.set_printoptions(profile="default")  # restore default
